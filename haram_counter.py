"""
عدّاد الحرم: يقرأ بث مباشر ويحسب (طائف / واقف / ماشي) + (إحرام / عادي) + (ذكر / أنثى)

التثبيت:
    pip install ultralytics opencv-python yt-dlp transformers torch

التشغيل:
    python haram_counter.py --url "https://www.youtube.com/watch?v=XXXX"
    (أو --url مسار ملف فيديو / رابط RTSP / رابط m3u8)

أول إطار هيظهر: اضغط بالماوس على مركز الكعبة (مركز الدوران) ثم اضغط أي زر.
اضغط q للخروج.
"""
import argparse
import json
import math
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from collections import Counter, defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------- إعدادات يمكن تعديلها ----------------
HISTORY = 45            # عدد الإطارات اللي بنحلل فيها الحركة
TAWAF_MIN_ANGLE = 0.25  # أقل زاوية دوران (راديان) حوالين المركز عشان نعتبره طائف
STAND_MAX_MOVE = 12     # أقصى إزاحة (بكسل) عشان نعتبره واقف
ATTR_EVERY = 20         # نحدّث الجنس/الإحرام كل كام إطار لكل شخص
WHITE_RATIO = 0.45      # نسبة الأبيض في الجذع عشان نعتبره لابس إحرام
YOLO_IMGSZ = 1920       # كبّره لو الناس صغيرة في الصورة (من فوق الناس صغيرة)
TOP_VIEW = True         # الكاميرا من فوق
# ------------------------------------------------------


STATE = {"ts": 0, "total": 0, "fps": 0,
         "motion": {"tawaf": 0, "standing": 0, "walking": 0},
         "ihram": {"ihram": 0, "regular": 0}, "gender": None}


def start_server(port: int):
    """سيرفر صغير بيعرض الأرقام JSON على http://127.0.0.1:PORT/stats للوحة index.html"""
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(STATE).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()


def get_stream_url(url: str) -> str:
    if "youtube.com" in url or "youtu.be" in url:
        out = subprocess.run(
            ["yt-dlp", "-g", "-f", "best[height<=720]/best", url],
            capture_output=True, text=True,
        )
        lines = out.stdout.strip().splitlines()
        if not lines:
            raise RuntimeError("yt-dlp مقدرش يجيب رابط البث:\n" + out.stderr)
        return lines[0]
    return url


class GenderClassifier:
    """تصنيف ذكر/أنثى بـ CLIP zero-shot. الدقة محدودة في اللقطات البعيدة."""

    def __init__(self):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        name = "openai/clip-vit-base-patch32"
        self.model = CLIPModel.from_pretrained(name).eval()
        self.proc = CLIPProcessor.from_pretrained(name)
        self.prompts = ["a photo of a man", "a photo of a woman"]

    def __call__(self, crop_bgr):
        img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.proc(text=self.prompts, images=img,
                           return_tensors="pt", padding=True)
        with self.torch.no_grad():
            logits = self.model(**inputs).logits_per_image
        return "ذكر" if int(logits.argmax()) == 0 else "أنثى"


def is_ihram(crop_bgr) -> bool:
    """إحرام الرجال = قطعتين بيضاء. بنقيس نسبة الأبيض في منطقة الجذع."""
    h, w = crop_bgr.shape[:2]
    if TOP_VIEW:  # من فوق: بنقيس منتصف الصندوق (الكتفين/الرأس)
        torso = crop_bgr[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)]
    else:
        torso = crop_bgr[int(h * 0.15):int(h * 0.65)]
    if torso.size == 0:
        return False
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    white = (hsv[..., 1] < 45) & (hsv[..., 2] > 150)
    return white.mean() > WHITE_RATIO


def classify_motion(track, center):
    """طائف / واقف / ماشي، بناءً على الدوران حوالين مركز الكعبة."""
    if len(track) < HISTORY // 2:
        return None
    pts = np.array(track, dtype=float)
    if np.linalg.norm(pts[-1] - pts[0]) < STAND_MAX_MOVE and \
            np.ptp(pts, axis=0).max() < STAND_MAX_MOVE * 2:
        return "واقف"
    ang = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    d = np.diff(np.unwrap(ang))
    total = d.sum()
    consistent = abs(total) / (np.abs(d).sum() + 1e-9)  # قريب من 1 = اتجاه ثابت
    if abs(total) > TAWAF_MIN_ANGLE and consistent > 0.7:
        return "طائف"
    return "ماشي"


def pick_center(frame):
    pt = []

    def cb(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            pt[:] = [x, y]

    cv2.namedWindow("pick center")
    cv2.setMouseCallback("pick center", cb)
    cv2.imshow("pick center", frame)
    cv2.waitKey(0)
    cv2.destroyWindow("pick center")
    return tuple(pt) if pt else (frame.shape[1] // 2, frame.shape[0] // 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="yolov8m.pt")
    ap.add_argument("--gender", action="store_true",
                    help="تفعيل تصنيف الجنس (غير موثوق من الكاميرات العلوية)")
    ap.add_argument("--port", type=int, default=8765, help="بورت لوحة التحكم")
    args = ap.parse_args()

    cap = cv2.VideoCapture(get_stream_url(args.url))
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("مقدرتش أقرأ البث")
    center = pick_center(frame)

    model = YOLO(args.model)
    gender_clf = GenderClassifier() if args.gender else None
    start_server(args.port)
    t_prev = time.time()

    tracks = defaultdict(lambda: deque(maxlen=HISTORY))
    gender_votes = defaultdict(lambda: deque(maxlen=7))
    ihram_votes = defaultdict(lambda: deque(maxlen=7))
    last_attr = {}
    n = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        res = model.track(frame, persist=True, classes=[0], imgsz=YOLO_IMGSZ,
                          tracker="bytetrack.yaml", verbose=False)[0]
        if res.boxes.id is None:
            continue

        stats = Counter()
        for box, tid in zip(res.boxes.xyxy.cpu().numpy(),
                            res.boxes.id.int().cpu().tolist()):
            x1, y1, x2, y2 = map(int, box)
            tracks[tid].append(((x1 + x2) / 2, (y1 + y2) / 2))

            if n - last_attr.get(tid, -999) >= ATTR_EVERY:
                crop = frame[max(y1, 0):y2, max(x1, 0):x2]
                if crop.shape[0] >= 12 and crop.shape[1] >= 12:
                    ihram_votes[tid].append(is_ihram(crop))
                    if gender_clf and crop.shape[0] >= 40:
                        gender_votes[tid].append(gender_clf(crop))
                    last_attr[tid] = n

            motion = classify_motion(tracks[tid], center) or "؟"
            ih = ("إحرام" if sum(ihram_votes[tid]) > len(ihram_votes[tid]) / 2
                  else "عادي") if ihram_votes[tid] else "؟"
            gd = (Counter(gender_votes[tid]).most_common(1)[0][0]
                  if gender_votes[tid] else "؟")

            stats[motion] += 1
            stats[ih] += 1
            stats[gd] += 1
            color = {"طائف": (0, 200, 0), "واقف": (0, 0, 255),
                     "ماشي": (0, 200, 255)}.get(motion, (200, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        cv2.circle(frame, center, 6, (255, 0, 0), -1)
        total = len(res.boxes.id)
        now = time.time()
        fps = 1 / (now - t_prev + 1e-9)
        t_prev = now
        STATE.update(
            ts=now, total=total, fps=round(fps, 1),
            motion={"tawaf": stats["طائف"], "standing": stats["واقف"],
                    "walking": stats["ماشي"]},
            ihram={"ihram": stats["إحرام"], "regular": stats["عادي"]},
            gender=({"male": stats["ذكر"], "female": stats["أنثى"]}
                    if gender_clf else None),
        )
        print(f"\rإجمالي={total} | طائف={stats['طائف']} واقف={stats['واقف']} "
              f"ماشي={stats['ماشي']} | إحرام={stats['إحرام']} عادي={stats['عادي']} "
              f"| ذكور={stats['ذكر']} إناث={stats['أنثى']}", end="")
        cv2.imshow("haram", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
