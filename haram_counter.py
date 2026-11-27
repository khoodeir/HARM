"""
منظومة رصد وإحصاء المطاف (الخادم).

يشتغل لوحده: يقرأ روابط البث من streams.json، ويعايِر مركز الكعبة تلقائياً،
ويقدّم اللوحة (index.html) والفيديو المحلَّل والأرقام عبر HTTP.
الروابط تُضاف وتُحذف من إعدادات اللوحة نفسها.

التشغيل:
    python haram_counter.py
خيارات: --port 8765  --host 127.0.0.1|0.0.0.0  --token سر_الإدارة  --device cuda:0
        --url رابط --name اسم     (اختياري، تتجاوز streams.json)
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------- إعدادات يمكن تعديلها ----------------
HISTORY = 45            # عدد الإطارات اللي بنحلل فيها الحركة
TAWAF_MIN_ANGLE = 0.25  # أقل زاوية دوران (راديان) حوالين المركز عشان نعتبره طائف
STAND_MAX_MOVE = 12     # أقصى إزاحة (بكسل) عشان نعتبره واقف
ATTR_EVERY = 20         # نحدّث الجنس/الإحرام كل كام إطار لكل شخص
WHITE_RATIO = 0.45      # نسبة الأبيض في الجذع عشان نعتبره لابس إحرام
TOP_VIEW = True         # الكاميرا من فوق
JPEG_WIDTH = 1280       # أقصى عرض للصورة المعروضة في اللوحة
CALIB_MIN_POINTS = 800  # أقل عدد نقاط حركة قبل المعايرة التلقائية
# ألوان الصناديق على الفيديو (BGR)
COLORS = {"طائف": (91, 125, 46), "واقف": (9, 83, 180),
          "ماشي": (170, 170, 170), "؟": (210, 210, 210)}
# ------------------------------------------------------

HERE = Path(__file__).resolve().parent
INDEX_FILE = HERE / "index.html"
CONFIG_FILE = HERE / "streams.json"
ARGS = None
CAMS = {}
CAMS_LOCK = threading.Lock()
_next_id = [0]


class Camera:
    def __init__(self, idx, name, url, center=None):
        self.idx, self.name, self.url = idx, name, url
        self.auto = center is None             # معايرة تلقائية للمركز
        self.center = tuple(center) if center else (0.5, 0.5)
        self.calibrated = not self.auto
        self.pts = deque(maxlen=6000)
        self.stop = False
        self.error = ""                          # آخر خطأ (يظهر في اللوحة)
        self.cond = threading.Condition()
        self.jpeg, self.seq = None, 0
        self.state = self._base("starting")

    def _base(self, status):
        return {"name": self.name, "status": status, "ts": 0, "total": 0, "fps": 0,
                "center": list(self.center),
                "center_mode": "auto" if self.auto else "manual",
                "calibrated": self.calibrated, "error": self.error,
                "motion": {"tawaf": 0, "standing": 0, "walking": 0},
                "ihram": {"ihram": 0, "regular": 0}, "gender": None}

    def set_status(self, status):
        with self.cond:
            self.state = dict(self.state, status=status, error=self.error)

    def publish(self, frame, stats, total, fps, has_gender):
        h, w = frame.shape[:2]
        small = frame if w <= JPEG_WIDTH else cv2.resize(
            frame, (JPEG_WIDTH, int(h * JPEG_WIDTH / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 72])
        self.error = ""
        state = self._base("live")
        state.update(
            ts=time.time(), total=total, fps=round(fps, 1),
            motion={"tawaf": stats["طائف"], "standing": stats["واقف"],
                    "walking": stats["ماشي"]},
            ihram={"ihram": stats["إحرام"], "regular": stats["عادي"]},
            gender=({"male": stats["ذكر"], "female": stats["أنثى"]}
                    if has_gender else None))
        with self.cond:
            self.state = state
            if ok:
                self.jpeg = buf.tobytes()
                self.seq += 1
            self.cond.notify_all()


# ---------------- إدارة البثوث والإعدادات ----------------
def save_config():
    with CAMS_LOCK:
        data = {"streams": [
            {"name": c.name, "url": c.url,
             "center": None if c.auto else list(c.center)}
            for c in CAMS.values()]}
    CONFIG_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                           encoding="utf-8")


def load_config():
    try:
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8")).get("streams", [])
    except (OSError, ValueError):
        return []


def add_camera(name, url, center=None, save=True):
    with CAMS_LOCK:
        cam = Camera(_next_id[0], name, url, center)
        _next_id[0] += 1
        CAMS[cam.idx] = cam
    threading.Thread(target=run_camera, args=(cam,), daemon=True).start()
    if save:
        save_config()
    return cam


def remove_camera(idx):
    with CAMS_LOCK:
        cam = CAMS.pop(idx, None)
    if cam:
        cam.stop = True
        with cam.cond:
            cam.cond.notify_all()
        save_config()
    return cam is not None


# ---------------- التحليل ----------------
def get_stream_url(url: str) -> str:
    if url.startswith("http") and ("youtube.com" in url or "youtu.be" in url):
        m = re.search(r"youtube\.com/live/([\w-]{6,})", url)
        if m:                                   # نشيل باراميترات المشاركة (?si=...)
            url = f"https://www.youtube.com/watch?v={m.group(1)}"
        cmd = [sys.executable, "-m", "yt_dlp", "-g", "-f", "best[height<=720]/best",
               "--no-warnings"]
        if ARGS.cookies_browser:
            cmd += ["--cookies-from-browser", ARGS.cookies_browser]
        if ARGS.cookies:
            cmd += ["--cookies", ARGS.cookies]
        if ARGS.js_runtimes:
            cmd += ["--js-runtimes", ARGS.js_runtimes]
        out = subprocess.run(cmd + [url], capture_output=True, text=True)
        lines = out.stdout.strip().splitlines()
        if not lines:
            err = [l for l in out.stderr.strip().splitlines() if l.strip()]
            raise RuntimeError("yt-dlp فشل: " + (err[-1] if err else "بدون رسالة")[:300])
        return lines[0]
    return url


class GenderClassifier:
    """تصنيف ذكر/أنثى بـ CLIP zero-shot. الدقة محدودة جداً من الكاميرات العلوية."""

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
    """إحرام الرجال = قطعتين بيضاء. بنقيس نسبة الأبيض في منتصف الصندوق."""
    h, w = crop_bgr.shape[:2]
    if TOP_VIEW:
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
    consistent = abs(total) / (np.abs(d).sum() + 1e-9)
    if abs(total) > TAWAF_MIN_ANGLE and consistent > 0.7:
        return "طائف"
    return "ماشي"


def fit_circle(pts):
    """أقل مربعات (Kåsa): أفضل دائرة تمر بنقاط حركة الطائفين → مركزها = الكعبة."""
    x, y = pts[:, 0], pts[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    sol = np.linalg.lstsq(A, x ** 2 + y ** 2, rcond=None)[0]
    cx, cy = sol[0], sol[1]
    r = math.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 0))
    d = np.hypot(x - cx, y - cy)
    return cx, cy, r, (d.std() / r if r else 9)


def auto_calibrate(cam, w, h):
    if len(cam.pts) < CALIB_MIN_POINTS:
        return
    cx, cy, r, spread = fit_circle(np.array(cam.pts, dtype=float))
    if not (0.08 * w <= r <= 0.9 * max(w, h) and spread < 0.5
            and -0.5 * w <= cx <= 1.5 * w and -0.5 * h <= cy <= 1.5 * h):
        return
    new = (cx / w, cy / h)
    cam.center = new if not cam.calibrated else (
        0.7 * cam.center[0] + 0.3 * new[0], 0.7 * cam.center[1] + 0.3 * new[1])
    cam.calibrated = True


def process(cam, cap, model, clf):
    tracks = defaultdict(lambda: deque(maxlen=HISTORY))
    gender_votes = defaultdict(lambda: deque(maxlen=7))
    ihram_votes = defaultdict(lambda: deque(maxlen=7))
    last_attr, last_seen = {}, {}
    n, fails, t_prev = 0, 0, time.time()
    kw = {"device": ARGS.device} if ARGS.device else {}

    while not cam.stop:
        ok, frame = cap.read()
        if not ok:
            fails += 1
            if fails > 50:
                raise RuntimeError("انقطع البث")
            time.sleep(0.1)
            continue
        fails = 0
        n += 1
        h, w = frame.shape[:2]
        center = (cam.center[0] * w, cam.center[1] * h)

        res = model.track(frame, persist=True, classes=[0], imgsz=ARGS.imgsz,
                          tracker="bytetrack.yaml", verbose=False, **kw)[0]
        stats, total, draws = Counter(), 0, []

        if res.boxes.id is not None:
            ids = res.boxes.id.int().cpu().tolist()
            boxes = res.boxes.xyxy.cpu().numpy()
            total = len(ids)
            for box, tid in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)
                tracks[tid].append(((x1 + x2) / 2, (y1 + y2) / 2))
                last_seen[tid] = n

                if n - last_attr.get(tid, -999) >= ATTR_EVERY:
                    crop = frame[max(y1, 0):y2, max(x1, 0):x2]
                    if crop.shape[0] >= 12 and crop.shape[1] >= 12:
                        ihram_votes[tid].append(is_ihram(crop))
                        if clf and crop.shape[0] >= 40:
                            gender_votes[tid].append(clf(crop))
                        last_attr[tid] = n

                motion = classify_motion(tracks[tid], center) or "؟"
                ih = ("إحرام" if sum(ihram_votes[tid]) > len(ihram_votes[tid]) / 2
                      else "عادي") if ihram_votes[tid] else "؟"
                gd = (Counter(gender_votes[tid]).most_common(1)[0][0]
                      if gender_votes[tid] else "؟")
                stats[motion] += 1
                stats[ih] += 1
                stats[gd] += 1
                draws.append((x1, y1, x2, y2, motion))

                # نقاط الحركة الواضحة تُستخدم لمعايرة مركز الكعبة تلقائياً
                tr = tracks[tid]
                if cam.auto and n % 3 == 0 and len(tr) >= HISTORY and \
                        math.dist(tr[0], tr[-1]) > 3 * STAND_MAX_MOVE:
                    cam.pts.append(tr[-1])

        if cam.auto and n % 150 == 0:
            auto_calibrate(cam, w, h)

        for x1, y1, x2, y2, motion in draws:
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[motion], 1)
        cx, cy = int(center[0]), int(center[1])
        r = max(10, min(h, w) // 40)
        for col, t in (((0, 0, 0), 3), ((255, 255, 255), 1)):
            cv2.line(frame, (cx - r, cy), (cx + r, cy), col, t)
            cv2.line(frame, (cx, cy - r), (cx, cy + r), col, t)

        if n % 300 == 0:   # تنظيف المسارات القديمة
            for tid in [t for t, s in last_seen.items() if s < n - 300]:
                for d in (tracks, gender_votes, ihram_votes, last_attr, last_seen):
                    d.pop(tid, None)

        now = time.time()
        fps, t_prev = 1 / (now - t_prev + 1e-9), now
        cam.publish(frame, stats, total, fps, clf is not None)


def run_camera(cam):
    clf = GenderClassifier() if ARGS.gender else None
    while not cam.stop:
        try:
            model = YOLO(ARGS.model)       # نموذج جديد = تتبع جديد بعد كل إعادة اتصال
            cap = cv2.VideoCapture(get_stream_url(cam.url))
            if not cap.isOpened():
                raise RuntimeError("تعذّر فتح البث")
            process(cam, cap, model, clf)
        except Exception as e:  # noqa
            cam.error = str(e)[:300]
            print(f"[{cam.name}] {cam.error}")
        if not cam.stop:
            cam.set_status("reconnecting")
            time.sleep(3)


# ---------------- HTTP ----------------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")

    def _send(self, body: bytes, ctype: str, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(json.dumps(obj, ensure_ascii=False).encode(),
                   "application/json; charset=utf-8", code)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.end_headers()

    def _admin_ok(self, q):
        if ARGS.token and q.get("token", [""])[0] != ARGS.token:
            self._json({"error": "forbidden: رمز الإدارة مطلوب أو خاطئ"}, 403)
            return False
        return True

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        p = u.path

        if p in ("/", "/index.html"):
            if INDEX_FILE.exists():
                return self._send(INDEX_FILE.read_bytes(), "text/html; charset=utf-8")
            return self._send(b"index.html not found", "text/plain", 404)
        if p == "/api/health":
            return self._json({"ok": True, "streams": len(CAMS),
                               "admin_token_required": bool(ARGS.token)})
        if p == "/api/cams":
            return self._json([{"id": c.idx, "name": c.name} for c in CAMS.values()])
        if p == "/api/streams":
            return self._json([{"id": c.idx, "name": c.name, "url": c.url,
                                "center_mode": "auto" if c.auto else "manual",
                                "calibrated": c.calibrated} for c in CAMS.values()])

        if p == "/api/streams/add":
            if not self._admin_ok(q):
                return
            url = q.get("url", [""])[0].strip()
            name = q.get("name", [""])[0].strip() or f"بث {_next_id[0] + 1}"
            pu = urlparse(url)
            if pu.scheme not in ("http", "https", "rtsp", "rtmp") or not pu.netloc:
                return self._json({"error": "رابط غير صالح (http/https/rtsp/rtmp)"}, 400)
            cam = add_camera(name, url)
            return self._json({"ok": True, "id": cam.idx})
        if p == "/api/streams/remove":
            if not self._admin_ok(q):
                return
            try:
                ok = remove_camera(int(q["id"][0]))
            except (KeyError, ValueError, IndexError):
                ok = False
            return self._json({"ok": ok}, 200 if ok else 404)

        try:
            cam = CAMS.get(int(q["cam"][0])) if "cam" in q else next(iter(CAMS.values()), None)
        except (ValueError, IndexError):
            cam = None
        if cam is None:
            return self._json({"error": "camera not found"}, 404)

        if p == "/api/stats":
            with cam.cond:
                return self._json(cam.state)
        if p == "/api/center":
            if not self._admin_ok(q):
                return
            try:
                x = min(1.0, max(0.0, float(q["x"][0])))
                y = min(1.0, max(0.0, float(q["y"][0])))
            except (KeyError, ValueError, IndexError):
                return self._json({"error": "bad x/y"}, 400)
            cam.center, cam.auto, cam.calibrated = (x, y), False, True
            save_config()
            return self._json({"ok": True, "center": [x, y]})
        if p == "/api/center_auto":
            if not self._admin_ok(q):
                return
            cam.auto, cam.calibrated = True, False
            cam.pts.clear()
            save_config()
            return self._json({"ok": True})
        if p == "/video":
            return self._stream(cam)
        self._json({"error": "not found"}, 404)

    def _stream(self, cam):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self._cors()
        self.end_headers()
        last = -1
        try:
            while not cam.stop:
                with cam.cond:
                    cam.cond.wait_for(lambda: cam.seq != last or cam.stop, timeout=5)
                    data, last = cam.jpeg, cam.seq
                if data is None or cam.stop:
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                 b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n")
                self.wfile.write(data)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", action="append", default=[],
                    help="رابط بث (اختياري، يتجاوز streams.json)")
    ap.add_argument("--name", action="append", default=[])
    ap.add_argument("--model", default="yolov8m.pt")
    ap.add_argument("--imgsz", type=int, default=1920,
                    help="دقة الكشف، كبّرها لو الناس صغيرة")
    ap.add_argument("--device", default=None, help="مثلاً cuda:0 أو cpu")
    ap.add_argument("--gender", action="store_true",
                    help="تفعيل تصنيف الجنس (غير موثوق من الكاميرات العلوية)")
    ap.add_argument("--cookies-from-browser", dest="cookies_browser", default=None,
                    help="لو يوتيوب طلب تسجيل الدخول: chrome أو edge أو firefox")
    ap.add_argument("--cookies", default=None, help="ملف cookies.txt بديل")
    ap.add_argument("--js-runtimes", dest="js_runtimes", default=None,
                    help="مثلاً node لو yt-dlp طلب مشغّل جافاسكربت")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8765)))
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    ap.add_argument("--token", default=os.environ.get("HARAM_TOKEN", ""),
                    help="رمز يحمي إضافة/حذف البثوث وتحديد المركز")
    ARGS = ap.parse_args()

    if ARGS.url:
        for i, url in enumerate(ARGS.url):
            add_camera(ARGS.name[i] if i < len(ARGS.name) else f"بث {i + 1}",
                       url, save=False)
    else:
        for s in load_config():
            add_camera(s.get("name") or "بث", s["url"], s.get("center"), save=False)

    srv = ThreadingHTTPServer((ARGS.host, ARGS.port), Handler)
    srv.daemon_threads = True
    print(f"اللوحة: http://{ARGS.host}:{ARGS.port}   |   البثوث: {len(CAMS)}")
    if ARGS.host not in ("127.0.0.1", "localhost") and not ARGS.token:
        print("تنبيه: الخادم مفتوح على الشبكة بدون --token، أي شخص يقدر يضيف أو يحذف بثوث.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
