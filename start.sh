#!/usr/bin/env bash
# تشغيل بضغطة واحدة (Linux / macOS): ينشئ بيئة ويثبّت المكتبات ويشغّل الخادم
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
pip install -q -r requirements.txt
pip install -q -U "yt-dlp[default]"   # يوتيوب بيغيّر نظامه كتير، فنحدّث yt-dlp كل مرة
exec python haram_counter.py "$@"
