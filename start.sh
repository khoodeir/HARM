#!/usr/bin/env bash
# تشغيل بضغطة واحدة (Linux / macOS): ينشئ بيئة ويثبّت المكتبات ويشغّل الخادم
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
pip install -q -r requirements.txt
exec python haram_counter.py "$@"
