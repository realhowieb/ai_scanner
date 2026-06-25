# ai_scanner/utils/market_time.py
from __future__ import annotations

import datetime as dt

import pytz

NY = pytz.timezone("America/New_York")

def now_ny():
    return dt.datetime.now(NY)

def is_premarket(ts: dt.datetime|None=None):
    ts = ts or now_ny()
    return ts.weekday() < 5 and ts.time() >= dt.time(4,0) and ts.time() < dt.time(9,30)

def is_regular(ts: dt.datetime|None=None):
    ts = ts or now_ny()
    return ts.weekday() < 5 and dt.time(9,30) <= ts.time() < dt.time(16,0)

def is_postmarket(ts: dt.datetime|None=None):
    ts = ts or now_ny()
    return ts.weekday() < 5 and dt.time(16,0) <= ts.time() < dt.time(20,0)