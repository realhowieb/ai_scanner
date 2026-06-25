# ai_scanner/utils/retry.py
from __future__ import annotations

import random
import time
from typing import Callable, Iterable, Type


def with_backoff(fn: Callable, exceptions: Iterable[Type[BaseException]], tries=3, base=0.5, jitter=0.3):
    def wrapped(*a, **kw):
        t = 0
        while True:
            try:
                return fn(*a, **kw)
            except tuple(exceptions):
                t += 1
                if t >= tries:
                    raise
                sleep = base * (2 ** (t-1)) + random.random()*jitter
                time.sleep(sleep)
    return wrapped