# ai_scanner/telemetry.py
from __future__ import annotations
import time
from typing import Callable, Any
from functools import wraps

def timed(save_cb: Callable[[float], None]):
    """
    Decorator to measure and save the execution time of a function.

    Parameters:
    save_cb (Callable[[float], None]): A callback function that accepts the elapsed time in seconds.
    """
    def deco(fn: Callable[..., Any]):
        @wraps(fn)
        def wrapper(*a, **kw) -> Any:
            t0 = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                elapsed = time.perf_counter() - t0
                print(f"Execution time of {fn.__name__}: {elapsed:.6f} seconds")
                save_cb(elapsed)
        return wrapper
    return deco