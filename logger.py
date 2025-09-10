# logger.py
import threading

def thread_safe_logger(log_fn):
    """
    Wrap a UI logger (e.g., from new_log_panel) so it can be called from worker threads
    without requiring a ScriptRunContext. If Streamlit isn't available on this thread,
    we just print to stdout. Also guards against exceptions inside the logger.
    """
    if log_fn is None:
        return None

    _lock = threading.Lock()

    def _safe(*args, **kwargs):
        msg = ""
        try:
            parts = []
            for a in args:
                try:
                    parts.append(str(a))
                except Exception:
                    parts.append("<unprintable>")
            msg = " ".join(parts)
        except Exception:
            msg = "<log>"

        try:
            with _lock:
                return log_fn(*args, **kwargs)
        except Exception:
            try:
                print(msg)
            except Exception:
                pass
            return None

    return _safe