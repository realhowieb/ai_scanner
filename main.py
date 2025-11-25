
import sys
from pathlib import Path

# Ensure project root is on sys.path so package imports work
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
if str(_THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent))

# Prefer app.main(); fall back to UI renderer if needed
try:
    from app import main as _run
except Exception:
    try:
        from ui.pages import render_app as _render_app
        def _run():
            _render_app()
    except Exception as e:
        # Last-resort: raise a helpful error so the developer knows what's missing
        raise RuntimeError(
            "Unable to load application entrypoint. Ensure `app.py` defines `main()` "
            "or `ui/pages.py` defines `render_app()`."
        ) from e

# Streamlit executes the script top-to-bottom; just call the entry function.
_run()
