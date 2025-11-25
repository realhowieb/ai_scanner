import sys
from pathlib import Path

# Ensure project root is on sys.path
THIS_DIR = Path(__file__).resolve().parent
PARENT = THIS_DIR.parent

if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

# Clean entrypoint for Streamlit / Python
try:
    from app import main as run_app
except Exception as e:
    raise RuntimeError(
        "Unable to load app.main(). Ensure app.py defines main()."
    ) from e

if __name__ == "__main__":
    run_app()
