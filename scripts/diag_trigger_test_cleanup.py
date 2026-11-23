"""Deletes the trigger alerts made by diag_trigger_test_setup.py (Diagnostics workflow)."""
import os
import runpy

os.environ["TRIGGER_TEST_MODE"] = "cleanup"
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "diag_trigger_test_setup.py"), run_name="__main__")
