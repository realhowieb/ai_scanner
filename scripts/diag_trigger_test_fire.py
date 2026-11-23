"""End-to-end trigger alert check: sets one trigger alert that should fire at once."""
import os
import runpy

os.environ["TRIGGER_TEST_MODE"] = "fire"
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "diag_trigger_test_setup.py"), run_name="__main__")
