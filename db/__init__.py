"""DB package initializer."""
try:
    from .runs import list_runs, save_run, load_run_results  # type: ignore
except Exception:
    pass