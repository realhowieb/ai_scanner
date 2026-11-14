"""Lean PreBreakout scoring import (HSF_LEAN_MODEL_IMPORT=1): xgboost without scikit-learn,
same predictions. Runs in subprocesses because the test process already has sklearn loaded."""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("xgboost")
pytest.importorskip("sklearn")

ROOT = Path(__file__).resolve().parents[1]


def _run(code: str, lean: str) -> str:
    env = {**os.environ, "HSF_LEAN_MODEL_IMPORT": lean, "PYTHONPATH": str(ROOT)}
    out = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], cwd=ROOT, env=env,
                         capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout.strip().splitlines()[-1]


def test_lean_scoring_skips_sklearn_and_scores_identically(tmp_path):
    model = tmp_path / "model.pkl"
    _run(f"""
        import numpy as np, joblib, ml_prebreakout as m
        m._load_ml_libs()
        rng = np.random.default_rng(0)
        X = rng.normal(size=(600, 6)); y = (X[:, 0] + rng.normal(size=600) > 0.3).astype(int)
        clf = m._new_prebreakout_classifier(); clf.set_params(n_estimators=40); clf.fit(X, y)
        joblib.dump({{"model": clf, "features": list("abcdef")}}, {str(model)!r})
        np.save({str(tmp_path / 'X.npy')!r}, X[:50])
        print("ok")
    """, "0")
    code = f"""
        import sys, numpy as np, ml_prebreakout as m
        m.load_latest_prebreakout_model_bundle = None
        from analytics import prediction_provenance as pp
        import joblib as jl
        pp.load_local = lambda joblib_module, path: (jl.load(path), "test")
        bundle = m.load_prebreakout_model({str(model)!r})
        p = bundle["model"].predict_proba(np.load({str(tmp_path / 'X.npy')!r}))[:, 1]
        print("sklearn" in sys.modules, "xgboost" in sys.modules, ",".join(repr(float(v)) for v in p))
    """
    lean = _run(code, "1").split(" ", 2)
    full = _run(code, "0").split(" ", 2)
    assert lean[:2] == ["False", "True"]
    assert full[:2] == ["True", "True"]
    assert lean[2] == full[2]


def test_lean_import_leaves_sklearn_importable_afterwards():
    line = _run("""
        import ml_prebreakout as m
        m._load_scoring_libs()
        import sklearn.metrics, xgboost
        print("sklearn" in sys.modules if (sys := __import__("sys")) else None, m.joblib is not None)
    """, "1")
    assert line == "True True"
