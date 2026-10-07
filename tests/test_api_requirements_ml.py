"""hsf-api must install the PreBreakout model's libraries.

Without them ml_prebreakout.load_prebreakout_model returns None and
score_prebreakout writes 0.0 to every row, so custom scans showed Premium
users PreBreakout 0% on every ticker.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _includes(path: Path) -> list[str]:
    return [line.split("#", 1)[0].strip() for line in path.read_text().splitlines()]


def test_api_requirements_include_ml_group():
    assert "-r ../requirements-ml.txt" in _includes(ROOT / "api" / "requirements.txt")


def test_ml_group_has_model_libraries():
    names = {line.split(">")[0].split("=")[0].split("<")[0]
             for line in _includes(ROOT / "requirements-ml.txt") if line}
    assert {"xgboost", "scikit-learn", "joblib"} <= names
