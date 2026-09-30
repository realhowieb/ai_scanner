#!/usr/bin/env bash
# P2-38 — rebuild the local development environment in one command.
#
#   bash scripts/setup_local_env.sh          # create .venv (keeps an existing one)
#   bash scripts/setup_local_env.sh --fresh  # delete and rebuild .venv
#
# Installs the same pinned versions production runs (requirements.lock, frozen
# on Python 3.13 by the Freeze Dependency Lock workflow), minus Linux-only GPU
# wheels that don't exist on macOS. Never edits requirements.lock.
set -euo pipefail

cd "$(dirname "$0")/.."
PY_VERSION="$(cat .python-version 2>/dev/null || echo 3.13)"
PY="${PYTHON:-python${PY_VERSION}}"

if ! command -v "$PY" >/dev/null 2>&1; then
  echo "Python ${PY_VERSION} not found (looked for '$PY')." >&2
  echo "Install it (macOS: brew install python@${PY_VERSION}) or set PYTHON=/path/to/python${PY_VERSION}." >&2
  exit 1
fi

if [[ "${1:-}" == "--fresh" && -d .venv ]]; then
  echo "Removing existing .venv"
  rm -rf .venv
fi

if [[ ! -x .venv/bin/python ]]; then
  echo "Creating .venv with $("$PY" --version)"
  "$PY" -m venv .venv
fi

LOCK_LOCAL="$(mktemp -t hsf-lock.XXXXXX)"
trap 'rm -f "$LOCK_LOCAL"' EXIT
# nvidia-* (and triton) are Linux-only CUDA wheels pulled in by xgboost on Linux.
grep -v -E '^(nvidia-|triton==)' requirements.lock > "$LOCK_LOCAL"

.venv/bin/python -m pip install --upgrade --quiet pip
echo "Installing pinned dependencies from requirements.lock"
.venv/bin/python -m pip install --quiet -r "$LOCK_LOCAL"
echo "Installing developer tools"
.venv/bin/python -m pip install --quiet -r requirements-dev.txt
.venv/bin/python -m pip check

if [[ ! -f .streamlit/secrets.toml ]]; then
  echo
  echo "No .streamlit/secrets.toml yet: copy .streamlit/secrets.toml.example and fill it in."
fi

echo
echo "Done: $(.venv/bin/python --version) in .venv"
echo "Run the app:   .venv/bin/streamlit run app.py"
echo "Run the tests: .venv/bin/python -m pytest -q"
