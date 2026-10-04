#!/bin/bash
# Claude Code on the web: install the app's Python dependencies so the full
# test suite and ruff work (without this, ~23 app-boot tests fail on a missing
# bcrypt). Same files CI installs (.github/workflows/smoke.yml).
#
# requirements.lock is resolved for Python 3.13 (CI / Streamlit Cloud). If the
# session's Python can't satisfy it (e.g. 3.11: numpy 2.5 needs 3.12+), fall
# back to the ranged requirement files.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}"
PIP=(python3 -m pip install --quiet --disable-pip-version-check --prefer-binary)
RANGED=(-r requirements-core.txt -r requirements-ml.txt -r requirements-extended.txt
        -r requirements-billing-test.txt)

if ! "${PIP[@]}" -c requirements.lock "${RANGED[@]}" 2>/tmp/session-start-pip.log; then
  echo "session-start: lock not installable on $(python3 --version 2>&1); using ranged requirements" >&2
  "${PIP[@]}" "${RANGED[@]}"
fi
