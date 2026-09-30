#!/usr/bin/env bash
# P2-37 — refresh the LOCAL Neon branch "dev-local" with a fresh copy of "live".
#
#   bash scripts/reset_dev_branch.sh
#
# Throws away everything written to dev-local and copies live's current data.
# Only ever resets dev-local: the branch name is fixed below on purpose. Never
# run "reset from parent" on live itself (its parent holds old data; see
# docs/PROVIDER_FAILOVER_AND_RESTORE.md). Needs the Neon CLI signed in
# (npx neon@latest auth). The connection string does not change.
set -euo pipefail

BRANCH="dev-local"
PROJECT_ID="${NEON_PROJECT_ID:?Set NEON_PROJECT_ID (Neon console -> Settings)}"

parent="$(npx -y neon@latest branches get "$BRANCH" --project-id "$PROJECT_ID" --output json \
  | python3 -c 'import json,sys; print(json.load(sys.stdin).get("parent_id",""))')"
parent_name="$(npx -y neon@latest branches get "$parent" --project-id "$PROJECT_ID" --output json \
  | python3 -c 'import json,sys; print(json.load(sys.stdin).get("name",""))')"
if [[ "$parent_name" != "live" ]]; then
  echo "Refusing: $BRANCH's parent is '$parent_name', expected 'live'." >&2
  exit 1
fi

read -r -p "Reset $BRANCH to a fresh copy of live? Local changes on $BRANCH are lost. [y/N] " ok
[[ "$ok" == "y" || "$ok" == "Y" ]] || { echo "Cancelled."; exit 0; }

npx -y neon@latest branches reset "$BRANCH" --parent --project-id "$PROJECT_ID"
echo "Done: $BRANCH now matches live as of $(date -u +%Y-%m-%dT%H:%MZ)."
