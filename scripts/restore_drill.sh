#!/usr/bin/env bash
# Database restore drill: proves the live database can be copied out and restored, and
# times it. Read-only against production.
#
# 1. Lists table sizes (metadata only).
# 2. pg_dump of the whole schema, with the rows of every table except very large research
#    tables (bigger than BIG_TABLE_MB, unless FULL=1). Accounts, billing, watchlists, alerts,
#    journals and policy acceptances are small, so they are always included.
# 3. pg_restore into a throwaway local Postgres (RESTORE_URL).
# 4. Compares row counts table by table and prints a summary. Never prints row contents.
#
# Nothing is kept: the dump lives in a temp dir that is deleted on exit, so no customer data
# leaves the runner. Neon counts the dump as network transfer (free plan: 5 GB a month), so
# the size is checked against MAX_DUMP_MB before anything is downloaded.
#
# Env: SOURCE_URL (production, read-only use), RESTORE_URL (empty local database),
#      BIG_TABLE_MB (default 50), MAX_DUMP_MB (default 300), FULL (default 0).
set -euo pipefail

: "${SOURCE_URL:?SOURCE_URL is required}"
: "${RESTORE_URL:?RESTORE_URL is required}"
# Neon's pooled host (…-pooler.…) doesn't suit pg_dump; use the direct host of the same branch.
SOURCE_URL="${SOURCE_URL/-pooler./.}"
BIG_TABLE_MB="${BIG_TABLE_MB:-50}"
MAX_DUMP_MB="${MAX_DUMP_MB:-300}"
FULL="${FULL:-0}"
OUT="${GITHUB_STEP_SUMMARY:-/dev/null}"

case "$RESTORE_URL" in
  *@localhost[:/]*|*@127.0.0.1[:/]*) ;;
  *) echo "::error::RESTORE_URL must be a local throwaway database (localhost)"; exit 1 ;;
esac

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
now() { date +%s; }
q() { psql "$SOURCE_URL" -X -A -t -v ON_ERROR_STOP=1 -c "$1"; }

echo "== Server"
q "SHOW server_version" | sed 's/^/source: Postgres /'

echo "== Table sizes (public schema)"
q "SELECT c.relname || '|' || pg_total_relation_size(c.oid) FROM pg_class c
   JOIN pg_namespace n ON n.oid = c.relnamespace
   WHERE n.nspname = 'public' AND c.relkind IN ('r', 'p') ORDER BY 1" > "$work/sizes"
big_bytes=$((BIG_TABLE_MB * 1024 * 1024))
exclude=()
kept_bytes=0
skipped=()
while IFS='|' read -r name bytes; do
  [ -z "$name" ] && continue
  if [ "$FULL" != "1" ] && [ "$bytes" -gt "$big_bytes" ]; then
    exclude+=("--exclude-table-data=public.\"$name\"")
    skipped+=("$name")
  else
    kept_bytes=$((kept_bytes + bytes))
  fi
  printf '%-45s %10.1f MB\n' "$name" "$(echo "$bytes / 1048576" | bc -l)"
done < "$work/sizes"
kept_mb=$((kept_bytes / 1024 / 1024))
echo "rows will be copied for tables totalling about ${kept_mb} MB on disk"
if [ "${#skipped[@]}" -gt 0 ]; then
  echo "schema only (rows skipped, bigger than ${BIG_TABLE_MB} MB): ${skipped[*]}"
fi
if [ "$kept_mb" -gt "$MAX_DUMP_MB" ]; then
  echo "::error::About ${kept_mb} MB would be downloaded, over MAX_DUMP_MB=${MAX_DUMP_MB}. Lower BIG_TABLE_MB or raise MAX_DUMP_MB."
  exit 1
fi

echo "== Dump"
t0=$(now)
pg_dump "$SOURCE_URL" --format=custom --no-owner --no-privileges --schema=public "${exclude[@]}" -f "$work/hsf.dump"
t1=$(now)
dump_mb=$(du -m "$work/hsf.dump" | cut -f1)
echo "dump: ${dump_mb} MB in $((t1 - t0)) s"

echo "== Restore"
psql "$RESTORE_URL" -X -q -v ON_ERROR_STOP=1 -c "DROP SCHEMA IF EXISTS public CASCADE"
pg_restore --dbname="$RESTORE_URL" --no-owner --no-privileges --exit-on-error "$work/hsf.dump"
t2=$(now)
echo "restore: $((t2 - t1)) s"

echo "== Row counts"
mismatch=0
tables=0
checked=0
printf '| Table | Live rows | Restored rows | Match |\n|---|---:|---:|---|\n' > "$work/table.md"
while IFS='|' read -r name _bytes; do
  [ -z "$name" ] && continue
  tables=$((tables + 1))
  if printf '%s\n' "${skipped[@]:-}" | grep -qx "$name"; then
    restored_schema=$(psql "$RESTORE_URL" -X -A -t -c "SELECT to_regclass('public.\"$name\"') IS NOT NULL")
    printf '| %s | (rows skipped) | schema only | %s |\n' "$name" "$([ "$restored_schema" = t ] && echo "schema ok" || echo "MISSING")" >> "$work/table.md"
    [ "$restored_schema" = t ] || mismatch=$((mismatch + 1))
    continue
  fi
  live=$(q "SELECT count(*) FROM public.\"$name\"")
  restored=$(psql "$RESTORE_URL" -X -A -t -c "SELECT count(*) FROM public.\"$name\"" 2>/dev/null || echo "missing")
  checked=$((checked + 1))
  # The live table can change between the dump and this count (new scans and alerts, expired
  # sessions), so allow a small difference either way.
  diff=$(( restored == live ? 0 : (restored > live ? restored - live : live - restored) ))
  if [ "$restored" = "missing" ] || [ "$diff" -gt $((live / 100 + 5)) ]; then
    ok="NO"; mismatch=$((mismatch + 1))
  else
    ok="yes"
  fi
  printf '| %s | %s | %s | %s |\n' "$name" "$live" "$restored" "$ok" >> "$work/table.md"
done < "$work/sizes"
cat "$work/table.md"
t3=$(now)

{
  echo "## Database restore drill"
  echo ""
  echo "- Tables: ${tables} (rows checked for ${checked}, schema only for ${#skipped[@]})"
  echo "- Dump: ${dump_mb} MB in $((t1 - t0)) s; restore: $((t2 - t1)) s; whole drill: $((t3 - t0)) s"
  echo "- Problems: ${mismatch}"
  echo ""
  cat "$work/table.md"
} >> "$OUT"

if [ "$mismatch" -gt 0 ]; then
  echo "::error::${mismatch} table(s) did not restore as expected"
  exit 1
fi
echo "Restore drill passed: ${tables} tables, dump ${dump_mb} MB, total $((t3 - t0)) s."
