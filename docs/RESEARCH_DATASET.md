# HSF research dataset (point-in-time foundation)

An internal, read-only layer that answers two questions separately for any
historical HSF observation:

1. **What exactly did HSF know at that moment?** Use `FeatureSnapshot` / `GET /v1/research/features/{id}`.
2. **What happened afterwards?** Use `OutcomeRecord`, built from Outcome Intelligence's canonical record.

It trains nothing and changes no scoring, ranking, signal or model. Code:

| File | Role |
|---|---|
| `analytics/research_schema.py` | Feature schema v1, label schema v1, `FeatureSnapshot`, `OutcomeRecord`, `join_features_labels` |
| `analytics/research_dataset.py` | Observation identity, temporal join, snapshot, outcome adapter, filters, coverage, quality, overlap, `build_dataset`, fingerprint |
| `db/research_datasets.py` | Two bulk reads per window, plus the insert-once version registry |
| `api/research.py` + `api/main.py` | Admin-only GET endpoints with cache |
| `scripts/research_dataset.py` | Report (default), `--export`, `--finalize` |
| `ml/reports/point_in_time_feature_audit.md` | Per-feature classification and join rules |

## Canonical research observation

One row of `signal_outcomes` with `source='opportunity'`. This is the same id
that Outcome Intelligence measures, so there is no competing observation or
outcome store. `observed_at` is the row's `fired_at`.

| Part | Fields | Source |
|---|---|---|
| identity | observation_id, ticker, observed_at | the row |
| HSF state | hsf_score, rank, setup, status, signals, scoring_version | frozen payload |
| provenance | scan_id, universe, scanner_scoring_version, scanner_commit_sha, prebreakout_model_code_constant, `model_version` (always null: never recorded), `run_id` (null: not stored) | the joined scan record |
| features | schema v1 (33 columns) | frozen payload + joined scan record |
| outcome | label schema v1 | Outcome Intelligence canonical record |
| maturity | per horizon: pending / matured / unavailable / invalid; certified | Outcome Intelligence |
| overlap | ticker + entry trading day group, group size, first in group, overlapping windows | derived from rows in the window |

## Feature / label separation

- `FeatureSnapshot` is a frozen dataclass whose values are a read-only mapping. Its constructor rejects any key outside the declared schema version and any key that looks like an outcome (`return`, `mfe`, `mae`, `benchmark`, `excess`, `future`, `matur`, `label` …).
- `OutcomeRecord` is a separate frozen dataclass. A feature matrix is built only from `FeatureSnapshot.vector()`, and a label matrix only from `OutcomeRecord.vector()`.
- `join_features_labels()` is the explicit pairing, and it checks that `observed_at` matches.
- `/v1/research/features/{id}` serializes only the snapshot. `/observations` adds an `outcome` object only with `include_outcomes=true`.

## Versioning

- **Feature schema** (`FEATURE_SCHEMA_VERSION = 1`): name, type, source, stored path, nullable, point-in-time classification, description. A test pins v1's column list. Adding, dropping or redefining a feature means adding v2 to `FEATURE_SCHEMAS`.
- **Label schema** (`LABEL_SCHEMA_VERSION = 1`): only existing definitions. These are 1/3/5 trading-day close returns, 5-day MFE/MAE, SPY benchmark and excess return (null unless both are present). No 10/15/20-bar labels exist, so none are defined.
- **Dataset versions** (`hsf-ml-YYYY-MM-DD-vN`): `scripts/research_dataset.py --finalize` writes one row to `research_dataset_versions` with the filters, member ids, fingerprint and metadata (schema versions, counts, date range, version distributions, data quality, feature coverage, code revision). The insert uses `ON CONFLICT DO NOTHING`, and no code path updates or deletes a version. `GET /v1/research/datasets/{v}?verify=true` rebuilds from current data and code and reports `reproducible` without touching the stored row. If the data changed, finalize a new version.

## Determinism and fingerprint

`build_dataset` sorts by `(observed_at, observation_id)` and ranks within
snapshots by `(score desc, ticker, id)`. The scan join breaks ties by context,
then record id. The fingerprint is
`sha256` of canonical JSON (sorted keys, compact separators, UTF-8) over: schema
versions, normalized filters, feature and label column names, ordered ids and
`observed_at`, the feature matrix, the label matrix and 5-day maturity.
`created_at`, the version name and the code revision are deliberately left out,
so unchanged data and code reproduce the same fingerprint. Input row order
doesn't matter (tested).

## Temporal join rules

See `ml/reports/point_in_time_feature_audit.md`. In short, a scan record is used
only when it was started **and written** at or before `observed_at`, within 3 h,
latest first. No join of the form `ticker -> latest metadata` exists anywhere in
this layer.

## Missingness

Nothing is filled. A missing historical value is `null`, and coverage reports it.
Imputation belongs in a future versioned preprocessing step. (Note: the current
PreBreakout training pipeline does `fillna(0.0)`.)

## Caching and performance

- One cached record set per date window (15 min fresh, then stale for up to 1 h while it refreshes). Coverage and observation pages share it.
- The dataset list and each version's registry row are cached. Finalized versions are immutable, so caching can't change reproducibility. Verification always reads uncached.
- Each window takes two queries: opportunities by date range, and scan records for those tickers via a slim JSON projection. No per-observation query and no provider calls.

## Operating

```
# read-only report against production (manual Diagnostics workflow)
workflow_dispatch diagnostics.yml  script=research_dataset.py
# local / with DATABASE_URL
python scripts/research_dataset.py --start 2026-09-12 --end 2026-10-08 --export artifacts/research/ds.parquet
python scripts/research_dataset.py --start 2026-09-12 --end 2026-10-08 --finalize   # writes one registry row
```

Parquet exports carry the full metadata in the file's schema metadata (key
`hsf_research`). Feature columns are prefixed `f__` and label columns `y__`.
