"""Read-only, slim operational evidence for the admin model summary."""
from db.engine import get_neon_conn
from db.ml_model_registry import RegistryUnavailable


def _values(row):
    return list(row.values()) if isinstance(row, dict) else row


def production_confidence():
    try:
        conn = get_neon_conn()
    except Exception as exc:
        raise RegistryUnavailable("database unavailable") from exc
    if conn is None:
        raise RegistryUnavailable("database unavailable")
    cur = None
    try:
        cur = conn.cursor()
        cur.execute("SELECT to_regclass('ai_confidence_models') AS models, to_regclass('signal_outcomes') AS signals")
        tables = _values(cur.fetchone())
        configured, captured = None, None
        if tables[0]:
            cur.execute("SELECT id, model_version, trained_at, metadata->>'dataset_version' AS dataset_version "
                        "FROM ai_confidence_models WHERE is_active = TRUE ORDER BY created_at DESC, id DESC LIMIT 1")
            row = _values(cur.fetchone())
            if row:
                configured = dict(zip(("registry_id", "model_version", "trained_at", "dataset_version"), row))
        if tables[1]:
            cur.execute("SELECT fired_at, raw_signal->'models'->'ai_confidence'->'loaded_artifact'->>'sha256' AS artifact_sha256, "
                        "raw_signal->'models'->'ai_confidence'->>'inferred_at' AS inferred_at, "
                        "raw_signal->'models'->'ai_confidence'->'loaded_artifact'->>'version' AS inferred_model_version, "
                        "raw_signal->>'ai_confidence_model_version' AS legacy_version FROM signal_outcomes "
                        "WHERE source='opportunity' AND ai_confidence IS NOT NULL ORDER BY fired_at DESC, id DESC LIMIT 1")
            row = _values(cur.fetchone())
            if row:
                captured = {"signal_at": row[0], "artifact_sha256": row[1], "inferred_at": row[2],
                            "model_version": row[3] or row[4]}
        return {"configured_model": configured, "latest_capture": captured,
                "note": "Configured database model and latest recorded inference are separate evidence. "
                        "The scorer can fall back to a local artifact; missing evidence is unknown."}
    except Exception as exc:
        raise RegistryUnavailable("production confidence evidence unavailable") from exc
    finally:
        if cur is not None:
            cur.close()
        conn.close()


def prebreakout_input_health(rows):
    """Describe captured input completeness, never prediction quality.

    Rows arrive newest first. Compare only the newest artifact/schema; malformed
    evidence remains unknown rather than counting as complete.
    """
    import math

    def vector(row):
        names, values, mask = (row.get(k) for k in ('names', 'values', 'mask'))
        if not (isinstance(names, list) and names and len(set(str(n) for n in names)) == len(names)
                and all(isinstance(n, str) and n for n in names)
                and isinstance(values, list) and isinstance(mask, list)
                and len(names) == len(values) == len(mask)
                and all(isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v) for v in values)
                and all(isinstance(v, bool) for v in mask)):
            return None
        return names, mask

    rows = list(rows)
    latest = rows[0] if rows else {}
    current = vector(latest)
    result = {'status': 'UNKNOWN', 'sample_limit': 500, 'lookback_days': 7,
              'sampled_predictions': len(rows), 'comparable_predictions': 0, 'unknown_predictions': 0,
              'other_identity_predictions': 0, 'latest_prediction_at': latest.get('fired_at'),
              'model_version': latest.get('version'), 'input_contract': latest.get('contract'),
              'feature_count': None, 'latest_real_inputs': None, 'latest_defaulted_inputs': None,
              'features': [], 'note': 'Input completeness only, not accuracy or calibration. '
              'At most 500 recent predictions from seven days; comparison uses the latest artifact, feature schema and input contract. '
              'A real zero is valid; only the captured default mask identifies a filled input.'}
    if current is None:
        result['unknown_predictions'] = len(rows)
        return result
    names, mask = current
    counts = [0] * len(names)
    for row in rows:
        parsed = vector(row)
        if parsed is None:
            result['unknown_predictions'] += 1
            continue
        if row.get('version') != latest.get('version') or row.get('sha') != latest.get('sha') or row.get('contract') != latest.get('contract') or parsed[0] != names:
            result['other_identity_predictions'] += 1
            continue
        result['comparable_predictions'] += 1
        counts = [a + int(b) for a, b in zip(counts, parsed[1])]
    result.update(feature_count=len(names), latest_real_inputs=len(names)-sum(mask), latest_defaulted_inputs=sum(mask))
    n = result['comparable_predictions']
    result['features'] = sorted([{'name': name, 'defaulted_predictions': count, 'predictions': n,
                                 'defaulted_pct': round(100*count/n, 1)} for name, count in zip(names, counts)],
                                key=lambda f: (-f['defaulted_predictions'], f['name']))
    result['status'] = ('UNKNOWN' if not latest.get('sha') or not latest.get('version') else
                        'WATCH' if any(counts) or result['unknown_predictions'] else 'HEALTHY')
    return result


def production_prebreakout():
    """Configured model metadata and recorded inference identity, without loading artifacts."""
    try:
        conn = get_neon_conn()
    except Exception as exc:
        raise RegistryUnavailable("database unavailable") from exc
    if conn is None:
        raise RegistryUnavailable("database unavailable")
    cur = None
    try:
        cur = conn.cursor()
        cur.execute("SELECT to_regclass('prebreakout_models') AS models, to_regclass('signal_outcomes') AS signals")
        tables = _values(cur.fetchone())
        configured, captured = None, None
        input_rows = []
        if tables[0]:
            cur.execute("SELECT id, model_version, trained_at, metadata->>'dataset_version' AS dataset_version "
                        "FROM prebreakout_models WHERE is_active = TRUE ORDER BY created_at DESC, id DESC LIMIT 1")
            row = _values(cur.fetchone())
            if row:
                configured = dict(zip(("registry_id", "model_version", "trained_at", "dataset_version"), row))
        if tables[1]:
            cur.execute("SELECT fired_at, raw_signal->'models'->'prebreakout'->'loaded_artifact'->>'sha256' AS artifact_sha256, "
                        "raw_signal->'models'->'prebreakout'->>'inferred_at' AS inferred_at, "
                        "raw_signal->'models'->'prebreakout'->'loaded_artifact'->>'version' AS inferred_model_version, "
                        "raw_signal->>'prebreakout_model_version' AS legacy_version FROM signal_outcomes "
                        "WHERE source='opportunity' AND prebreakout_prob IS NOT NULL "
                        "AND (raw_signal->'models'->'prebreakout'->>'status' = 'captured' OR "
                        "(raw_signal->'models'->'prebreakout'->>'status' IS NULL "
                        "AND raw_signal->>'prebreakout_model_version' IS NOT NULL AND prebreakout_prob > 0)) "
                        "ORDER BY fired_at DESC, id DESC LIMIT 1")
            row = _values(cur.fetchone())
            if row:
                captured = {"signal_at": row[0], "artifact_sha256": row[1], "inferred_at": row[2],
                            "model_version": row[3] or row[4]}
            cur.execute("SELECT fired_at, raw_signal->'models'->'prebreakout'->'feature_names' AS names, "
                        "raw_signal->'models'->'prebreakout'->'feature_values' AS values, "
                        "raw_signal->'models'->'prebreakout'->'default_mask' AS mask, "
                        "raw_signal->'models'->'prebreakout'->'loaded_artifact'->>'version' AS version, "
                        "raw_signal->'models'->'prebreakout'->'loaded_artifact'->>'sha256' AS sha, "
                        "raw_signal->'models'->'prebreakout'->'enrichment_availability'->>'input_contract' AS contract "
                        "FROM signal_outcomes WHERE source='opportunity' AND fired_at >= NOW()-INTERVAL '7 days' "
                        "AND fired_at <= NOW() AND raw_signal->'models'->'prebreakout'->>'status'='captured' "
                        "ORDER BY fired_at DESC, id DESC LIMIT 500")
            keys = ('fired_at', 'names', 'values', 'mask', 'version', 'sha', 'contract')
            input_rows = [{key: row.get(key) for key in keys} if isinstance(row, dict) else dict(zip(keys, row))
                          for row in (cur.fetchall() or [])]
        return {"configured_model": configured, "latest_capture": captured,
                "input_health": prebreakout_input_health(input_rows),
                "note": "Configured database model and latest recorded inference are separate evidence. "
                        "The scorer can fall back to a local artifact; missing evidence is unknown."}
    except Exception as exc:
        raise RegistryUnavailable("production PreBreakout evidence unavailable") from exc
    finally:
        if cur is not None:
            cur.close()
        conn.close()


def capture_summary(rows, *, now, since):
    """Capture health relative to the observation stream, never model quality."""
    by_role = {r['role']: r for r in rows}
    models = []
    for role in ('ai_confidence', 'prebreakout'):
        r = by_role.get(role, {})
        total, present, identity = (int(r.get(k) or 0) for k in ('total', 'present', 'identity'))
        latest, predicted = r.get('latest_observation'), r.get('latest_prediction')
        missing = total - present
        identity_missing = present - identity
        # These values come from MAX(fired_at), so direct comparison shares a type/timezone.
        behind = bool(latest and (not predicted or predicted < latest))
        status = 'UNKNOWN' if not total else 'WATCH' if missing or identity_missing or behind else 'HEALTHY'
        models.append({'role': role, 'status': status, 'observations': total, 'predictions': present,
                       'missing_predictions': missing, 'identity_missing': identity_missing,
                       'latest_observation_at': latest, 'latest_prediction_at': predicted,
                       'behind_latest_observation': behind, 'model_versions': r.get('versions') or []})
    return {'generated_at': now, 'since': since, 'models': models,
            'note': 'Capture completeness and identity only, not predictive quality or a process heartbeat. '
                    'Freshness is compared with the latest saved opportunity; scanner cadence is monitored separately.'}


def production_capture_health(now=None):
    import datetime as dt

    from analytics.ml_readiness import _day_close_utc, _trading_days_ago
    from db.signal_outcomes import _SCANNER_SUPERSEDES_BRIEF

    now = now or dt.datetime.now(dt.timezone.utc)
    since = _day_close_utc(_trading_days_ago(now, 5))
    conn = None
    cur = None
    try:
        conn = get_neon_conn()
        if conn is None:
            raise RegistryUnavailable('database unavailable')
        cur = conn.cursor()
        cur.execute("SELECT to_regclass('signal_outcomes') AS signals")
        if not _values(cur.fetchone())[0]:
            return capture_summary([], now=now, since=since)
        # A single bounded aggregate. No model bytes, raw matrices, tickers or secrets leave the DB.
        query = """WITH observations AS (
          SELECT fired_at, ai_confidence, prebreakout_prob, raw_signal
          FROM signal_outcomes WHERE source='opportunity' AND fired_at >= %s AND fired_at <= %s
          """ + _SCANNER_SUPERSEDES_BRIEF + """
        ), values_by_role AS (
          SELECT fired_at, x.* FROM observations CROSS JOIN LATERAL (VALUES
            ('ai_confidence', ai_confidence, raw_signal->'models'->'ai_confidence',
             raw_signal->>'ai_confidence_model_version', 100.0),
            ('prebreakout', prebreakout_prob, raw_signal->'models'->'prebreakout',
             raw_signal->>'prebreakout_model_version', 100.0)
          ) AS x(role, value, provenance, legacy_version, ceiling)
        ), classified AS (
          SELECT *, (value BETWEEN 0 AND ceiling
            AND (provenance->>'status' IS NULL OR provenance->>'status'='captured')
            AND (role='ai_confidence' OR provenance->>'status'='captured'
              OR (provenance->>'status' IS NULL AND NULLIF(legacy_version,'') IS NOT NULL AND value>0))) IS TRUE AS present,
            (provenance->>'status'='captured'
              AND NULLIF(provenance->'loaded_artifact'->>'version','') IS NOT NULL
              AND (provenance->'loaded_artifact'->>'sha256') ~ '^[a-fA-F0-9]{64}$'
              AND NULLIF(provenance->>'inferred_at','') IS NOT NULL) IS TRUE AS identity
          FROM values_by_role
        ) SELECT role, COUNT(*) AS total, COUNT(*) FILTER (WHERE present) AS present,
          COUNT(*) FILTER (WHERE present AND identity) AS identity,
          MAX(fired_at) AS latest_observation, MAX(fired_at) FILTER (WHERE present) AS latest_prediction,
          array_agg(DISTINCT COALESCE(NULLIF(provenance->'loaded_artifact'->>'version',''), legacy_version))
            FILTER (WHERE present AND COALESCE(provenance->'loaded_artifact'->>'version', legacy_version) IS NOT NULL) AS versions
          FROM classified GROUP BY role ORDER BY role"""
        cur.execute(query, (since, now))
        names = ('role', 'total', 'present', 'identity', 'latest_observation', 'latest_prediction', 'versions')
        rows = [dict(r) if isinstance(r, dict) else dict(zip(names, r)) for r in cur.fetchall()]
        return capture_summary(rows, now=now, since=since)
    except RegistryUnavailable:
        raise
    except Exception as exc:
        raise RegistryUnavailable('production prediction capture unavailable') from exc
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()
