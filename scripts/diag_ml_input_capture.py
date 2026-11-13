"""Prediction-metadata coverage only; no outcome table or label access."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    from db.engine import get_neon_conn

    conn=get_neon_conn()
    if conn is None:
        raise RuntimeError('Database unavailable')
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT (timestamp AT TIME ZONE 'UTC')::date AS day, "
                        "COALESCE(record->>'research_cohort','UNTAGGED') AS cohort,COUNT(*) AS observations, "
                        "COUNT(*) FILTER(WHERE record->'research_metadata'->>'rsi_14' IS NOT NULL) AS rsi_present, "
                        "COUNT(*) FILTER(WHERE record->'research_metadata'->>'market_regime_at_observation' IS NOT NULL) AS regime_present "
                        "FROM hsf_observations WHERE context LIKE 'scheduled:%%' AND timestamp>=%s AND timestamp<%s "
                        "GROUP BY 1,2 ORDER BY 1,2",('2026-09-14T00:00:00Z','2026-11-16T00:00:00Z'))
            names=[c[0] for c in cur.description]
            rows=[dict(r) if isinstance(r,dict) else dict(zip(names,r)) for r in cur.fetchall()]
        print(json.dumps({'outcomes_read':False,'coverage':rows},indent=2,default=str))
    finally:
        conn.close()


if __name__=='__main__':
    main()
