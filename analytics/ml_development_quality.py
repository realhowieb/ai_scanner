"""Development-only feature quality and temporal drift; no model fitting or final outcomes."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from analytics import ml_v3_audit as A


def numeric(values):
    return np.array([A.Encoder._num(v) for v in values], dtype=float)


def describe(values):
    x = numeric(values)
    valid = x[np.isfinite(x)]
    return {"rows": len(x), "missing": int((~np.isfinite(x)).sum()),
            "missing_pct": round(100*float((~np.isfinite(x)).mean()), 2) if len(x) else None,
            "unique": int(len(np.unique(valid))), "zero_pct_of_observed": round(100*float((valid == 0).mean()), 2) if len(valid) else None,
            "median": float(np.median(valid)) if len(valid) else None,
            "p10": float(np.quantile(valid, 0.1)) if len(valid) else None,
            "p90": float(np.quantile(valid, 0.9)) if len(valid) else None}


def psi(early, late):
    """Bins learned on early observed values only; missing is a separate bin."""
    a, b = numeric(early), numeric(late)
    observed = a[np.isfinite(a)]
    if not len(a) or not len(b) or not len(observed):
        return None
    cuts = np.unique(np.quantile(observed, [0.1, 0.25, 0.5, 0.75, 0.9]))
    edges = np.concatenate(([-math.inf], cuts, [math.inf]))
    def shares(v):
        counts = np.append(np.histogram(v[np.isfinite(v)], bins=edges)[0], (~np.isfinite(v)).sum())
        return (counts + 0.5) / (len(v) + 0.5*len(counts))
    x, y = shares(a), shares(b)
    return round(float(((y-x)*np.log(y/x)).sum()), 4)


def report(rows, raw_rows, *, numeric_features, categorical_features):
    if not rows:
        raise ValueError("No development rows")
    days = sorted({r.entry_day for r in rows})
    if len(days) < 4:
        raise ValueError("Need at least four development entry days for temporal diagnostics")
    split = days[len(days)//2]
    early, late = [r for r in rows if r.entry_day < split], [r for r in rows if r.entry_day >= split]
    out = {"development_rows": len(rows), "entry_days": len(days), "first_entry": days[0].isoformat(),
           "last_entry": days[-1].isoformat(), "split_entry": split.isoformat(),
           "early_rows": len(early), "late_rows": len(late), "features": {}, "categorical": {}, "weekly": [], "provenance": {}}
    for name in numeric_features:
        a, b = [r.features.get(name) for r in early], [r.features.get(name) for r in late]
        out["features"][name] = {"all": describe(a+b), "early": describe(a), "late": describe(b), "psi": psi(a,b)}
    for name in categorical_features:
        a = Counter(str(r.features.get(name)) for r in early if r.features.get(name) is not None)
        b = Counter(str(r.features.get(name)) for r in late if r.features.get(name) is not None)
        out["categorical"][name] = {"early": dict(a), "late": dict(b),
                                    "late_unseen_pct": round(100*sum(n for k,n in b.items() if k not in a)/max(len(late),1),2)}
    for week in sorted({r.entry_day.isocalendar()[:2] for r in rows}):
        group = [r for r in rows if r.entry_day.isocalendar()[:2] == week]
        labels = [A.LABELS['F_clean_path_5d'].fn(r.labels,5) for r in group]
        out["weekly"].append({"week": f"{week[0]}-W{week[1]:02}", "rows": len(group),
                              "hit_rate": round(sum(labels)/len(labels),4),
                              "missing_pct": {n: describe([r.features.get(n) for r in group])["missing_pct"] for n in numeric_features}})
    wanted = {r.observation_id for r in rows}
    for role in ('ai_confidence','prebreakout'):
        counts, fields = Counter(), {}
        for raw in raw_rows:
            if int(raw['id']) not in wanted:
                continue
            signal = raw.get('raw_signal') or {}
            if isinstance(signal,str):
                import json
                signal=json.loads(signal)
            p = (signal.get('models') or {}).get(role) or {}
            counts[str(p.get('status') or 'unrecorded')] += 1
            names, mask = p.get('feature_names'), p.get('default_mask')
            if isinstance(names,list) and isinstance(mask,list) and len(names)==len(mask):
                for name, defaulted in zip(names,mask):
                    field=fields.setdefault(str(name),Counter())
                    if isinstance(defaulted,bool):
                        field['recorded']+=1
                        field['defaulted']+=int(defaulted)
        out['provenance'][role]={'status_counts':dict(counts), 'features': {n:dict(v) for n,v in fields.items()}}
    out['note']='Development diagnostics only. PSI is descriptive, not a significance test. Zero values are not assumed to be defaults; provenance masks are reported separately.'
    return out
