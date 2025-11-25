from __future__ import annotations

def normalize_ticker(sym: str) -> str:
    s = str(sym).strip().upper()
    if not s: return ""
    if s.startswith("$"): s = s[1:]
    if "." in s:
        parts = s.split(".")
        if len(parts[-1]) == 1:  # class share (BRK.B)
            s = "-".join(parts)
    return s.replace(" ", "")

def sanitize_ticker_list(tickers):
    seen, out = set(), []
    for t in tickers or []:
        s = normalize_ticker(t)
        if not s: continue
        if any(ch in s for ch in (":", "/", "\\")): continue
        if s not in seen:
            seen.add(s); out.append(s)
    return out

def as_ticker_list(obj):
    if obj is None: return []
    if isinstance(obj, (list, tuple, set)): return [str(x) for x in obj]
    if isinstance(obj, dict): return [str(k) for k in obj.keys()]
    try:
        import pandas as pd
        if isinstance(obj, pd.DataFrame):
            for col in ("Ticker","Symbol","ticker","symbol"):
                if col in obj.columns: return obj[col].dropna().astype(str).tolist()
            if obj.shape[1]>0: return obj.iloc[:,0].dropna().astype(str).tolist()
    except Exception: pass
    return [str(obj)]