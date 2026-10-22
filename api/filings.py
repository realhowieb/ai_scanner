"""SEC EDGAR filings: insider trades (Form 4) and company events (8-K). Free, no key.

EDGAR asks every caller to identify itself: HSF_SEC_USER_AGENT ("HSFinest.AI you@example.com"),
or "HSFinest.AI <SMTP_FROM>" when that is an email address. Without either, filings are off
(available=False) rather than sending anonymous requests EDGAR would block.

Lookups are shared through api.shared_cache: the ticker -> CIK map for a day, each company's
filing list and parsed Form 4s for FILINGS_TTL_S. EDGAR allows 10 requests a second; a cold
stock page makes at most 2 + FORM4_MAX of them.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

log = logging.getLogger("hsf_api")

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"
INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/"
CIK_TTL_S = 24 * 3600
FILINGS_TTL_S = 3 * 3600
FORM4_TTL_S = 7 * 24 * 3600   # a filed Form 4 never changes
LOOKBACK_DAYS = 90
FORM4_MAX = 8
EVENTS_MAX = 8
_TICKER = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")

# 8-K items people act on; the rest are summarised as "Other".
ITEMS_8K = {
    "1.01": "Material agreement", "1.02": "Agreement ended", "1.03": "Bankruptcy", "1.05": "Cybersecurity incident",
    "2.01": "Acquisition or sale completed", "2.02": "Earnings results", "2.03": "New debt",
    "2.05": "Restructuring costs", "2.06": "Impairment", "3.01": "Listing notice", "3.02": "Unregistered share sale",
    "4.01": "Auditor change", "4.02": "Restatement", "5.01": "Change in control",
    "5.02": "Executive or director change", "5.03": "Bylaws change", "5.07": "Shareholder vote",
    "7.01": "Regulation FD disclosure", "8.01": "Other events",
}
# Form 4 transaction codes shown; grants, gifts and option exercises are left out.
CODES = {"P": "Buy", "S": "Sale"}


def user_agent() -> Optional[str]:
    ua = os.environ.get("HSF_SEC_USER_AGENT", "").strip()
    if ua:
        return ua[:200]
    sender = os.environ.get("SMTP_FROM", "").strip()
    match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", sender)
    return f"HSFinest.AI {match.group(0)}" if match else None


def enabled() -> bool:
    return os.environ.get("HSF_FILINGS_ENABLED", "1").strip() != "0" and user_agent() is not None


def _get(url: str, *, client=None, as_json: bool = True) -> Any:
    import httpx

    own = client is None
    client = client or httpx.Client(timeout=8.0, follow_redirects=False)
    try:
        r = client.get(url, headers={"User-Agent": user_agent() or "", "Accept-Encoding": "gzip, deflate"})
        r.raise_for_status()
        return r.json() if as_json else r.text
    finally:
        if own:
            client.close()


def cik_for(ticker: str) -> Optional[int]:
    from api import shared_cache

    def load() -> Dict[str, int]:
        body = _get(TICKERS_URL)
        rows = body.values() if isinstance(body, dict) else []
        return {str(r.get("ticker", "")).upper(): int(r["cik_str"]) for r in rows
                if isinstance(r, dict) and r.get("ticker") and r.get("cik_str") is not None}

    table = shared_cache.cached("sec:tickers", load, CIK_TTL_S)
    t = ticker.upper()
    return table.get(t) or table.get(t.replace(".", "-"))


def _recent(cik: int) -> List[Dict[str, Any]]:
    """The company's recent filings as rows {form, date, acc, doc, items}, newest first."""
    from api import shared_cache

    def load() -> List[Dict[str, Any]]:
        body = _get(SUBMISSIONS_URL.format(cik=cik))
        r = (body.get("filings") or {}).get("recent") or {}
        forms, dates, accs = r.get("form") or [], r.get("filingDate") or [], r.get("accessionNumber") or []
        docs, items = r.get("primaryDocument") or [], r.get("items") or []
        cutoff = (dt.date.today() - dt.timedelta(days=LOOKBACK_DAYS)).isoformat()
        out = []
        for i, form in enumerate(forms):
            if form not in ("4", "8-K") or i >= len(dates) or i >= len(accs) or dates[i] < cutoff:
                continue
            out.append({"form": form, "date": dates[i], "acc": accs[i], "doc": docs[i] if i < len(docs) else "",
                        "items": items[i] if i < len(items) else ""})
        return out

    return shared_cache.cached(f"sec:recent:{cik}", load, FILINGS_TTL_S, empty_ttl_s=FILINGS_TTL_S,
                               is_empty=lambda v: v is None)


def _num(v: Optional[str]) -> Optional[float]:
    try:
        return float(v) if v not in (None, "") else None
    except ValueError:
        return None


def _find(node: Optional[ET.Element], path: str) -> Optional[str]:
    if node is None:
        return None
    hit = node.find(path)
    if hit is None:
        return None
    value = hit.find("value")
    text = (value.text if value is not None else hit.text) or ""
    return text.strip() or None


def parse_form4(xml_text: str) -> List[Dict[str, Any]]:
    """Open-market buys and sales in one Form 4 (non-derivative table)."""
    root = ET.fromstring(xml_text)
    owner = root.find("reportingOwner")
    name = _find(owner, "reportingOwnerId/rptOwnerName") or "Insider"
    rel = owner.find("reportingOwnerRelationship") if owner is not None else None
    role = (_find(rel, "officerTitle") or ("Director" if _find(rel, "isDirector") in ("1", "true") else None)
            or ("10% owner" if _find(rel, "isTenPercentOwner") in ("1", "true") else None) or "Insider")
    out = []
    for tx in root.findall("nonDerivativeTable/nonDerivativeTransaction"):
        code = _find(tx, "transactionCoding/transactionCode")
        if code not in CODES:
            continue
        shares = _num(_find(tx, "transactionAmounts/transactionShares"))
        px = _num(_find(tx, "transactionAmounts/transactionPricePerShare"))
        out.append({"insider": name[:80], "role": role[:60], "type": CODES[code],
                    "date": _find(tx, "transactionDate"), "shares": shares, "price": px,
                    "value": round(shares * px, 2) if shares is not None and px is not None else None})
    return out


def _form4(cik: int, row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parsed trades for one Form 4. The primary document is the XSL-rendered path; the raw
    XML sits at the same name without the xsl folder."""
    from api import shared_cache

    acc = row["acc"].replace("-", "")
    doc = row["doc"].rsplit("/", 1)[-1]
    if not doc.endswith(".xml"):
        return []
    url = ARCHIVE_URL.format(cik=cik, acc=acc, doc=doc)

    def load() -> List[Dict[str, Any]]:
        trades = parse_form4(_get(url, as_json=False))
        for t in trades:
            t["url"] = INDEX_URL.format(cik=cik, acc=acc)
        return trades

    return shared_cache.cached(f"sec:f4:{acc}", load, FORM4_TTL_S, empty_ttl_s=FORM4_TTL_S, is_empty=lambda v: v is None)


def _event(cik: int, row: Dict[str, Any]) -> Dict[str, Any]:
    codes = [c.strip() for c in str(row.get("items") or "").split(",") if c.strip()]
    labels = [ITEMS_8K[c] for c in codes if c in ITEMS_8K and c != "9.01"]
    return {"date": row["date"], "form": "8-K", "items": codes, "labels": labels or ["Other"],
            "url": INDEX_URL.format(cik=cik, acc=row["acc"].replace("-", ""))}


def stock_filings(ticker: str) -> Dict[str, Any]:
    """{ticker, available, insider: [...], events: [...], insider_summary} for the stock page."""
    t = (ticker or "").strip().upper()
    empty = {"ticker": t[:10], "available": False, "insider": [], "events": [],
             "insider_summary": {"buys": 0, "sales": 0, "buy_value": 0.0, "sale_value": 0.0, "days": LOOKBACK_DAYS}}
    if not _TICKER.match(t) or not enabled():
        return empty
    try:
        cik = cik_for(t)
        if cik is None:
            return {**empty, "available": True}
        recent = _recent(cik)
        insider: List[Dict[str, Any]] = []
        for row in [r for r in recent if r["form"] == "4"][:FORM4_MAX]:
            try:
                insider.extend(_form4(cik, row))
            except Exception as e:  # one unreadable filing doesn't hide the rest
                log.warning(json.dumps({"event": "sec_form4_failed", "error": type(e).__name__}))
        events = [_event(cik, r) for r in recent if r["form"] == "8-K"][:EVENTS_MAX]
    except Exception as e:
        log.warning(json.dumps({"event": "sec_filings_failed", "error": type(e).__name__}))
        return empty
    insider.sort(key=lambda x: x.get("date") or "", reverse=True)
    buys = [x for x in insider if x["type"] == "Buy"]
    sales = [x for x in insider if x["type"] == "Sale"]
    summary = {"buys": len(buys), "sales": len(sales), "days": LOOKBACK_DAYS,
               "buy_value": round(sum(x["value"] or 0 for x in buys), 2),
               "sale_value": round(sum(x["value"] or 0 for x in sales), 2)}
    return {"ticker": t, "available": True, "insider": insider[:12], "events": events, "insider_summary": summary}
