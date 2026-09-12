"""Canonical HSF opportunity state-change engine (pure, deterministic).

ONE definition of "what meaningfully changed" between two comparable HSF
opportunity states — reused by the background evaluator, storage, and UI. It
composes the CANONICAL movement semantics from ui.opportunities
(compare_opportunities: NO_BASELINE / NEW / RISING / FALLING / UNCHANGED /
VERSION_CHANGED, status transitions, the ±3 threshold) — it never re-defines
movement or the HSF score. No Claude, no fabrication.

Two stages live elsewhere; this module is Stage A (market event detection) plus
the deterministic severity / collapsing / dedupe / policy helpers that Stage B
(user matching) consumes.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

_STATUS_RANK = {"CAUTION": 1, "WATCH": 2, "STRONG": 3}

# Event types (restrained V1 set).
NEW_OPPORTUNITY = "NEW_OPPORTUNITY"
STATUS_UPGRADE = "STATUS_UPGRADE"
STATUS_DOWNGRADE = "STATUS_DOWNGRADE"
RISING = "RISING"
FALLING = "FALLING"
FADING = "FADING"
DROPPED = "DROPPED"
SIGNAL_ADDED = "SIGNAL_ADDED"
SIGNAL_REMOVED = "SIGNAL_REMOVED"
VERSION_CHANGED = "VERSION_CHANGED"

# Which pref key gates each event type. VERSION_CHANGED has no pref (never a
# user market event — recorded internal only).
_EVENT_PREF = {
    NEW_OPPORTUNITY: "new", STATUS_UPGRADE: "upgrade", STATUS_DOWNGRADE: "downgrade",
    RISING: "rising", FALLING: "falling", FADING: "fading", DROPPED: "dropped",
    SIGNAL_ADDED: "signal_added", SIGNAL_REMOVED: "signal_removed",
}

# Conservative, high-value defaults (Run 22 §17/§19): status changes, fading,
# dropped, new on; noisy generic movement / signal churn off by default.
DEFAULT_PREFERENCES = {
    "new": True, "upgrade": True, "downgrade": True, "fading": True, "dropped": True,
    "rising": False, "falling": False, "signal_added": False, "signal_removed": False,
}

# Collapsing priority — strongest semantic wins (Run 22 §26).
_PRIORITY = [STATUS_DOWNGRADE, FADING, STATUS_UPGRADE, DROPPED, NEW_OPPORTUNITY,
             FALLING, RISING, SIGNAL_REMOVED, SIGNAL_ADDED, VERSION_CHANGED]
_IMPORTANT_SIGNALS = {"breakout", "golden_cross"}

_SIGNAL_LABELS = {"golden_cross": "Golden Cross", "breakout": "Breakout",
                  "prebreakout": "PreBreakout", "gapper": "Gapper", "gainer": "Momentum"}


def _rank(s: Optional[str]) -> int:
    return _STATUS_RANK.get(str(s or "").upper(), 0)


def _bucket(score: Optional[float]) -> Optional[int]:
    if score is None:
        return None
    try:
        return int(float(score)) // 10  # 10-wide bucket so tiny wiggles don't re-fire
    except (TypeError, ValueError):
        return None


def _dedupe_by_ticker(rows: Optional[List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in (rows or []):
        t = str(r.get("ticker") or "").strip().upper()
        if t and t not in out:  # first wins
            out[t] = r
    return out


def derive_opportunity_events(
    previous: Optional[List[Dict[str, Any]]],
    current: Optional[List[Dict[str, Any]]],
    *,
    previous_snapshot_time: Any = None,
    current_snapshot_time: Any = None,
    source_context: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """All structured events between two comparable opportunity states.

    NO_BASELINE yields NO events (never masquerades as NEW). VERSION_CHANGED
    yields only a metadata event (never RISING/FALLING). Signal add/remove is
    set-based (ordering never matters). DROPPED = present-before, absent-now.
    Malformed rows are skipped, not fatal.
    """
    from ui.opportunities import compare_opportunities

    current = [c for c in (current or []) if isinstance(c, dict) and c.get("ticker") and c.get("score") is not None]
    compared = compare_opportunities(current, previous)
    prev_by = _dedupe_by_ticker(previous)
    events: List[Dict[str, Any]] = []

    def base(c: Dict[str, Any]) -> Dict[str, Any]:
        cur_sigs = set(c.get("signals") or [])
        p = prev_by.get(str(c.get("ticker") or "").upper())
        prev_sigs = set(p.get("signals") or []) if p else set()
        return {
            "ticker": str(c.get("ticker")).upper(),
            "previous_score": c.get("previous_score"), "current_score": c.get("score"),
            "score_delta": c.get("score_delta"), "score_version": c.get("score_version"),
            "previous_status": c.get("previous_status"), "current_status": c.get("status"),
            "movement": c.get("movement_state"), "primary_setup": c.get("primary_setup"),
            "signals_added": sorted(cur_sigs - prev_sigs),
            "signals_removed": sorted(prev_sigs - cur_sigs),
            "fading": bool(c.get("fading")),
            "source_context": source_context,
            "previous_snapshot_time": previous_snapshot_time,
            "current_snapshot_time": current_snapshot_time,
        }

    def mk(event_type: str, b: Dict[str, Any], **extra) -> Dict[str, Any]:
        e = {"event_type": event_type, **b, **extra}
        e["severity"] = event_severity(e)
        return e

    for c in compared:
        ms = c.get("movement_state")
        if ms == "NO_BASELINE":
            continue  # mandatory: no baseline -> no events
        b = base(c)
        if ms == "VERSION_CHANGED":
            events.append(mk(VERSION_CHANGED, b))
            continue  # never a comparable score move
        if ms == "NEW":
            events.append(mk(NEW_OPPORTUNITY, b))
        tr = c.get("status_transition")
        if tr:
            if _rank(tr[1]) > _rank(tr[0]):
                events.append(mk(STATUS_UPGRADE, b))
            elif _rank(tr[1]) < _rank(tr[0]):
                events.append(mk(STATUS_DOWNGRADE, b))
        if ms == "RISING":
            events.append(mk(RISING, b))
        elif ms == "FALLING":
            events.append(mk(FALLING, b))
        # FADING: canonical fading flag AND weakening (falling or a downgrade).
        weakening = ms == "FALLING" or (tr and _rank(tr[1]) < _rank(tr[0]))
        if b["fading"] and weakening:
            events.append(mk(FADING, b))
        for s in b["signals_added"]:
            events.append(mk(SIGNAL_ADDED, b, signal=s))
        for s in b["signals_removed"]:
            events.append(mk(SIGNAL_REMOVED, b, signal=s))

    # DROPPED — only meaningful with a real baseline.
    if previous:
        cur_tk = {str(c.get("ticker")).upper() for c in current}
        for t, p in prev_by.items():
            if t not in cur_tk:
                events.append(mk(DROPPED, {
                    "ticker": t, "previous_score": p.get("score"), "current_score": None,
                    "score_delta": None, "score_version": p.get("score_version"),
                    "previous_status": p.get("status"), "current_status": None,
                    "movement": DROPPED, "primary_setup": p.get("primary_setup"),
                    "signals_added": [], "signals_removed": [], "fading": False,
                    "source_context": source_context,
                    "previous_snapshot_time": previous_snapshot_time,
                    "current_snapshot_time": current_snapshot_time,
                }))
    return events


def event_severity(event: Dict[str, Any]) -> str:
    """Deterministic HIGH / MEDIUM / LOW (Run 22 §16). No AI."""
    t = event.get("event_type")
    prev, cur = event.get("previous_status"), event.get("current_status")
    if t == STATUS_UPGRADE:
        return "HIGH" if _rank(cur) == 3 else "MEDIUM"
    if t in (STATUS_DOWNGRADE, FADING, DROPPED):
        return "HIGH" if _rank(prev) == 3 else "MEDIUM"
    if t == NEW_OPPORTUNITY:
        return "MEDIUM"
    if t in (SIGNAL_ADDED, SIGNAL_REMOVED):
        return "MEDIUM" if event.get("signal") in _IMPORTANT_SIGNALS else "LOW"
    return "LOW"  # RISING/FALLING/VERSION_CHANGED


def collapse_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One notification per ticker: the strongest semantic event, enriched with
    the related facts (score move, transition, signal changes). Preserves the
    underlying events in `contributing`."""
    by_ticker: Dict[str, List[Dict[str, Any]]] = {}
    for e in events:
        by_ticker.setdefault(e["ticker"], []).append(e)
    order = {t: i for i, t in enumerate(_PRIORITY)}
    out = []
    for ticker, evs in by_ticker.items():
        primary = min(evs, key=lambda e: order.get(e["event_type"], 99))
        added = sorted({s for e in evs for s in (e.get("signals_added") or [])})
        removed = sorted({s for e in evs for s in (e.get("signals_removed") or [])})
        note = dict(primary)
        note["signals_added"] = added
        note["signals_removed"] = removed
        note["contributing"] = sorted({e["event_type"] for e in evs})
        note["severity"] = max((event_severity(e) for e in evs),
                               key=lambda s: {"LOW": 0, "MEDIUM": 1, "HIGH": 2}[s])
        out.append(note)
    return out


def event_fingerprint(user_id: str, note: Dict[str, Any]) -> str:
    """State-aware dedupe key (Run 22 §22) — NOT timestamp-based, so a persistent
    state won't re-fire every cron run, but a genuine later re-transition (whose
    prior/next state differs) produces a different key. Uses score BUCKETS so
    sub-threshold wiggles don't create new fingerprints."""
    return "|".join([
        str(user_id), str(note.get("ticker")), str(note.get("event_type")),
        str(note.get("previous_status")), str(note.get("current_status")),
        str(_bucket(note.get("previous_score"))), str(_bucket(note.get("current_score"))),
        ",".join(note.get("signals_added") or []), ",".join(note.get("signals_removed") or []),
        str(note.get("signal") or ""),
    ])


def should_notify(
    note: Dict[str, Any],
    preferences: Optional[Dict[str, bool]],
    recent_fingerprints: Optional[Set[str]] = None,
    *,
    user_id: str = "",
) -> Dict[str, Any]:
    """Deterministic notification policy (Run 22 §20). Separate from detection.
    Returns {notify, reason}."""
    prefs = {**DEFAULT_PREFERENCES, **(preferences or {})}
    t = note.get("event_type")
    if t == VERSION_CHANGED:
        return {"notify": False, "reason": "version change is not a market event"}
    pref_key = _EVENT_PREF.get(t)
    if pref_key and not prefs.get(pref_key, False):
        return {"notify": False, "reason": f"preference '{pref_key}' off"}
    fp = event_fingerprint(user_id, note)
    if recent_fingerprints and fp in recent_fingerprints:
        return {"notify": False, "reason": "deduped (already delivered / within cooldown)"}
    return {"notify": True, "reason": "eligible", "fingerprint": fp}


def notification_copy(note: Dict[str, Any]) -> str:
    """Compact deterministic copy. No execution language; DROPPED is a ranking
    change, never a price claim."""
    t = note.get("event_type")
    ticker = note.get("ticker")
    ps, cs = note.get("previous_status"), note.get("current_status")
    pscore, cscore = note.get("previous_score"), note.get("current_score")
    added = ", ".join(_SIGNAL_LABELS.get(s, s) for s in (note.get("signals_added") or []))
    removed = ", ".join(_SIGNAL_LABELS.get(s, s) for s in (note.get("signals_removed") or []))
    move = f"HSF {pscore} → {cscore}" if pscore is not None and cscore is not None else (
        f"HSF {cscore}" if cscore is not None else "")

    if t == STATUS_UPGRADE:
        head = f"{ticker} strengthened · {ps} → {cs}"
    elif t == STATUS_DOWNGRADE:
        head = f"{ticker} weakened · {ps} → {cs}"
    elif t == FADING:
        head = f"{ticker} fading"
    elif t == DROPPED:
        return f"{ticker} dropped from HSF opportunities · no longer ranked in the current opportunity set"
    elif t == NEW_OPPORTUNITY:
        head = f"{ticker} · new HSF opportunity · {cs}"
    elif t == RISING:
        head = f"{ticker} ▲ RISING"
    elif t == FALLING:
        head = f"{ticker} ▼ FALLING"
    elif t in (SIGNAL_ADDED, SIGNAL_REMOVED):
        head = f"{ticker} · signals changed"
    else:
        head = f"{ticker} · {t}"
    parts = [head]
    if move:
        parts.append(move)
    extra = []
    if added:
        extra.append(f"{added} added")
    if removed:
        extra.append(f"{removed} removed")
    if extra:
        parts.append(" · ".join(extra))
    return "\n".join(parts)
