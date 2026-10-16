"""Order tracking transitions, applied while the durable row is locked."""

TERMINAL = {"filled", "canceled", "expired", "rejected"}


def transition(current, state, result=None, error=None):
    """Keep completed receipts and higher fill counts when responses arrive late."""
    if current["state"] in {"synced", "rejected"}:
        return current["state"], current.get("result"), current.get("error")
    old = current.get("result")
    if old:
        if not result:
            return current["state"], old, current.get("error")
        old_fill = float(old.get("filled_qty") or 0)
        new_fill = float(result.get("filled_qty") or 0)
        if new_fill < old_fill or (old.get("status") in TERMINAL and result.get("status") not in TERMINAL):
            return current["state"], old, current.get("error")
        if new_fill == old_fill and old.get("status") == "partially_filled" and result.get("status") not in TERMINAL:
            result = old
        if state in {"pending", "rejected"}:
            return current["state"], old, current.get("error")
    if current["state"] == "submitted" and state == "pending":
        state = "submitted"
    return state, result or old, error
