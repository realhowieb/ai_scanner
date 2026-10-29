"""Customer-facing policies (Terms of Service, Founding Membership Terms, Refund Policy,
Privacy Policy):
reading the versioned texts in legal/, filling in business details, and hashing them.

Each legal/<type>.v<N>.txt file is one version. The text a buyer sees is the file
with its {placeholders} filled from settings; its SHA-256 is what checkout records, so
a purchase can always be matched to the exact words accepted (db/legal.py).
Pure: no database.
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

TERMS, REFUNDS, PRIVACY = "founding_terms", "founding_refunds", "privacy"
SITE_TERMS = "terms"  # the Terms of Service every account agrees to
POLICY_TYPES = (TERMS, REFUNDS, PRIVACY, SITE_TERMS)
# Shown to everyone (sign-up, monthly checkout), switched on by HSF_LEGAL_PAGES_ENABLED;
# the founding pages keep their own switch.
GENERAL_POLICIES = (SITE_TERMS, PRIVACY)
GENERAL_PAGES_FLAG = "HSF_LEGAL_PAGES_ENABLED"
# What a founding checkout must accept, in this order.
CHECKOUT_POLICIES = (TERMS, REFUNDS, PRIVACY)
PATHS = {TERMS: "/founding-100/terms", REFUNDS: "/founding-100/refunds", PRIVACY: "/privacy",
         SITE_TERMS: "/terms"}
STATUSES = ("proposed", "approved")

ROOT = Path(__file__).resolve().parent.parent / "legal"
_FILE = re.compile(r"^(?P<type>[a-z_]+)\.v(?P<version>[1-9]\d{0,3})\.txt$")
_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")

ACKNOWLEDGMENT = ("I understand that I am purchasing a five-year HSF AI Founding Membership for a one-time "
                  "payment. My membership does not automatically renew. I have reviewed the Founding "
                  "Membership Terms, Refund Policy, and Privacy Policy.")


# What signing up and starting a monthly or yearly checkout mean, once the pages are live.
SIGNUP_STATEMENT = ("I agree to the Terms of Service and have read the Privacy Policy. I understand HSF AI is a "
                    "research tool, not financial advice.")
SUBSCRIPTION_STATEMENT = ("Your plan renews automatically each month or year until you cancel in Account. By "
                          "continuing you agree to the Terms of Service.")


class PolicyError(ValueError):
    pass


def general_pages_open(env: Optional[Mapping[str, str]] = None) -> bool:
    """Terms of Service and Privacy Policy are public (and accepted at sign-up and
    checkout). The founding pages switch also opens the Privacy Policy, as before."""
    env = os.environ if env is None else env
    return str(env.get(GENERAL_PAGES_FLAG, "0")).strip() == "1"


def details(env: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """Business details for the texts. Missing ones stay visibly marked, never invented."""
    env = os.environ if env is None else env
    base = (env.get("APP_BASE_URL") or "https://app.hsfinest.com").strip().rstrip("/")
    return {
        "legal_entity": (env.get("HSF_LEGAL_ENTITY") or "").strip() or "HSFinest LLC",
        "support_email": (env.get("HSF_SUPPORT_EMAIL") or "").strip() or "[support email to be added]",
        "governing_law": (env.get("HSF_GOVERNING_LAW") or "").strip() or "[governing-law state to be added]",
        "terms_url": base + PATHS[TERMS],
        "refunds_url": base + PATHS[REFUNDS],
        "privacy_url": base + PATHS[PRIVACY],
        "site_terms_url": base + PATHS[SITE_TERMS],
    }


def missing_details(env: Optional[Mapping[str, str]] = None) -> List[str]:
    """Settings still shown as placeholders (for the admin preview and pre-launch checks)."""
    env = os.environ if env is None else env
    return [k for k in ("HSF_SUPPORT_EMAIL", "HSF_GOVERNING_LAW") if not (env.get(k) or "").strip()]


def _fill(text: str, values: Mapping[str, str]) -> str:
    def sub(m: "re.Match[str]") -> str:
        if m.group(1) not in values:
            raise PolicyError(f"unknown placeholder {{{m.group(1)}}}")
        return values[m.group(1)]

    return _PLACEHOLDER.sub(sub, text)


def parse(raw: str, values: Mapping[str, str]) -> Dict[str, Any]:
    """One policy file -> {title, version, effective_date, status, text, content_hash, sections}.
    sections: [{"heading": str | None, "blocks": [{"p": str} | {"list": [str]}]}]."""
    head, sep, body = raw.partition("\n---\n")
    if not sep:
        raise PolicyError("missing front matter")
    meta: Dict[str, str] = {}
    for line in head.strip().splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    for key in ("title", "version", "effective_date", "status"):
        if not meta.get(key):
            raise PolicyError(f"missing {key}")
    if meta["status"] not in STATUSES:
        raise PolicyError(f"status must be one of {STATUSES}")
    text = _fill(body.strip() + "\n", values)
    sections: List[Dict[str, Any]] = [{"heading": None, "blocks": []}]
    para: List[str] = []

    def flush() -> None:
        if para:
            sections[-1]["blocks"].append({"p": " ".join(para)})
            para.clear()

    for line in text.splitlines():
        line = line.rstrip()
        if line.startswith("## "):
            flush()
            sections.append({"heading": line[3:].strip(), "blocks": []})
        elif line.startswith("- "):
            flush()
            blocks = sections[-1]["blocks"]
            if not blocks or "list" not in blocks[-1]:
                blocks.append({"list": []})
            blocks[-1]["list"].append(line[2:].strip())
        elif not line.strip():
            flush()
        else:
            para.append(line.strip())
    flush()
    if not sections[0]["blocks"]:
        sections.pop(0)
    title = _fill(meta["title"], values)
    canonical = f"{title}\nversion {meta['version']}\n{meta['effective_date']}\n\n{text}"
    return {"title": title, "version": int(meta["version"]), "effective_date": meta["effective_date"],
            "status": meta["status"], "text": text, "sections": sections,
            "content_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest()}


def load(root: Path = ROOT, env: Optional[Mapping[str, str]] = None) -> Dict[str, List[Dict[str, Any]]]:
    """Every version of every policy in `root`, oldest first: {type: [policy, ...]}."""
    values = details(env)
    out: Dict[str, List[Dict[str, Any]]] = {t: [] for t in POLICY_TYPES}
    for path in sorted(root.glob("*.txt")):
        m = _FILE.match(path.name)
        if not m or m.group("type") not in POLICY_TYPES:
            continue
        policy = parse(path.read_text(encoding="utf-8"), values)
        if policy["version"] != int(m.group("version")):
            raise PolicyError(f"{path.name}: version {policy['version']} doesn't match the file name")
        out[m.group("type")].append({**policy, "policy_type": m.group("type"), "path": PATHS[m.group("type")]})
    for policies in out.values():
        policies.sort(key=lambda p: p["version"])
    return out


def current(policies: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """The newest version of each policy type that has one."""
    return {t: versions[-1] for t, versions in policies.items() if versions}
