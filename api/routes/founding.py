"""Founding 100: policy pages (public), the founder offer and a one-time founder checkout
(signed in), and admin previews of the policies. See api/founding.py."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from api import founding as svc
from api import ratelimit
from api.deps import _ADMIN, _AUTH, _user, current_account, require_admin

PolicyType = Literal["founding_terms", "founding_refunds", "privacy"]
FounderTier = Literal["founder_pro", "founder_elite"]


class PolicyBlock(BaseModel):
    p: Optional[str] = None
    list: Optional[List[str]] = None


class PolicySection(BaseModel):
    heading: Optional[str] = None
    blocks: List[PolicyBlock]


class Policy(BaseModel):
    policy_type: PolicyType
    version: int
    title: str
    effective_date: str
    status: Literal["proposed", "approved"] = Field(description="proposed = pending legal review")
    content_hash: str = Field(description="SHA-256 of the text shown; checkout sends it back")
    path: str
    sections: List[PolicySection]


class AdminPolicy(Policy):
    missing_details: List[str] = Field(description="Settings still shown as placeholders")
    versions: List[int]


class PolicyRef(BaseModel):
    policy_type: PolicyType
    version: int = Field(ge=1)
    content_hash: str = Field(min_length=64, max_length=64)


class OfferPolicy(PolicyRef):
    title: str
    path: str
    status: str


class Disclosure(BaseModel):
    name: str
    payment: str
    duration: str
    recurring: str
    refund: str


class OfferTier(BaseModel):
    tier: FounderTier
    label: str
    plan: str
    price_cents: int
    capacity: int
    remaining: int
    on_sale: bool
    disclosure: Disclosure


class OfferUpgrade(BaseModel):
    price_cents: int
    eligible: bool
    on_sale: bool
    disclosure: Disclosure


class FounderMembership(BaseModel):
    tier: FounderTier
    label: str
    status: str
    number: Optional[str] = None
    badge: Optional[str] = None
    activated_at: Optional[str] = None
    expires_at: Optional[str] = None
    upgraded_at: Optional[str] = None


class FoundingOffer(BaseModel):
    open: bool
    checkout_open: bool
    preview: bool = Field(description="Program still off; only admins see this page")
    tiers: List[OfferTier]
    upgrade: OfferUpgrade
    membership: Optional[FounderMembership] = None
    policies: List[OfferPolicy]
    acknowledgment: str
    missing_details: List[str]


class FoundingCheckoutBody(BaseModel):
    tier: FounderTier
    upgrade: bool = Field(default=False, description="Founder Pro -> Founder Elite for $500")
    acknowledged: bool = Field(description="The unticked-by-default acknowledgment box; must be true")
    policies: List[PolicyRef] = Field(max_length=6, description="The versions and hashes the page showed")


class FoundingCheckout(BaseModel):
    checkout_url: str
    expires_at: str


class FoundingCancel(BaseModel):
    released: bool
    paid: bool = False


def _policy_or_404(policy_type: str, version: Optional[int]) -> Dict[str, Any]:
    p = svc.policy(policy_type, version)
    if not p:
        raise HTTPException(404, "No such policy version.")
    return p


def register(app: FastAPI) -> None:
    @app.get("/v1/founding/policies/{policy_type}", response_model=Policy,
             responses={404: {"description": "Unknown policy, or the pages aren't published yet"}},
             summary="A Founding 100 policy (public)")
    def get_policy(policy_type: PolicyType, version: Optional[int] = Query(default=None, ge=1)) -> Dict[str, Any]:
        """The Founding Membership Terms, Refund Policy or Privacy Policy: the newest version,
        or an earlier one with ?version=. Not found until FOUNDING_LEGAL_PAGES_ENABLED=1."""
        if not svc.pages_open():
            raise HTTPException(404, "This page isn't published yet.")
        return svc.public_policy(_policy_or_404(policy_type, version))

    @app.get("/v1/admin/founding/policies/{policy_type}", response_model=AdminPolicy, responses=_ADMIN,
             summary="Preview a Founding 100 policy (admin)")
    def admin_policy(policy_type: PolicyType, version: Optional[int] = Query(default=None, ge=1),
                     account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Same text as the public page, available while the pages are still off, plus the
        business details still missing and the versions on file."""
        require_admin(account)
        from domain import legal

        p = _policy_or_404(policy_type, version)
        return {**svc.public_policy(p), "missing_details": legal.missing_details(),
                "versions": [v["version"] for v in svc.loaded().get(policy_type, [])]}

    @app.get("/v1/founding/offer", response_model=FoundingOffer,
             responses={**_AUTH, 404: {"description": "The Founding 100 isn't open"}},
             summary="Founding 100 tiers, seats left and your membership")
    def get_offer(account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Prices, seats left, the checkout disclosures, the policy versions to accept and
        this account's membership. Admins can preview it before the program opens."""
        try:
            return svc.offer(account)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(503, "The Founding 100 isn't available right now. Try again shortly.") from e

    @app.post("/v1/founding/checkout", response_model=FoundingCheckout,
              responses={**_AUTH, 400: {"description": "Acknowledgment missing"},
                         403: {"description": "Checkout isn't open"},
                         409: {"description": "Sold out, policies changed, already a member or a plan to cancel first"},
                         503: {"description": "Billing unavailable"}},
              summary="Start a one-time founder checkout")
    def checkout(body: FoundingCheckoutBody, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Holds a seat and returns a Stripe Checkout link for the server-set price. Needs
        acknowledged=true and the current version and hash of all three policies; records
        what was accepted against the checkout session."""
        ratelimit.check("founding_checkout", _user(account))
        return svc.start_checkout(account, tier=body.tier, upgrade=body.upgrade, acknowledged=body.acknowledged,
                                  accepted=[p.model_dump() for p in body.policies])

    @app.post("/v1/founding/checkout/cancel", response_model=FoundingCancel, responses=_AUTH,
              summary="Give back an unfinished founder checkout")
    def cancel(account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Closes your open founder checkout and releases its seat. A paid one is kept."""
        return svc.cancel_checkout(account)
