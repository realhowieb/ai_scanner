"""Founding 100: policy pages (public), the founder offer and a one-time founder checkout
(signed in), refund requests (members) and refund review (admins), and admin previews of
the policies. See api/founding.py."""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Literal, Optional, Union

from fastapi import Depends, FastAPI, HTTPException, Path, Query
from pydantic import BaseModel, Field

from api import founding as svc
from api import ratelimit
from api.deps import _ADMIN, _AUTH, _user, current_account, require_admin

PolicyType = Literal["founding_terms", "founding_refunds", "privacy", "terms"]
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


class FiveYear(BaseModel):
    monthly_cents: int = Field(description="The tier's plan, paying monthly")
    five_years_monthly_cents: int = Field(description="60 months of that plan")
    price_cents: int = Field(description="The one-time founder price")
    saved_cents: int


class OfferTier(BaseModel):
    tier: FounderTier
    label: str
    plan: str
    price_cents: int
    capacity: int
    remaining: int
    on_sale: bool
    disclosure: Disclosure
    five_year: FiveYear


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


PaymentKind = Literal["purchase", "upgrade"]
RefundStatus = Literal["requested", "approved", "denied", "initiated", "completed", "failed"]
RefundBasis = Literal["guarantee", "legal_requirement", "billing_error", "service_failure", "discontinuation",
                      "exception"]


class FounderPayment(BaseModel):
    kind: PaymentKind
    tier: FounderTier
    amount_cents: int
    currency: Optional[str] = None
    paid_at: str
    refunded_cents: int
    deadline: str = Field(description="End of this payment's 14-day guarantee")
    within_guarantee: bool
    refundable_cents: int


class RefundRequest(BaseModel):
    id: int
    kind: PaymentKind
    status: RefundStatus
    within_guarantee: bool = Field(description="Asked within 14 days of that payment")
    member_reason: Optional[str] = None
    amount_cents: Optional[int] = None
    requested_at: str
    due_at: str = Field(description="Approved refunds start by this date (7 business days)")
    decided_at: Optional[str] = None
    initiated_at: Optional[str] = None
    completed_at: Optional[str] = None
    decision_reason: Optional[str] = None


class MyRefunds(BaseModel):
    payments: List[FounderPayment]
    requests: List[RefundRequest]
    guarantee_days: int
    start_within_business_days: int
    support_email: str


class RefundRequestBody(BaseModel):
    kind: PaymentKind = Field(description="Which of your payments: the purchase or the upgrade")
    reason: str = Field(default="", max_length=1000, description="Optional")


class AdminRefundRequest(RefundRequest):
    user_id: str
    basis: Optional[RefundBasis] = None
    decided_by: Optional[str] = None
    provider_refund_id: Optional[str] = None
    failure_reason: Optional[str] = None
    tier: FounderTier
    payment_reference: str
    paid_cents: int
    refunded_cents: int
    paid_at: str
    number: Optional[str] = None
    membership_status: str
    activated_at: Optional[str] = None
    expires_at: Optional[str] = None
    upgraded_at: Optional[str] = None
    updated_at: Optional[str] = None


class AdminRefunds(BaseModel):
    requests: List[AdminRefundRequest]
    refunds_live: bool = Field(description="FOUNDING_REFUNDS_ENABLED: approvals are sent to Stripe")
    prorated_live: bool = Field(description="FOUNDING_PRORATED_REFUNDS_ENABLED")


class AcceptedPolicy(BaseModel):
    policy_type: str
    version: int
    content_hash: str
    accepted_at: str


class AuditEntry(BaseModel):
    at: str
    actor: str
    action: str
    detail: Optional[str] = None


class AdminRefundDetail(BaseModel):
    request: AdminRefundRequest
    accepted_policies: List[AcceptedPolicy]
    audit: List[AuditEntry]
    refunds_live: bool
    prorated_live: bool


class ProratedQuote(BaseModel):
    request_id: int
    kind: PaymentKind
    paid_cents: int
    refunded_cents: int
    start: str
    end: str
    stopped_at: str
    unused_months: str
    total_months: str
    amount_cents: int


class ApproveBody(BaseModel):
    basis: RefundBasis
    amount_cents: Optional[int] = Field(default=None, ge=1, description="Defaults to all that is left, or the "
                                        "prorated amount for discontinuation")
    reason: str = Field(default="", max_length=1000)
    stopped_at: Optional[dt.datetime] = Field(default=None, description="Discontinuation: when service stopped")


class DenyBody(BaseModel):
    reason: str = Field(min_length=1, max_length=1000, description="Shown to the member")


class RefundDecision(BaseModel):
    request: AdminRefundRequest
    sent: bool = False
    note: Optional[str] = None


class FeatureRef(BaseModel):
    key: str
    label: str


class PollOption(FeatureRef):
    votes: Optional[int] = None


class FounderPoll(BaseModel):
    open: bool
    choice: Optional[str] = None
    options: List[PollOption]


class FounderInvites(BaseModel):
    days: int = Field(description="Days of Pro each invited friend gets")
    total: int
    left: int


class EliteLimits(BaseModel):
    alert_limit: int
    ai_multiplier: int


class FounderValue(BaseModel):
    months_used: int
    months_left: int
    term_months: int
    monthly_cents: int
    paid_cents: int
    monthly_cost_so_far_cents: int = Field(description="What the months used would have cost monthly")
    ahead_cents: int
    break_even_month: int
    alerts_used: Optional[int] = Field(default=None, description="Founder Elite: active alerts")


class FounderBenefits(BaseModel):
    status: str
    tier: FounderTier
    badge: Optional[str] = None
    show_badge: bool = Field(description="Shared stock links show the founder badge")
    alumni_discount_percent: int = Field(description="Off any plan after the five years, for life")
    alumni_rate_ready: bool
    early_access: List[FeatureRef]
    invites: Optional[FounderInvites] = None
    elite_limits: Optional[EliteLimits] = None
    poll: FounderPoll
    value: Optional[FounderValue] = Field(default=None, description="Active members: the value meter")
    wall_name: Optional[str] = Field(default=None, description="Name on the Founders Wall; null = not listed")
    card_path: Optional[str] = Field(default=None, description="The shareable founder card page")
    reply_target: Optional[str] = Field(default=None, description="How soon founder feedback is answered, if set")


class WallBody(BaseModel):
    show: bool
    name: Optional[str] = Field(default=None, max_length=60)


class WallSeat(BaseModel):
    number: str
    taken: bool
    tier: Optional[str] = None
    label: Optional[str] = None
    name: Optional[str] = Field(default=None, description="Only when the member opted in")
    since: Optional[str] = Field(default=None, description="YYYY-MM the membership started")


class FoundersWall(BaseModel):
    open: bool
    seats: List[WallSeat]
    taken: int


class PublicFounderCard(BaseModel):
    badge: Optional[str] = None
    number: Optional[str] = None
    tier: str
    label: str
    status: str
    since: Optional[str] = None
    until: Optional[str] = None
    code: str


class BriefingBody(BaseModel):
    note: str = Field(default="", max_length=4000, description="What shipped this month, in your words")


class BriefingSendBody(BriefingBody):
    confirm: bool = Field(description="Must be true: sends to every active founder")


class BriefingHistory(BaseModel):
    month: str
    subject: str
    sent_by: Optional[str] = None
    started_at: Optional[str] = None
    sent: Optional[int] = None
    failed: Optional[int] = None


class BriefingPreview(BaseModel):
    month: str
    subject: str
    html: str
    text: str
    recipients: int
    already_sent: bool
    sent_at: Optional[str] = None
    history: List[BriefingHistory]


class BriefingSent(BaseModel):
    month: str
    sent: int
    failed: int


class BadgeBody(BaseModel):
    show: bool


class PollBody(BaseModel):
    option: str = Field(min_length=1, max_length=40)


class AdminPoll(BaseModel):
    options: List[PollOption]
    total: int


class PublicFounderBadge(BaseModel):
    badge: str


class LegalPolicyRef(BaseModel):
    policy_type: PolicyType
    version: int
    title: str
    path: str
    content_hash: str
    status: str


class LegalStatus(BaseModel):
    open: bool = Field(description="Terms of Service and Privacy Policy are published (HSF_LEGAL_PAGES_ENABLED)")
    policies: List[LegalPolicyRef]
    signup_statement: Optional[str] = Field(default=None, description="The sign-up checkbox text")
    subscription_statement: Optional[str] = Field(default=None, description="Shown under upgrade buttons")


class PublicFoundingTier(BaseModel):
    tier: str
    label: str
    price_cents: int
    capacity: int
    remaining: int
    on_sale: bool
    five_year: Optional[FiveYear] = None


class PublicFounding(BaseModel):
    open: bool
    checkout_open: bool
    tiers: List[PublicFoundingTier]
    remaining: int


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
        """The Terms of Service, Founding Membership Terms, Refund Policy or Privacy Policy: the
        newest version, or an earlier one with ?version=. Terms of Service and Privacy Policy are
        found once HSF_LEGAL_PAGES_ENABLED=1; the founding ones once FOUNDING_LEGAL_PAGES_ENABLED=1
        (which also opens the Privacy Policy)."""
        from api import legal

        if not legal.page_open(policy_type):
            raise HTTPException(404, "This page isn't published yet.")
        return svc.public_policy(_policy_or_404(policy_type, version))

    @app.get("/v1/legal/status", response_model=LegalStatus, summary="Terms of Service status (public)")
    def legal_status() -> Dict[str, Any]:
        """Whether the Terms of Service and Privacy Policy are published, their current versions,
        and the agreement text for the sign-up form and upgrade buttons."""
        from api import legal

        return legal.status()

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

    @app.get("/v1/founding/refunds/me", response_model=MyRefunds, responses=_AUTH,
             summary="Your founding payments and refund requests")
    def my_refunds(account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Each payment with its 14-day deadline and what is left to refund, and your requests."""
        return svc.my_refunds(account)

    @app.post("/v1/founding/refunds/request", response_model=RefundRequest,
              responses={**_AUTH, 409: {"description": "Already requested, or nothing left to refund"}},
              summary="Ask for a refund of one of your founding payments")
    def request_refund(body: RefundRequestBody, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Always accepted: within 14 days it's covered by the guarantee, later it is reviewed by
        a person. Access is unchanged until a refund is actually made."""
        ratelimit.check("founding_refund", _user(account))
        return svc.request_refund(account, kind=body.kind, reason=body.reason)

    @app.get("/v1/admin/founding/refunds", response_model=AdminRefunds, responses=_ADMIN,
             summary="Founding refund requests (admin)")
    def admin_refunds(status: Optional[RefundStatus] = Query(default=None),
                      account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        require_admin(account)
        return svc.admin_refunds(status)

    @app.get("/v1/admin/founding/refunds/{request_id}", response_model=AdminRefundDetail, responses=_ADMIN,
             summary="One founding refund request with policies accepted and its audit trail (admin)")
    def admin_refund(request_id: int, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        require_admin(account)
        return svc.admin_refund(request_id)

    @app.get("/v1/admin/founding/refunds/{request_id}/prorated", response_model=ProratedQuote, responses=_ADMIN,
             summary="Prorated discontinuation refund for a request (admin, read-only)")
    def admin_quote(request_id: int, stopped_at: dt.datetime = Query(...),
                    account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        require_admin(account)
        return svc.admin_quote(request_id, stopped_at)

    @app.post("/v1/admin/founding/refunds/{request_id}/approve", response_model=RefundDecision, responses=_ADMIN,
              summary="Approve one founding refund request (admin)")
    def admin_approve(request_id: int, body: ApproveBody,
                      account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Records the approval; sends it to Stripe only when FOUNDING_REFUNDS_ENABLED=1. Calling
        it again on an approved request retries the Stripe step (never refunds twice)."""
        require_admin(account)
        return svc.admin_approve(account, request_id, basis=body.basis, amount_cents=body.amount_cents,
                                 reason=body.reason, stopped_at=body.stopped_at)

    @app.post("/v1/admin/founding/refunds/{request_id}/deny", response_model=RefundDecision, responses=_ADMIN,
              summary="Deny one founding refund request (admin)")
    def admin_deny(request_id: int, body: DenyBody, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        require_admin(account)
        return svc.admin_deny(account, request_id, reason=body.reason)

    @app.get("/v1/founding/benefits", response_model=FounderBenefits,
             responses={**_AUTH, 404: {"description": "Not a founding member"}},
             summary="Your founder benefits")
    def get_benefits(account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Founder rate after the five years, early access, founder invites, Elite limits,
        the shared-link badge setting and the founder poll."""
        return svc.benefits(account)

    @app.post("/v1/founding/benefits/badge", response_model=FounderBenefits, responses=_AUTH,
              summary="Show or hide your founder badge on shared links")
    def set_badge(body: BadgeBody, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        return svc.set_show_badge(account, body.show)

    @app.post("/v1/founding/poll", response_model=FounderBenefits,
              responses={**_AUTH, 403: {"description": "Active founders only"}},
              summary="Vote in the founder poll (one choice; voting again changes it)")
    def vote(body: PollBody, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        ratelimit.check("founding_refund", _user(account))
        return svc.poll_vote(account, body.option)

    @app.post("/v1/founding/wall", response_model=FounderBenefits,
              responses={**_AUTH, 400: {"description": "Name not allowed"}, 404: {"description": "Not a founding member"}},
              summary="Join or leave the Founders Wall")
    def set_wall(body: WallBody, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Off by default. show=true lists you under `name` (2-30 characters, no emails or links)
        on the public Founders Wall; show=false takes your name off (your seat shows as taken)."""
        ratelimit.check("founding_refund", _user(account))
        return svc.set_wall_name(account, body.show, body.name)

    @app.post("/v1/admin/founding/briefing/preview", response_model=BriefingPreview, responses=_ADMIN,
              summary="Preview this month's Founder Briefing (admin)")
    def briefing_preview(body: BriefingBody, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """The email as a founder would get it, how many it goes to and what was sent before.
        Sends nothing."""
        require_admin(account)
        return svc.briefing_preview(body.note)

    @app.post("/v1/admin/founding/briefing/send", response_model=BriefingSent,
              responses={**_ADMIN, 400: {"description": "Note too short or not confirmed"},
                         409: {"description": "Already sent this month"}},
              summary="Send this month's Founder Briefing to every active founder (admin)")
    def briefing_send(body: BriefingSendBody, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Once per calendar month. Never sent automatically."""
        require_admin(account)
        if not body.confirm:
            raise HTTPException(400, "Confirm the send first.")
        return svc.briefing_send(account, body.note)

    @app.get("/v1/admin/founding/poll", response_model=AdminPoll, responses=_ADMIN,
             summary="Founder poll results (admin)")
    def admin_poll(account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        require_admin(account)
        return svc.admin_poll()

    @app.get("/v1/public/founding", response_model=PublicFounding,
             summary="Whether the Founding 100 is open, and seats left (public)")
    def public_founding() -> Dict[str, Any]:
        try:
            return svc.public_status()
        except Exception:
            # The landing page treats this as "not open"; never break it.
            return {"open": False, "checkout_open": False, "tiers": [], "remaining": 0}

    @app.get("/v1/public/founding/wall", response_model=FoundersWall,
             summary="The Founders Wall: every founder seat, with names members chose to show (public)")
    def founders_wall() -> Dict[str, Any]:
        try:
            return svc.public_wall()
        except Exception:
            return {"open": False, "seats": [], "taken": 0}

    @app.get("/v1/public/founder-card/{code}", response_model=PublicFounderCard,
             responses={404: {"description": "No founder card for this invite code"}},
             summary="A shared founder card: number, tier and dates, never a name or email (public)")
    def founder_card(code: str = Path(pattern=r"^[a-z0-9]{4,32}$"),
                     _l: None = Depends(ratelimit.limit("share"))) -> Dict[str, Any]:
        card = svc.public_card(code) if svc.program_open() else None
        if not card:
            raise HTTPException(404, "No founder card.")
        return card

    @app.get("/v1/public/founder-badge/{code}", response_model=PublicFounderBadge,
             responses={404: {"description": "No founder badge for this invite code"}},
             summary="The founder badge behind a shared link's invite code (public)")
    def founder_badge(code: str = Path(pattern=r"^[a-z0-9]{4,32}$"),
                      _l: None = Depends(ratelimit.limit("share"))) -> Dict[str, Any]:
        badge = svc.public_badge(code)
        if not badge:
            raise HTTPException(404, "No founder badge.")
        return {"badge": badge}
