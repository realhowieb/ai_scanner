import re

import bcrypt
import streamlit as st

from db.users import load_users, seed_neon_users_from_local, update_neon_user_password

try:
    from db.users import is_login_rate_limited as _is_login_rate_limited_db
    from db.users import record_login_attempt as _record_login_attempt_db
except ImportError:
    _record_login_attempt_db = None
    _is_login_rate_limited_db = None
from ui.auth_lockout import (
    clear_failed_login_attempts as _clear_failed_login_attempts,
)
from ui.auth_lockout import (
    is_login_locked as _is_login_locked,
)
from ui.auth_lockout import (
    lockout_remaining_seconds as _lockout_remaining_seconds,
)
from ui.auth_lockout import (
    register_failed_login_attempt as _register_failed_login_attempt,
)
from ui.auth_sessions import (
    COOKIE_MANAGER_STATE_KEY,
    COOKIE_NAME,
)
from ui.auth_sessions import (
    cookies_ready_or_stop as _cookies_ready_or_stop,
)
from ui.auth_sessions import (
    create_session as _create_session,
)
from ui.auth_sessions import (
    delete_session as _delete_session,
)
from ui.auth_sessions import (
    get_username_for_session as _get_username_for_session,
)
from ui.auth_sessions import (
    reset_cookie_save_guard as _reset_cookie_save_guard,
)
from ui.auth_sessions import (
    save_cookies as _save_cookies,
)

# Direct Neon lookup fallback for username -> email mapping
try:
    from db.engine import get_neon_conn
    from db.schema import ensure_neon_users_schema
except ImportError:
    get_neon_conn = None
    ensure_neon_users_schema = None

# Optional helpers for username-login and persisting display names
try:
    from db.users import find_username_by_display_name
except ImportError:
    find_username_by_display_name = None

try:
    from db.users import update_neon_user_full_name
except ImportError:
    update_neon_user_full_name = None

# Optional: account creation helpers may exist depending on your db.users implementation.
try:
    from db.users import create_user_account  # preferred name
except ImportError:
    create_user_account = None

try:
    from db.users import create_neon_user  # alternate name
except ImportError:
    create_neon_user = None


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_AUTH_BACKEND_ERRORS = (RuntimeError, OSError, TypeError, ValueError, KeyError, AttributeError)


# --- Fallback: direct Neon query for display username -> email mapping ---
def _lookup_email_by_display_username(display_username: str) -> str | None:
    """Resolve display username (users.full_name) to email key (users.username)."""
    try:
        if get_neon_conn is None or ensure_neon_users_schema is None:
            return None

        name = (display_username or "").strip()
        if not name:
            return None

        conn = get_neon_conn()
        if conn is None:
            return None

        ensure_neon_users_schema(conn)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT username
            FROM users
            WHERE is_active = TRUE
              AND LOWER(TRIM(full_name)) = LOWER(TRIM(%s))
            ORDER BY created_at DESC NULLS LAST
            LIMIT 1
            """,
            (name,),
        )
        row = cur.fetchone()
        cur.close()
        conn.close()

        if not row:
            return None
        return row[0] if isinstance(row, tuple) else row.get("username")
    except _AUTH_BACKEND_ERRORS:
        return None


def _tier_display(tier_val: object) -> str:
    """Normalize a tier value (str / dict / Tier object) to its key string."""
    if tier_val is None:
        return "basic"
    if isinstance(tier_val, str):
        return tier_val.strip().lower() or "basic"
    # Tier dataclass/namedtuple exposes .key; debug dicts use 'tier_key'.
    key = getattr(tier_val, "key", None)
    if not key and isinstance(tier_val, dict):
        key = tier_val.get("tier_key") or tier_val.get("forced_tier_key")
    return str(key).strip().lower() if key else "basic"


def _poll_for_tier_upgrade(username: str, max_attempts: int = 10) -> None:
    """After Stripe checkout, webhook fires async. Poll DB until tier upgrades or we give up."""
    import time

    def _clear_stripe_params() -> None:
        # Strip return flags so refreshes / interactions don't re-trigger polling.
        for _p in ("checkout", "portal", "rt"):
            try:
                st.query_params.pop(_p, None)
            except _AUTH_BACKEND_ERRORS:
                pass

    current_key = _tier_display(st.session_state.get("tier", "basic"))
    attempt = st.session_state.get("_tier_poll_attempt", 0)
    if current_key != "basic" or attempt >= max_attempts:
        st.session_state.pop("_tier_poll_attempt", None)
        _clear_stripe_params()
        if current_key != "basic":
            st.success(f"🎉 Plan upgraded to **{current_key.title()}**! Enjoy your new features.")
        else:
            st.info("Upgrade is still processing. If you completed payment, refresh in a moment.")
        return
    with st.spinner("Activating your plan upgrade…"):
        time.sleep(2)
    t = _resolve_tier_key(username)
    if t and t != "basic":
        st.session_state["tier"] = t
        st.session_state["plan"] = t
        st.session_state.pop("_tier_poll_attempt", None)
        _clear_stripe_params()
        st.rerun()
    st.session_state["_tier_poll_attempt"] = attempt + 1
    st.rerun()


def _resolve_tier_key(username: str) -> str | None:
    """Return the resolved tier as a plain string.

    resolve_user_tier() returns a debug dict; the tier string lives under
    'tier_key'. Guard against it being a dict so we never store the whole
    structure into session_state['tier'].
    """
    try:
        from auth.tier_sync import resolve_user_tier
        result = resolve_user_tier(username)
    except (ImportError, *_AUTH_BACKEND_ERRORS):
        return None
    if isinstance(result, dict):
        key = result.get("tier_key") or result.get("forced_tier_key")
        return str(key) if key else None
    return str(result) if result else None


def auth_ui():
    # Reset the once-per-run cookie-save guard at the start of each run.
    _reset_cookie_save_guard()

    # --- URL token restore (Stripe redirect carries ?rt=<session_id>) ---
    # Browser cookies are unreliable across the Stripe round-trip on Streamlit
    # Cloud, so we also accept a session id passed back in the success_url.
    if "username" not in st.session_state:
        try:
            rt = (st.query_params.get("rt") or "").strip()
        except _AUTH_BACKEND_ERRORS:
            rt = ""
        if rt:
            u = _get_username_for_session(rt)
            if u:
                st.session_state["username"] = (u or "").strip().lower()
                t = _resolve_tier_key(st.session_state["username"])
                if t:
                    st.session_state["tier"] = t
                    st.session_state["plan"] = t
            # Strip the token from the URL so it isn't reused / bookmarked.
            try:
                st.query_params.pop("rt", None)
            except _AUTH_BACKEND_ERRORS:
                pass

    # --- Cookie session restore (persists login across refresh / Stripe redirects) ---
    cookies = _cookies_ready_or_stop()
    if cookies is not None and "username" not in st.session_state:
        try:
            sid = cookies.get(COOKIE_NAME)
        except _AUTH_BACKEND_ERRORS:
            sid = None
        if sid:
            u = _get_username_for_session(str(sid))
            if u:
                st.session_state["username"] = (u or "").strip().lower()
                # Re-read tier from DB so post-Stripe-upgrade redirects reflect the new plan.
                t = _resolve_tier_key(st.session_state["username"])
                if t:
                    st.session_state["tier"] = t
                    st.session_state["plan"] = t
            else:
                # Session expired or invalid — clear the stale cookie so the user
                # gets a clean login form rather than a silent broken state.
                try:
                    cookies.pop(COOKIE_NAME, None)
                    _save_cookies(cookies)
                except _AUTH_BACKEND_ERRORS:
                    pass
                st.info("Your session has expired. Please log in again.")

    # Handle Stripe redirects: ?checkout=success, ?checkout=cancel, ?portal=return
    try:
        checkout_flag = (st.query_params.get("checkout") or "").strip().lower()
        portal_flag = (st.query_params.get("portal") or "").strip().lower()
    except _AUTH_BACKEND_ERRORS:
        checkout_flag = ""
        portal_flag = ""

    _stripe_return = checkout_flag == "success" or portal_flag == "return"

    if _stripe_return and "username" in st.session_state:
        _poll_for_tier_upgrade(st.session_state["username"])

    if "username" not in st.session_state and _stripe_return:
        st.success(
            "Payment completed ✅ — log in below to access your upgraded plan. "
            "Your account has already been upgraded."
        )

    if "username" not in st.session_state and checkout_flag == "cancel":
        st.info("Checkout was cancelled. You can log in and try upgrading again anytime.")

    if "username" in st.session_state:
        username = (st.session_state["username"] or "").strip().lower()
        st.session_state["username"] = username
        display_name = st.session_state.get("display_name", username)
        return True, username, display_name

    # Basic login rate limiting to prevent brute-force attempts
    MAX_FAILED_ATTEMPTS = 5
    LOCKOUT_SECONDS = 300  # 5 minutes

    # If the user is currently locked out, show a message and block login
    if _is_login_locked():
        remaining = _lockout_remaining_seconds()
        if remaining < 1:
            remaining = 1
        st.error(f"Too many failed login attempts. Please try again in {remaining} seconds.")
        return False, None, None
    else:
        # Lockout expired; reset counters (best-effort)
        if st.session_state.get("login_locked_until"):
            _clear_failed_login_attempts()

    # Brand logo, centered above the login / sign-up tabs.
    try:
        from ui.header import _logo_path

        _lc1, _lc2, _lc3 = st.columns([1, 2, 1])
        with _lc2:
            st.image(_logo_path(), width="stretch")
    except (ImportError, *_AUTH_BACKEND_ERRORS):
        pass

    login_placeholder = st.empty()
    with login_placeholder.container():
        tabs = st.tabs(["🔐 Log In", "🧠 Sign Up"])

        # ---- Login tab ----
        with tabs[0]:
            st.markdown("### 🔐 Login")
            card = st.container(border=True)
            with card:
                username = st.text_input("Email or Username", key="login_username")
                password = st.text_input("Password", type="password", key="login_password")
                login_clicked = st.button("Login", key="login_button")
                st.page_link("pages/reset_password.py", label="Forgot password?", icon="🔑")

        # ---- Sign Up tab ----
        with tabs[1]:
            st.markdown("### 🧠 Create Your Free Account")
            st.caption("Start finding breakout opportunities in minutes.")

            signup_card = st.container(border=True)
            with signup_card:
                su_username = st.text_input(
                    "👤 Username",
                    key="signup_username",
                    placeholder="e.g. chimera47",
                    help="This will be your public display name. You can change it later."
                )

                su_email = st.text_input("✉️ Email", key="signup_email", placeholder="you@example.com")
                su_pw1 = st.text_input("🔒 Password", type="password", key="signup_password_1", placeholder="At least 8 characters")
                su_pw2 = st.text_input("🔒 Confirm Password", type="password", key="signup_password_2", placeholder="Re-enter password")
                su_agree = st.checkbox("I agree to use this tool for educational/informational purposes only.", key="signup_agree")
                signup_clicked = st.button("🟢 Create Free Account", key="signup_button")

            with st.expander("✅ What you get with a free Basic account", expanded=True):
                st.write("- ✔️ Access to curated breakout scans")
                st.write("- ✔️ Breakout Score (technical setup quality)")
                st.write("- ✔️ Interactive charts")
                st.write("- ✔️ Mobile-friendly results")
                st.write("- ✔️ No credit card required")
                st.caption("Upgrade anytime to unlock advanced filters, AI-powered rankings, and export features.")

            st.caption("🔐 Passwords are securely encrypted. We never store plain-text passwords.")

    # ----------------------
    # Sign Up handling
    # ----------------------
    if 'signup_clicked' in locals() and signup_clicked:
        email_raw = (su_email or "").strip().lower()
        username_raw = (su_username or "").strip()

        if not username_raw:
            st.error("Please choose a username.")
            return False, None, None

        if "@" in username_raw or " " in username_raw:
            st.error("Username cannot contain spaces or '@'.")
            return False, None, None

        if not email_raw or not EMAIL_RE.match(email_raw):
            st.error("Please enter a valid email address.")
            return False, None, None

        p1 = (su_pw1 or "").strip()
        p2 = (su_pw2 or "").strip()

        if not p1 or len(p1) < 8:
            st.error("Password must be at least 8 characters.")
            return False, None, None

        if p1 != p2:
            st.error("Passwords do not match.")
            return False, None, None

        if not su_agree:
            st.error("Please confirm the usage agreement to continue.")
            return False, None, None

        # Ensure Neon demo users are present; ignore failures.
        try:
            seed_neon_users_from_local()
        except _AUTH_BACKEND_ERRORS:
            pass

        try:
            users = load_users() or {}
        except _AUTH_BACKEND_ERRORS as e:
            st.error(f"Sign up failed while loading users: {e}")
            return False, None, None

        if email_raw in users or username_raw.lower() in users:
            st.error("That email or username is already taken.")
            return False, None, None

        # Create bcrypt hash
        pw_hash = bcrypt.hashpw(p1.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

        # Try to create the user in the DB via whichever helper exists.
        created_user = None
        create_fn = create_user_account or create_neon_user
        if create_fn is None:
            st.error(
                "Sign up is not enabled on this deployment yet (missing DB create_user function). "
                "Ask an admin to enable account creation."
            )
            return False, None, None

        try:
            # Prefer a simple signature: (username/email, password_hash, tier/plan)
            try:
                created_user = create_fn(
                    email=email_raw,
                    password_hash=pw_hash,
                    tier="basic",
                    full_name=username_raw,
                )
            except TypeError:
                try:
                    created_user = create_fn(username=email_raw, password_hash=pw_hash, tier="basic")
                except TypeError:
                    try:
                        created_user = create_fn(email=email_raw, password_hash=pw_hash, tier="basic")
                    except TypeError:
                        # Last-resort: just pass email + hash
                        created_user = create_fn(email_raw, pw_hash)
        except _AUTH_BACKEND_ERRORS as e:
            st.error(f"Sign up failed: {e}")
            return False, None, None

        # Best-effort: persist chosen Username into DB full_name so username-login works
        try:
            if update_neon_user_full_name is not None:
                update_neon_user_full_name(email_raw, username_raw)
        except _AUTH_BACKEND_ERRORS:
            pass

        # Send email verification link (best-effort — never blocks signup).
        try:
            from config import APP_BASE_URL
            from db.email_verification import create_verification_token
            from ui.email_utils import send_verification_email
            v_token = create_verification_token(email_raw)
            if v_token:
                verify_url = f"{APP_BASE_URL.rstrip('/')}/verify_email?token={v_token}"
                send_verification_email(to_address=email_raw, verify_url=verify_url)
        except _AUTH_BACKEND_ERRORS:
            pass

        # If db returned nothing, still proceed to login with defaults.
        user_rec = created_user if isinstance(created_user, dict) else {"password": pw_hash, "tier": "basic"}

        # Create session keys
        st.session_state["user_id"] = user_rec.get("id") or user_rec.get("user_id") or email_raw
        st.session_state["username"] = email_raw  # login identifier
        st.session_state["display_name"] = username_raw
        st.session_state["tier"] = user_rec.get("tier", "basic")
        st.session_state["plan"] = user_rec.get("plan", user_rec.get("tier", "basic"))
        st.session_state["is_admin"] = bool(user_rec.get("is_admin", False))
        # Persist login across refreshes using cookie session
        try:
            cookies2 = _cookies_ready_or_stop()
            if cookies2 is not None:
                sid = _create_session(email_raw)
                if sid:
                    cookies2[COOKIE_NAME] = sid
                    _save_cookies(cookies2)
        except _AUTH_BACKEND_ERRORS:
            pass

        # Clear any lockout counters
        _clear_failed_login_attempts()

        # Remove the auth UI and rerun into app
        login_placeholder.empty()
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()

    if login_clicked:
        # Normalize inputs (mobile keyboards often add whitespace/case)
        username_input = (username or "").strip()
        username = username_input.lower()
        raw_password = (password or "")
        password = raw_password.strip()

        if not username or not password:
            st.error("Please enter username and password.")
            # Do not count this as a brute-force attempt; user simply forgot fields.
            return False, None, None

        # DB-backed rate limit check (persistent across sessions/refreshes)
        if _is_login_rate_limited_db is not None:
            try:
                if _is_login_rate_limited_db(username):
                    st.error("Too many failed login attempts. Please try again in 10 minutes.")
                    return False, None, None
            except _AUTH_BACKEND_ERRORS:
                pass

        # Map display-username -> email (DB key) when user doesn't type an email
        login_key = username
        if "@" not in username:
            mapped = None
            # Prefer helper if present
            if find_username_by_display_name is not None:
                try:
                    mapped = find_username_by_display_name(username_input)
                except _AUTH_BACKEND_ERRORS:
                    mapped = None
            # Fallback: direct Neon query
            if not mapped:
                mapped = _lookup_email_by_display_username(username_input)
            if mapped:
                login_key = mapped.strip().lower()

        # Try to ensure Neon users are seeded from local demo users.
        # If this fails, we still fall back to local USERS_DB via load_users().
        try:
            seed_neon_users_from_local()
        except _AUTH_BACKEND_ERRORS:
            # Don't hard-fail login just because seeding didn't work.
            pass

        try:
            users = load_users() or {}
        except _AUTH_BACKEND_ERRORS as e:
            st.error(f"Login failed while loading users: {e}")
            return False, None, None

        # Primary lookup: email (stored as `username` in DB)
        user = users.get(login_key)

        # Secondary lookup: allow login by display username (full_name/display_name)
        if user is None and username:
            uname_guess = username
            for k, u in users.items():
                try:
                    dn = (u.get("display_name") or u.get("full_name") or u.get("name") or "").strip().lower()
                except _AUTH_BACKEND_ERRORS:
                    dn = ""
                if dn and dn == uname_guess:
                    user = u
                    login_key = k  # actual DB key (email)
                    break

        def _fail(msg: str, reason: str = "unknown") -> tuple:
            st.error(msg)
            _register_failed_login_attempt(MAX_FAILED_ATTEMPTS, LOCKOUT_SECONDS)
            if _record_login_attempt_db is not None:
                try:
                    _record_login_attempt_db(username, success=False, failure_reason=reason)
                except _AUTH_BACKEND_ERRORS:
                    pass
            return False, None, None

        if user is None:
            return _fail("User not found. Please use the email you signed up with, or your username.", reason="user_not_found")

        # Expect user dict to contain a 'password' field.
        # Supports legacy plain-text passwords and new bcrypt hashes, with auto-migration.
        stored_password = user.get("password")
        if stored_password is None:
            return _fail("User record is missing a password field.", reason="no_password_field")

        # Normalize stored password to a clean string (handles bytes from DB as well)
        if isinstance(stored_password, (bytes, bytearray)):
            stored_str = stored_password.decode("utf-8", errors="ignore")
        else:
            stored_str = str(stored_password)

        # Helper: does this string look like a bcrypt hash?
        def _looks_bcrypt(s: str) -> bool:
            return s.startswith("$2a$") or s.startswith("$2b$") or s.startswith("$2y$")

        if _looks_bcrypt(stored_str):
            # New-style bcrypt hash
            if not bcrypt.checkpw(password.encode("utf-8"), stored_str.encode("utf-8")):
                # Fallback: try raw input (older accounts may have accidental whitespace)
                if not bcrypt.checkpw(raw_password.encode("utf-8"), stored_str.encode("utf-8")):
                    return _fail("Incorrect password.", reason="wrong_password")
                else:
                    # Normalize forward: re-hash stripped password
                    try:
                        new_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
                        update_neon_user_password(login_key, new_hash)
                        user["password"] = new_hash
                    except _AUTH_BACKEND_ERRORS:
                        pass
        else:
            # Legacy plain-text password in DB
            if stored_str != password and stored_str != raw_password:
                return _fail("Incorrect password.")

            # Auto-migrate: convert legacy plain-text to bcrypt hash in Neon (normalize to stripped password)
            try:
                new_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
                update_neon_user_password(login_key, new_hash)
                user["password"] = new_hash
            except _AUTH_BACKEND_ERRORS:
                # Migration failure should not block a valid login
                pass

        # Success → create session keys based on user record.
        # Use whatever fields exist in the user dict; fall back where needed.
        st.session_state["user_id"] = user.get("id") or user.get("user_id") or login_key
        st.session_state["username"] = login_key

        display_name = user.get("display_name") or user.get("full_name") or user.get("name") or login_key
        st.session_state["display_name"] = display_name

        if "plan" in user:
            st.session_state["plan"] = user.get("plan")
        if "tier" in user:
            st.session_state["tier"] = user.get("tier")
        if "is_admin" in user:
            st.session_state["is_admin"] = bool(user.get("is_admin"))

        # Persist login across refreshes using cookie session
        try:
            cookies2 = _cookies_ready_or_stop()
            if cookies2 is not None:
                sid = _create_session(login_key)
                if sid:
                    cookies2[COOKIE_NAME] = sid
                    _save_cookies(cookies2)
        except _AUTH_BACKEND_ERRORS:
            pass

        # Reset failed-attempt tracking on successful login
        _clear_failed_login_attempts()
        if _record_login_attempt_db is not None:
            try:
                _record_login_attempt_db(login_key, success=True)
            except _AUTH_BACKEND_ERRORS:
                pass

        # Remove the login form from the screen after successful login.
        login_placeholder.empty()
        # No success banner to keep the UI clean after login.
        return True, login_key, display_name

    return False, None, None

def logout_and_reset_session() -> None:
    """Clear auth-related session state and rerun so the login screen is shown again.

    We avoid deleting *all* keys to prevent Streamlit from entering a weird state that
    can result in a blank screen. Instead, we clear the keys we know we own.
    """
    # Clear cookie-backed session (best-effort)
    _reset_cookie_save_guard()
    try:
        cookies = _cookies_ready_or_stop()
        if cookies is not None:
            sid = cookies.get(COOKIE_NAME)
            if sid:
                _delete_session(str(sid))
            try:
                cookies.pop(COOKIE_NAME, None)
            except _AUTH_BACKEND_ERRORS:
                pass
            _save_cookies(cookies)
    except _AUTH_BACKEND_ERRORS:
        pass

    auth_keys = [
        "user_id",
        "username",
        "display_name",
        "plan",
        "tier",
        "is_admin",
        "authentication_status",
        COOKIE_MANAGER_STATE_KEY,
        # Common app-level keys we control
        "results_df",
        "last_scan_at",
        "last_scan_universe",
        "scan_settings",
        "user_settings",
        # Filter / scan settings we want to reset on logout so defaults reload from DB on next login
        "universe",
        "min_price",
        "max_price",
        "min_dollar_vol",
        "include_ta",
        "apply_gap_filter",
        "show_diagnostics_ui",
        "min_gap",
        "top_n",
        "max_nasdaq_scan",
        "max_combo_scan",
        "premarket",
        "afterhours",
        "unusual_vol",
        "profile_loaded_for_user",
        "user_profile_loaded",
    ]
    try:
        for key in auth_keys:
            if key in st.session_state:
                st.session_state.pop(key, None)
    except _AUTH_BACKEND_ERRORS:
        # Best-effort cleanup; ignore any issues here.
        pass

    # Trigger a full rerun so the app re-evaluates from the top and shows login
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()
