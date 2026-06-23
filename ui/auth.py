import streamlit as st
import bcrypt
import re
import os
from datetime import datetime, timedelta, timezone
from typing import Optional
from db.users import load_users, seed_neon_users_from_local, update_neon_user_password
from ui.auth_lockout import (
    clear_failed_login_attempts as _clear_failed_login_attempts,
    is_login_locked as _is_login_locked,
    lockout_remaining_seconds as _lockout_remaining_seconds,
    register_failed_login_attempt as _register_failed_login_attempt,
)

# Cookie-based persistent sessions (recommended)
try:
    from streamlit_cookies_manager import EncryptedCookieManager
except Exception:
    EncryptedCookieManager = None

# Direct Neon lookup fallback for username -> email mapping
try:
    from db.engine import get_neon_conn
    from db.schema import ensure_neon_users_schema
except Exception:
    get_neon_conn = None
    ensure_neon_users_schema = None

# Optional helpers for username-login and persisting display names
try:
    from db.users import find_username_by_display_name
except Exception:
    find_username_by_display_name = None

try:
    from db.users import update_neon_user_full_name
except Exception:
    update_neon_user_full_name = None

# Optional: account creation helpers may exist depending on your db.users implementation.
try:
    from db.users import create_user_account  # preferred name
except Exception:
    create_user_account = None

try:
    from db.users import create_neon_user  # alternate name
except Exception:
    create_neon_user = None


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# -------------------- Cookie sessions (Neon-backed) --------------------
COOKIE_PREFIX = os.environ.get("COOKIE_PREFIX", "ai_scanner")
COOKIE_NAME = os.environ.get("COOKIE_NAME", f"{COOKIE_PREFIX}_sid")


def _get_secret(name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass
    return None


def _profile() -> str:
    return (os.environ.get("PROFILE") or os.environ.get("ENV") or "dev").strip().lower()


COOKIE_PASSWORD = _get_secret("COOKIE_PASSWORD")
if not COOKIE_PASSWORD and _profile() in {"dev", "local", "test"}:
    COOKIE_PASSWORD = "dev-only-cookie-password"


def _cookies_ready_or_stop() -> Optional["EncryptedCookieManager"]:
    """Return cookie manager if available+ready; otherwise show guidance and stop."""
    if EncryptedCookieManager is None:
        st.error(
            "Cookie sessions are not available because `streamlit-cookies-manager` is not installed. "
            "Add it to requirements.txt and redeploy."
        )
        return None
    if not COOKIE_PASSWORD:
        st.error("Cookie sessions require COOKIE_PASSWORD to be configured.")
        return None

    cookies = EncryptedCookieManager(prefix=COOKIE_PREFIX, password=COOKIE_PASSWORD)
    if not cookies.ready():
        # One rerun is needed to initialize cookies
        st.stop()
    return cookies


def _ensure_auth_sessions_schema(conn) -> None:
    """Create auth_sessions table if missing."""
    cur = conn.cursor()
    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
    except Exception:
        # If extension isn't available, we'll rely on uuid generation elsewhere.
        pass

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS auth_sessions (
          session_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
          username text NOT NULL,
          created_at timestamptz NOT NULL DEFAULT now(),
          expires_at timestamptz NOT NULL,
          last_seen_at timestamptz NOT NULL DEFAULT now()
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_auth_sessions_username ON auth_sessions(username);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_auth_sessions_expires ON auth_sessions(expires_at);")
    conn.commit()
    cur.close()


def _create_session(username: str, ttl_days: int = 14) -> Optional[str]:
    """Create a new session row and return session_id as str."""
    try:
        if get_neon_conn is None:
            return None
        u = (username or "").strip().lower()
        if not u:
            return None
        conn = get_neon_conn()
        if conn is None:
            return None
        _ensure_auth_sessions_schema(conn)

        expires = datetime.now(timezone.utc) + timedelta(days=int(ttl_days))
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO auth_sessions (username, expires_at)
            VALUES (%s, %s)
            RETURNING session_id;
            """,
            (u, expires),
        )
        row = cur.fetchone()
        conn.commit()
        cur.close()
        conn.close()
        return str(row[0]) if row else None
    except Exception:
        return None


def _get_username_for_session(session_id: str) -> Optional[str]:
    """Return username for a valid (unexpired) session_id."""
    try:
        if get_neon_conn is None:
            return None
        sid = (session_id or "").strip()
        if not sid:
            return None
        conn = get_neon_conn()
        if conn is None:
            return None
        _ensure_auth_sessions_schema(conn)

        cur = conn.cursor()
        cur.execute(
            """
            SELECT username
            FROM auth_sessions
            WHERE session_id = %s
              AND expires_at > now()
            LIMIT 1;
            """,
            (sid,),
        )
        row = cur.fetchone()
        if row:
            cur.execute("UPDATE auth_sessions SET last_seen_at = now() WHERE session_id = %s;", (sid,))
            conn.commit()
        cur.close()
        conn.close()
        return row[0] if row else None
    except Exception:
        return None


def _delete_session(session_id: str) -> None:
    try:
        if get_neon_conn is None:
            return
        sid = (session_id or "").strip()
        if not sid:
            return
        conn = get_neon_conn()
        if conn is None:
            return
        _ensure_auth_sessions_schema(conn)
        cur = conn.cursor()
        cur.execute("DELETE FROM auth_sessions WHERE session_id = %s;", (sid,))
        conn.commit()
        cur.close()
        conn.close()
    except Exception:
        return


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
    except Exception:
        return None


def auth_ui():
    # --- Cookie session restore (persists login across refresh / Stripe redirects) ---
    cookies = _cookies_ready_or_stop()
    if cookies is not None and "username" not in st.session_state:
        try:
            sid = cookies.get(COOKIE_NAME)
        except Exception:
            sid = None
        if sid:
            u = _get_username_for_session(str(sid))
            if u:
                st.session_state["username"] = (u or "").strip().lower()

    # If Stripe redirects back with ?checkout=success but this Streamlit session isn't authenticated,
    # it usually means the success/cancel URL is pointing at a different deployment (domain) than
    # the one that initiated checkout, so session_state is empty here.
    try:
        checkout_flag = (st.query_params.get("checkout") or "").strip().lower()
    except Exception:
        checkout_flag = ""

    if "username" not in st.session_state and checkout_flag in ("success", "cancel"):
        if checkout_flag == "success":
            st.info(
                "Payment completed ✅ — please log in to unlock your upgraded plan. "
                "If you keep landing here after checkout, update your billing service env vars "
                "APP_SUCCESS_URL / APP_CANCEL_URL to THIS deployment’s URL so Stripe redirects back "
                "to the same app instance."
            )
        else:
            st.info(
                "Checkout was cancelled. You can log in and try upgrading again anytime."
            )

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
        except Exception:
            pass

        try:
            users = load_users() or {}
        except Exception as e:
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
        except Exception as e:
            st.error(f"Sign up failed: {e}")
            return False, None, None

        # Best-effort: persist chosen Username into DB full_name so username-login works
        try:
            if update_neon_user_full_name is not None:
                update_neon_user_full_name(email_raw, username_raw)
        except Exception:
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
                    cookies2.save()
        except Exception:
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

        # Map display-username -> email (DB key) when user doesn't type an email
        login_key = username
        if "@" not in username:
            mapped = None
            # Prefer helper if present
            if find_username_by_display_name is not None:
                try:
                    mapped = find_username_by_display_name(username_input)
                except Exception:
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
        except Exception:
            # Don't hard-fail login just because seeding didn't work.
            pass

        try:
            users = load_users() or {}
        except Exception as e:
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
                except Exception:
                    dn = ""
                if dn and dn == uname_guess:
                    user = u
                    login_key = k  # actual DB key (email)
                    break

        if user is None:
            st.error("User not found. Please use the email you signed up with, or your username.")
            _register_failed_login_attempt(MAX_FAILED_ATTEMPTS, LOCKOUT_SECONDS)
            return False, None, None

        # Expect user dict to contain a 'password' field.
        # Supports legacy plain-text passwords and new bcrypt hashes, with auto-migration.
        stored_password = user.get("password")
        if stored_password is None:
            st.error("User record is missing a password field.")
            _register_failed_login_attempt(MAX_FAILED_ATTEMPTS, LOCKOUT_SECONDS)
            return False, None, None

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
                    st.error("Incorrect password.")
                    _register_failed_login_attempt(MAX_FAILED_ATTEMPTS, LOCKOUT_SECONDS)
                    return False, None, None
                else:
                    # Normalize forward: re-hash stripped password
                    try:
                        new_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
                        update_neon_user_password(login_key, new_hash)
                        user["password"] = new_hash
                    except Exception:
                        pass
        else:
            # Legacy plain-text password in DB
            if stored_str != password and stored_str != raw_password:
                st.error("Incorrect password.")
                _register_failed_login_attempt(MAX_FAILED_ATTEMPTS, LOCKOUT_SECONDS)
                return False, None, None

            # Auto-migrate: convert legacy plain-text to bcrypt hash in Neon (normalize to stripped password)
            try:
                new_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
                update_neon_user_password(login_key, new_hash)
                user["password"] = new_hash
            except Exception:
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
                    cookies2.save()
        except Exception:
            pass

        # Reset failed-attempt tracking on successful login
        _clear_failed_login_attempts()

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
    try:
        cookies = _cookies_ready_or_stop()
        if cookies is not None:
            sid = cookies.get(COOKIE_NAME)
            if sid:
                _delete_session(str(sid))
            try:
                cookies.pop(COOKIE_NAME, None)
            except Exception:
                pass
            cookies.save()
    except Exception:
        pass

    auth_keys = [
        "user_id",
        "username",
        "display_name",
        "plan",
        "tier",
        "is_admin",
        "authentication_status",
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
    except Exception:
        # Best-effort cleanup; ignore any issues here.
        pass

    # Trigger a full rerun so the app re-evaluates from the top and shows login
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()
