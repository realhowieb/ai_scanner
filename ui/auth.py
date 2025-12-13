import streamlit as st
import bcrypt
import time
import re
from db.users import load_users, seed_neon_users_from_local, update_neon_user_password

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


# -------------------- Login lockout helpers --------------------

def _is_login_locked() -> bool:
    """Return True if the user is currently locked out due to too many failed logins."""
    try:
        locked_until = st.session_state.get("login_locked_until")
        if not locked_until:
            return False
        return time.time() < float(locked_until)
    except Exception:
        return False


def _lockout_remaining_seconds() -> int:
    try:
        locked_until = st.session_state.get("login_locked_until")
        if not locked_until:
            return 0
        remaining = int(float(locked_until) - time.time())
        return max(0, remaining)
    except Exception:
        return 0


def _register_failed_login_attempt(max_attempts: int = 5, lockout_seconds: int = 300) -> None:
    """Increment failed attempts and apply lockout if threshold exceeded."""
    try:
        failed = int(st.session_state.get("failed_login_attempts") or 0) + 1
    except Exception:
        failed = 1

    st.session_state["failed_login_attempts"] = failed
    if failed >= int(max_attempts):
        st.session_state["login_locked_until"] = time.time() + int(lockout_seconds)


def _clear_failed_login_attempts() -> None:
    st.session_state.pop("failed_login_attempts", None)
    st.session_state.pop("login_locked_until", None)


def auth_ui():
    if "username" in st.session_state:
        username = st.session_state["username"]
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
                username = st.text_input("Username", key="login_username")
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

        # If db returned nothing, still proceed to login with defaults.
        user_rec = created_user if isinstance(created_user, dict) else {"password": pw_hash, "tier": "basic"}

        # Create session keys
        st.session_state["user_id"] = user_rec.get("id") or user_rec.get("user_id") or email_raw
        st.session_state["username"] = email_raw  # login identifier
        st.session_state["display_name"] = username_raw
        st.session_state["tier"] = user_rec.get("tier", "basic")
        st.session_state["plan"] = user_rec.get("plan", user_rec.get("tier", "basic"))
        st.session_state["is_admin"] = bool(user_rec.get("is_admin", False))

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
        username = (username or "").strip().lower()
        password = (password or "").strip()

        if not username or not password:
            st.error("Please enter username and password.")
            # Do not count this as a brute-force attempt; user simply forgot fields.
            return False, None, None

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

        user = users.get(username)
        if user is None:
            st.error("User not found.")
            _register_failed_login_attempt(MAX_FAILED_ATTEMPTS, LOCKOUT_SECONDS)
            return False, None, None

        # Expect user dict to contain a 'password' field.
        # Supports legacy plain-text passwords and new bcrypt hashes, with auto-migration.
        stored_password = user.get("password")
        if stored_password is None:
            st.error("User record is missing a password field.")
            _register_failed_login_attempt(MAX_FAILED_ATTEMPTS, LOCKOUT_SECONDS)
            return False, None, None

        pwd_bytes = password.encode("utf-8")
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
            if not bcrypt.checkpw(pwd_bytes, stored_str.encode("utf-8")):
                st.error("Incorrect password.")
                _register_failed_login_attempt(MAX_FAILED_ATTEMPTS, LOCKOUT_SECONDS)
                return False, None, None
        else:
            # Legacy plain-text password in DB
            if stored_str != password:
                st.error("Incorrect password.")
                _register_failed_login_attempt(MAX_FAILED_ATTEMPTS, LOCKOUT_SECONDS)
                return False, None, None

            # Auto-migrate: convert legacy plain-text to bcrypt hash in Neon
            try:
                new_hash = bcrypt.hashpw(pwd_bytes, bcrypt.gensalt()).decode("utf-8")
                update_neon_user_password(username, new_hash)
                # Optionally update the in-memory user dict so future checks see the hash
                user["password"] = new_hash
            except Exception:
                # Migration failure should not block a valid login
                pass

        # Success → create session keys based on user record.
        # Use whatever fields exist in the user dict; fall back where needed.
        st.session_state["user_id"] = user.get("id") or user.get("user_id") or username
        st.session_state["username"] = username

        display_name = user.get("display_name") or user.get("full_name") or username
        st.session_state["display_name"] = display_name

        if "plan" in user:
            st.session_state["plan"] = user.get("plan")
        if "tier" in user:
            st.session_state["tier"] = user.get("tier")
        if "is_admin" in user:
            st.session_state["is_admin"] = bool(user.get("is_admin"))

        # Reset failed-attempt tracking on successful login
        _clear_failed_login_attempts()

        # Remove the login form from the screen after successful login.
        login_placeholder.empty()
        # No success banner to keep the UI clean after login.
        return True, username, display_name

    return False, None, None

def logout_and_reset_session() -> None:
    """Clear auth-related session state and rerun so the login screen is shown again.

    We avoid deleting *all* keys to prevent Streamlit from entering a weird state that
    can result in a blank screen. Instead, we clear the keys we know we own.
    """
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