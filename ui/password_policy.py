"""P1-52 — one password rule for sign-up, password reset and Admin "create user".

Length-first (NIST-style): no forced symbols or capitals, but reject passwords
that are short, common, keyboard/sequence patterns, repeated characters, or
built from the account's own email/username. Checked only when a password is
set, so existing passwords keep working. Pure; safe to import anywhere.
"""
from __future__ import annotations

from typing import Optional

MIN_LENGTH = 10

# Frequently breached passwords of MIN_LENGTH or more (lowercased).
COMMON = frozenset({
    "password123", "password1234", "password12345", "passw0rd123", "p@ssword123", "password!",
    "iloveyou123", "welcome123", "welcome1234", "letmein123", "football123", "baseball123",
    "sunshine123", "princess123", "trustno1234", "monkey12345", "dragon12345", "master12345",
    "superman123", "batman12345", "starwars123", "whatever123", "changeme123", "administrator",
    "admin12345", "admin123456", "abc1234567", "qwerty123456", "1q2w3e4r5t", "1q2w3e4r5t6y",
    "zaq12wsxcde", "q1w2e3r4t5", "q1w2e3r4t5y6", "passwordpassword", "stockmarket", "stocks12345",
    "trading123", "hsfinest123", "hsfinestai", "streamlit123",
})

# Runs of adjacent keys / characters that make up "pattern" passwords.
_SEQUENCES = (
    "qwertyuiopasdfghjklzxcvbnm", "qwertyuiop", "asdfghjkl", "zxcvbnm",
    "1234567890", "0987654321", "abcdefghijklmnopqrstuvwxyz",
    "1qaz2wsx3edc4rfv5tgb", "qazwsxedcrfvtgbyhnujm",
)


def _in_sequence(chunk: str) -> bool:
    return any(chunk in seq or chunk in seq[::-1] for seq in _SEQUENCES)


def _is_pattern(pw: str, max_pieces: int = 3) -> bool:
    """True when pw is made of at most `max_pieces` keyboard/alphabet/number runs
    of 3+ characters (e.g. "qwerty12345", "asdfghjkl123", "1234567890")."""
    n = len(pw)
    best = [None] * (n + 1)  # fewest pieces to cover pw[:i]
    best[0] = 0
    for i in range(n):
        if best[i] is None:
            continue
        for j in range(i + 3, n + 1):
            if _in_sequence(pw[i:j]) and (best[j] is None or best[j] > best[i] + 1):
                best[j] = best[i] + 1
    return best[n] is not None and best[n] <= max_pieces


def _is_repeated(pw: str) -> bool:
    """A 1–3 character unit repeated (e.g. "abababab"), ignoring trailing digits/symbols."""
    core = pw.rstrip("0123456789!@#$%^&*?.")
    if len(core) < 6:
        return False
    return any(len(core) % k == 0 and core == core[:k] * (len(core) // k) for k in (1, 2, 3))


def password_problem(password: str, *, email: Optional[str] = None,
                     username: Optional[str] = None) -> Optional[str]:
    """None if the password is acceptable, else one plain sentence saying why."""
    pw = password or ""
    if len(pw) < MIN_LENGTH:
        return f"Password must be at least {MIN_LENGTH} characters."
    low = pw.lower()
    if low in COMMON:
        return "That password is too common. Choose something less guessable."
    if len(set(low)) <= 3 or _is_repeated(low):
        return "Password can't be mostly the same few characters repeated."
    if _is_pattern(low):
        return "Password can't be a keyboard or number pattern (like qwerty or 12345)."
    for ident in (email, (email or "").split("@")[0], username):
        ident = (ident or "").strip().lower()
        if len(ident) >= 4 and ident in low:
            return "Password can't contain your email address or username."
    return None


RULES_HINT = (f"At least {MIN_LENGTH} characters. Avoid common passwords, keyboard patterns "
              "and your email or username. A short phrase works well.")
