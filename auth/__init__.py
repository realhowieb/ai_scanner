"""
auth package initializer.

Exposes:
    - tiering (Tier, get_user_tier, USERS_DB, ADMIN_USERS)
"""

from .tiering import (
    Tier,
    get_user_tier,
    USERS_DB,
    ADMIN_USERS,
)
