"""Point the HSF Telegram bot at the API (run once after creating the bot).

    TELEGRAM_BOT_TOKEN=... TELEGRAM_WEBHOOK_SECRET=... \
        python scripts/telegram_set_webhook.py https://hsf-api-v1.onrender.com

Use the same TELEGRAM_WEBHOOK_SECRET as the API (any 32+ random letters/digits).
Telegram then sends each message to <api>/v1/telegram/webhook with that secret.
Prints Telegram's answer; never prints the token.
"""
from __future__ import annotations

import os
import sys

import httpx


def main(argv: list[str]) -> int:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    secret = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "").strip()
    if not token or not secret or len(argv) < 2 or not argv[1].startswith("https://"):
        print(__doc__)
        return 2
    url = argv[1].rstrip("/") + "/v1/telegram/webhook"
    r = httpx.post(f"https://api.telegram.org/bot{token}/setWebhook", timeout=10,
                   json={"url": url, "secret_token": secret, "allowed_updates": ["message"],
                         "drop_pending_updates": True})
    print(r.status_code, r.json().get("description"))
    return 0 if r.status_code == 200 and r.json().get("ok") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
