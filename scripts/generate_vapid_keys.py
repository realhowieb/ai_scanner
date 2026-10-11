"""Print a new VAPID key pair for browser notifications (api/webpush.py).

    python scripts/generate_vapid_keys.py

Set VAPID_PRIVATE_KEY (and VAPID_SUBJECT, e.g. mailto:you@example.com) on hsf-api.
The public key is derived from the private one; it's printed only for reference.
Changing the key later means every browser has to turn notifications on again.
"""
import base64

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def main() -> None:
    key = ec.generate_private_key(ec.SECP256R1())
    private = key.private_numbers().private_value.to_bytes(32, "big")
    public = key.public_key().public_bytes(Encoding.X962, PublicFormat.UncompressedPoint)
    print(f"VAPID_PRIVATE_KEY={_b64(private)}")
    print(f"# public key (served by GET /v1/web-push/config): {_b64(public)}")


if __name__ == "__main__":
    main()
