"""Write the HSF API's OpenAPI schema to a file (default web/openapi.json).

The Web v2 client types (web/src/api/schema.d.ts) are generated from it; CI fails
when the committed copy no longer matches the API. Run after changing the API:

    API_JWT_SECRET=<any 32+ chars> python scripts/export_openapi.py
    (cd web && npm run api:types)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main(argv: list[str]) -> int:
    out = Path(argv[1]) if len(argv) > 1 else ROOT / "web" / "openapi.json"
    os.environ.setdefault("API_JWT_SECRET", "export-only-not-a-secret-0123456789")
    from api.main import create_app

    spec = create_app().openapi()
    out.write_text(json.dumps(spec, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out} ({len(spec['paths'])} paths)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
