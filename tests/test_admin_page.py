from pathlib import Path

from ui import nav

ROOT = Path(__file__).resolve().parents[1]


def test_admin_console_is_in_an_admin_only_sidebar_section():
    sections = dict(nav._NAV_SECTIONS)
    assert ("pages/admin.py", "Admin Console", "🛠️") in sections["Admin"]
    assert "Admin" in nav.ADMIN_ONLY_SECTIONS


