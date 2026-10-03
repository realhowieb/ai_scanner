from pathlib import Path

from ui import nav

ROOT = Path(__file__).resolve().parents[1]


def test_admin_console_is_in_an_admin_only_sidebar_section():
    sections = dict(nav._NAV_SECTIONS)
    assert ("pages/admin.py", "Admin Console", "🛠️") in sections["Admin"]
    assert "Admin" in nav.ADMIN_ONLY_SECTIONS


def test_admin_page_checks_identity_and_role_before_rendering_tools():
    source = (ROOT / "pages" / "admin.py").read_text()
    login_guard = source.index('if not username:')
    role_guard = source.index('if not bool(st.session_state.get("is_admin"))')
    render = source.index("render_admin_page(")
    assert login_guard < role_guard < render


def test_scanner_results_no_longer_render_admin_console():
    source = (ROOT / "ui" / "results_tabs.py").read_text()
    assert "render_admin_tab" not in source
    assert '🛠 Admin' not in source
    assert 'can_admin_panel' not in source


def test_provider_health_lives_in_the_admin_console_not_scanner():
    app = (ROOT / "app.py").read_text()
    assert "render_provider_health" not in app
    assert "🩺 Provider Health" not in app
    admin = (ROOT / "ui" / "admin_results_tab.py").read_text()
    diagnostics = admin.index('st.markdown("### Diagnostics")')
    panel = admin.index('with st.expander("🩺 Provider Health", expanded=False):')
    assert diagnostics < panel
    assert "render_provider_health()" in admin[panel:panel + 120]
