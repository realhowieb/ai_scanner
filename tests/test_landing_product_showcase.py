"""Landing-page product showcase remains authentic, light, and responsive."""
from pathlib import Path

from PIL import Image

from ui import landing

ROOT = Path(__file__).resolve().parents[1]
STUB_ASSETS = {
    "scanner": "data:image/webp;base64,scanner",
    "stair-stepper": "data:image/webp;base64,stair",
    "market-brief": "data:image/webp;base64,brief",
}


def test_showcase_defaults_to_scanner_and_preserves_view_order():
    html = landing.showcase_html(STUB_ASSETS)

    assert html.index('id="hsf-view-scanner" checked') < html.index('id="hsf-view-stair-stepper"')
    assert html.index('id="hsf-view-stair-stepper"') < html.index('id="hsf-view-market-brief"')
    assert "Ranked Opportunities" in html
    assert "Stair-Stepper" in html
    assert "Market Brief" in html


def test_primary_image_is_eager_and_secondary_images_are_lazy():
    html = landing.showcase_html(STUB_ASSETS)

    assert html.count('loading="eager"') == 1
    assert html.count('fetchpriority="high"') == 1
    assert html.count('loading="lazy"') == 2
    assert html.index("base64,scanner") < html.index('loading="eager"')


def test_showcase_has_keyboard_focus_and_mobile_crop_guards():
    html = landing.showcase_html(STUB_ASSETS)

    assert 'type="radio"' in html
    assert 'role="radiogroup"' in html
    assert ".hsf-show-control:focus-visible+.hsf-show-tab" in html
    assert "@media (max-width:520px)" in html
    assert ".hsf-show-media{aspect-ratio:1.15/1}" in html
    assert "overflow:hidden" in html


def test_showcase_assets_are_webp_retina_sized_and_compact():
    for _slug, _eyebrow, _title, _description, relative_path, _alt in landing.SHOWCASE_VIEWS:
        path = ROOT / relative_path
        assert path.exists(), relative_path
        assert path.suffix == ".webp"
        assert path.stat().st_size < 250_000
        with Image.open(path) as image:
            assert image.width >= 1440
            assert image.height >= 900


def test_showcase_copy_and_assets_exclude_private_or_admin_chrome():
    html = landing.showcase_html(STUB_ASSETS)
    lowered = html.lower()

    assert "actual product view" in lowered
    assert "admin" not in lowered
    assert "streamlitapp" not in lowered
    assert "@" not in " ".join(alt for *_view, alt in landing.SHOWCASE_VIEWS)


def test_hero_links_to_showcase_without_changing_signup_target():
    html = landing.hero_html("")

    assert 'href="#hsf-signup"' in html
    assert 'href="#hsf-product-showcase"' in html
