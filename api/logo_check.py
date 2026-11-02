"""Is FMP's image for a ticker a usable logo, and is it light (needs a dark chip)?

FMP's public image URL sometimes returns a photo instead of a logo (TSEM), and many
logos are white on transparent (ANET, ON, VRTX), which vanish on a light tile. This
reads the PNG with the standard library only (no Pillow on the API) and answers:

    {"kind": "logo", "light": False}   show it on the normal light chip
    {"kind": "logo", "light": True}    show it on a dark chip
    {"kind": "none"}                   no logo (missing, not a PNG, or a photo): letter tile

Answers are cached a week in shared_cache. Best effort: any error reads as "none".
"""
from __future__ import annotations

import struct
import zlib
from typing import Any, Callable, Dict, Iterable, List, Optional

TTL_S = 7 * 24 * 3600
MAX_BYTES = 2_000_000
LIGHT_LUMA = 0.82          # mean luminance of the visible pixels above this = a light logo
TRANSPARENT_SHARE = 0.05   # at least this share of see-through pixels = a cut-out logo
PHOTO_TOP_SHARE = 0.30     # an opaque image whose most common (4-bit) color covers less than this is a photo
                           # (logos on a background: 62-95% one color; the TSEM photo: 5%)
_PNG = b"\x89PNG\r\n\x1a\n"
NONE: Dict[str, Any] = {"kind": "none"}


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def decode_png(data: bytes) -> Optional[Dict[str, Any]]:
    """{width, height, pixels: [(r, g, b, a)]} for 8-bit PNGs (gray, RGB, palette, with or
    without alpha); None for anything else."""
    if not data.startswith(_PNG):
        return None
    pos, idat, palette, trns = 8, [], None, None
    width = height = depth = ctype = interlace = None
    while pos + 8 <= len(data):
        length, kind = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if kind == b"IHDR":
            width, height, depth, ctype, _c, _f, interlace = struct.unpack(">IIBBBBB", body)
        elif kind == b"PLTE":
            palette = [tuple(body[i:i + 3]) for i in range(0, len(body) - 2, 3)]
        elif kind == b"tRNS":
            trns = body
        elif kind == b"IDAT":
            idat.append(body)
        elif kind == b"IEND":
            break
    if not width or depth != 8 or interlace or ctype not in (0, 2, 3, 4, 6) or width * height > 1_000_000:
        return None
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[ctype]
    stride = width * channels
    try:
        raw = zlib.decompress(b"".join(idat))
    except zlib.error:
        return None
    if len(raw) < (stride + 1) * height:
        return None
    prev = bytearray(stride)
    pixels: List[tuple] = []
    for y in range(height):
        f = raw[y * (stride + 1)]
        line = bytearray(raw[y * (stride + 1) + 1:(y + 1) * (stride + 1)])
        for i in range(stride):
            left = line[i - channels] if i >= channels else 0
            up = prev[i]
            if f == 1:
                line[i] = (line[i] + left) & 255
            elif f == 2:
                line[i] = (line[i] + up) & 255
            elif f == 3:
                line[i] = (line[i] + ((left + up) >> 1)) & 255
            elif f == 4:
                line[i] = (line[i] + _paeth(left, up, prev[i - channels] if i >= channels else 0)) & 255
        prev = line
        for x in range(0, stride, channels):
            if ctype == 6:
                pixels.append(tuple(line[x:x + 4]))
            elif ctype == 2:
                pixels.append((line[x], line[x + 1], line[x + 2], 255))
            elif ctype == 0:
                pixels.append((line[x], line[x], line[x], 255))
            elif ctype == 4:
                pixels.append((line[x], line[x], line[x], line[x + 1]))
            else:
                idx = line[x]
                r, g, b = palette[idx] if palette and idx < len(palette) else (0, 0, 0)
                a = trns[idx] if trns and idx < len(trns) else 255
                pixels.append((r, g, b, a))
    return {"width": width, "height": height, "pixels": pixels}


def classify(data: bytes) -> Dict[str, Any]:
    img = decode_png(data)
    if not img or not img["pixels"]:
        return NONE
    px = img["pixels"]
    visible = [p for p in px if p[3] >= 128]
    if not visible:
        return NONE
    see_through = 1 - len(visible) / len(px)
    if see_through < TRANSPARENT_SHARE:
        counts: Dict[tuple, int] = {}
        for r, g, b, _a in px:
            key = (r >> 4, g >> 4, b >> 4)
            counts[key] = counts.get(key, 0) + 1
        if max(counts.values()) / len(px) < PHOTO_TOP_SHARE:
            return NONE                       # no background color: an opaque photo, not a logo
        return {"kind": "logo", "light": False}   # carries its own background
    luma = sum(0.2126 * r + 0.7152 * g + 0.0722 * b for r, g, b, _a in visible) / (255.0 * len(visible))
    return {"kind": "logo", "light": luma > LIGHT_LUMA}


def _fetch(url: str) -> Optional[bytes]:
    import requests

    r = requests.get(url, timeout=5, stream=True)
    if r.status_code != 200:
        return None
    data = r.raw.read(MAX_BYTES + 1, decode_content=True)
    return None if len(data) > MAX_BYTES else data


def logo_info(ticker: str, *, fetch: Optional[Callable[[str], Optional[bytes]]] = None) -> Dict[str, Any]:
    from api import shared_cache
    from api.earnings_week import logo_url

    def load() -> Dict[str, Any]:
        try:
            data = (fetch or _fetch)(logo_url(ticker))
        except Exception:
            return NONE
        return classify(data) if data else NONE

    try:
        return shared_cache.cached(f"logo_info:v2:{str(ticker).upper()}", load, TTL_S,
                                   empty_ttl_s=TTL_S, is_empty=lambda v: False) or NONE
    except Exception:
        return NONE


def logo_infos(tickers: Iterable[str], *, workers: int = 12,
               fetch: Optional[Callable[[str], Optional[bytes]]] = None) -> Dict[str, Dict[str, Any]]:
    """{ticker: info} for many tickers, fetched in parallel (cached ones are instant)."""
    from concurrent.futures import ThreadPoolExecutor

    names = list(dict.fromkeys(t for t in tickers if t))
    if not names:
        return {}
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(names)))) as pool:
        return dict(zip(names, pool.map(lambda t: logo_info(t, fetch=fetch), names)))
