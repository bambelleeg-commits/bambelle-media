#!/usr/bin/env python3
"""Clean all Rand Baby cuffed-set photos (lifestyle, mid-shots, flats, lining)."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Reuse proven lifestyle helpers
sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean_rand_baby_photo import (  # noqa: E402
    detect_logo_mask as lifestyle_logo_mask,
    detect_text_mask as lifestyle_text_mask,
    inpaint_smooth,
    fill_rug_text,
    crop_to_aspect,
    resize_export,
)


def load(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def save_jpg(path: Path, img: np.ndarray, quality: int = 93) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path, format="JPEG", quality=quality, subsampling=1, optimize=True)


def classify(path: Path) -> str:
    n = path.stem.lower()
    if "group" in n:
        return "group"
    if "flat" in n:
        return "flat"
    if "lining" in n:
        return "lining"
    # mid-shot: this is *-1 and a sibling *-2 exists
    if n.endswith("-1"):
        sib = path.with_name(path.stem[:-2] + "-2" + path.suffix)
        if sib.exists():
            return "mid"
    return "lifestyle"


def cyan_mask(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    cyan = (hue >= 82) & (hue <= 115) & (sat > 50) & (val > 80)
    pink = ((hue >= 145) | (hue <= 12)) & (sat > 50) & (val > 80)
    return cyan, pink, hsv


def badge_from_ink(ink: np.ndarray, pad: int = 18, dilate: int = 17) -> np.ndarray:
    """Solid badge mask from cyan/pink ink, without long triangular spikes."""
    h, w = ink.shape
    if ink.sum() < 80:
        return np.zeros((h, w), np.uint8)
    u8 = ink.astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(u8, 8)
    keep = np.zeros((h, w), np.uint8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 40:
            keep[labels == i] = 255
    if keep.sum() == 0:
        return keep
    # Close and fill holes so the white badge interior is included
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    cnts, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros((h, w), np.uint8)
    cv2.drawContours(filled, cnts, -1, 255, thickness=cv2.FILLED)
    # Convex hull closes the swirl loop and letter counters, but only when
    # it does not jump to a distant print (e.g. an orange cherry on a hem).
    ys_k, xs_k = np.where(filled > 0)
    if len(xs_k) >= 8:
        hull_mask = np.zeros_like(filled)
        hull = cv2.convexHull(np.stack([xs_k, ys_k], axis=1))
        cv2.fillConvexPoly(hull_mask, hull, 255)
        if hull_mask.sum() <= filled.sum() * 1.65 + 2000:
            filled = hull_mask
    core = filled.copy()
    filled = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate)))
    # Cap only the halo, never punch holes in the filled badge interior
    dist = cv2.distanceTransform(np.where(core > 0, 0, 255).astype(np.uint8), cv2.DIST_L2, 5)
    filled[(core == 0) & (dist > dilate + 2)] = 0
    return filled


def estimate_plain_bg(img: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Median color of bright, low-texture corner / edge samples (white or light grey)."""
    h, w = img.shape[:2]
    patches = [
        img[8:70, 8:70],
        img[8:70, w - 70 : w - 8],
        img[h - 70 : h - 8, 8:70],
        img[h - 70 : h - 8, w - 70 : w - 8],
    ]
    pix = np.concatenate([p.reshape(-1, 3) for p in patches], axis=0)
    # prefer bright pixels (the backdrop), drop dark text if a corner has it
    lum = pix.mean(axis=1)
    bright = pix[lum > 200] if (lum > 200).sum() > 50 else pix
    return np.median(bright, axis=0).astype(np.float32)


def sample_local_bg(img: np.ndarray, mask: np.ndarray, bg: np.ndarray | None = None) -> np.ndarray:
    """Backdrop colour from a far ring, ignoring grey outline crumbs and garments."""
    far = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61)))
    far = (far > 0) & (cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))) == 0)
    # Stay near the hole so we don't wander into another surface
    if mask.sum():
        ys, xs = np.where(mask > 0)
        roi = np.zeros(mask.shape, np.uint8)
        y0, y1 = max(0, ys.min() - 80), min(mask.shape[0], ys.max() + 80)
        x0, x1 = max(0, xs.min() - 80), min(mask.shape[1], xs.max() + 80)
        roi[y0:y1, x0:x1] = 1
        far = far & (roi > 0)
    if far.sum() < 40:
        ring = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
        far = (ring > 0) & (mask == 0)
    if far.sum() < 20:
        if bg is not None:
            return bg.astype(np.float32)
        return estimate_plain_bg(img, mask)
    pix = img[far].reshape(-1, 3).astype(np.float32)
    med = np.median(pix, axis=0)
    dist = np.linalg.norm(pix - med.reshape(1, 3), axis=1)
    close = pix[dist <= 16]
    if len(close) < 15:
        close = pix[dist <= 28]
    if len(close) < 8:
        close = pix
    return np.median(close, axis=0).astype(np.float32)


def expand_outline(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Add leftover grey / cyan logo hooks around an incomplete badge mask."""
    if mask.sum() == 0:
        return mask
    bg = sample_local_bg(img, mask)
    ys, xs = np.where(mask > 0)
    y0, y1 = max(0, int(ys.min()) - 22), min(mask.shape[0], int(ys.max()) + 22)
    x0, x1 = max(0, int(xs.min()) - 22), min(mask.shape[1], int(xs.max()) + 22)
    near = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    window = np.zeros_like(mask)
    window[y0:y1, x0:x1] = 255
    near = (near > 0) & (window > 0) & (mask == 0)
    diff = np.linalg.norm(img.astype(np.float32) - bg.reshape(1, 1, 3), axis=2)
    lum = img.astype(np.float32).mean(axis=2)
    bg_lum = float(bg.mean())
    # Hooks are slightly darker than the backdrop, not hair / deep shadows
    # On near-white paper the sticker keyline is very faint (diff ~4-10)
    thr = 4 if bg_lum > 230 else 7
    leftover = near & (diff > thr) & (lum > bg_lum - 42) & (lum < bg_lum + 18)
    cyan, pink, _ = cyan_mask(img)
    leftover = leftover | (near & (cyan | pink) & (lum > bg_lum - 50))
    out = mask.copy()
    out[leftover] = 255
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    return out


def fill_plain(img: np.ndarray, mask: np.ndarray, bg: np.ndarray | None = None) -> np.ndarray:
    if mask.sum() == 0:
        return img
    local_bg = sample_local_bg(img, mask, bg)
    rng = np.random.default_rng(3)
    ys, xs = np.where(mask > 0)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    canvas = img.copy().astype(np.float32)
    region = np.empty((y1 - y0, x1 - x0, 3), np.float32)
    region[:] = local_bg.reshape(1, 1, 3)
    region += rng.normal(0, 0.6, region.shape)
    canvas[y0:y1, x0:x1] = np.clip(region, 0, 255)
    feather = cv2.GaussianBlur((mask > 0).astype(np.float32), (21, 21), 0)
    interior = cv2.erode((mask > 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    alpha = np.clip(np.maximum(feather, interior.astype(np.float32)), 0, 1)[..., None]
    return np.clip(img.astype(np.float32) * (1 - alpha) + canvas * alpha, 0, 255).astype(np.uint8)




def fill_clean(img: np.ndarray, mask: np.ndarray, extra_dilate: int = 11) -> np.ndarray:
    """Fill a logo/text hole with the true local backdrop colour (no garment clones)."""
    if mask.sum() == 0:
        return img
    mask = expand_outline(img, mask)
    mask = expand_outline(img, mask)
    if extra_dilate > 0:
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (extra_dilate, extra_dilate)))
    return fill_plain(img, mask)


def fill_from_neighbor(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # Kept as an alias so existing call sites stay valid
    return fill_clean(img, mask)


def protect_garment(img: np.ndarray, mask: np.ndarray, sat_cut: int = 35) -> np.ndarray:
    """Don't let a mask eat colorful garment prints (high local texture / saturation)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # garment prints are more saturated than a white/cream backdrop or have more edge energy
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Laplacian(gray, cv2.CV_32F)
    tex = cv2.GaussianBlur(np.abs(edges), (5, 5), 0)
    # Only apply this *inside* the mask, and only drop pixels that look like fabric
    # (not the badge itself, which is also saturated). Keep saturated pixels that
    # are in a compact badge-like region already in the mask — we only strip
    # isolated saturated dots far from the badge centroid? Safer: for flats,
    # the garment is mid-frame; we already ROI the corners. Skip aggressive
    # protection except: if a masked pixel is well outside the main badge
    # component, drop it.
    n, labels, stats, cents = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if n <= 1:
        return mask
    areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)]
    areas.sort(key=lambda t: -t[1])
    keep = np.zeros_like(mask)
    keep[labels == areas[0][0]] = 255
    # Also keep nearby pieces (the "Baby" wordmark next to the icon)
    near = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81)))
    for i, area in areas[1:]:
        if area >= 40 and near[labels == i].any():
            keep[labels == i] = 255
    return keep


def detect_badge_in_roi(img: np.ndarray, x0, y0, x1, y1, dilate=17) -> np.ndarray:
    h, w = img.shape[:2]
    cyan, pink, _ = cyan_mask(img)
    roi = np.zeros((h, w), dtype=bool)
    roi[y0:y1, x0:x1] = True
    ink = (cyan | pink) & roi
    # pink only if near cyan in this roi
    c_only = cyan & roi
    if c_only.sum() >= 80:
        # Keep large cyan pieces (icon + "Rand" strokes), drop distant tiny prints.
        n, labels, stats, _ = cv2.connectedComponentsWithStats(c_only.astype(np.uint8) * 255, 8)
        main = np.zeros((h, w), np.uint8)
        # locate the largest cyan blob, then keep other cyan blobs near it
        if n > 1:
            areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)]
            big = max(areas, key=lambda t: t[1])[0]
            main[labels == big] = 255
            near_big = cv2.dilate(main, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (90, 90)))
            for i, area in areas:
                if i == big or (area >= 40 and near_big[labels == i].any()):
                    main[labels == i] = 255
        else:
            main = c_only.astype(np.uint8) * 255
        # Generous reach so the pink "Baby" wordmark is included, but not
        # distant hem prints (those stay outside this neighbourhood).
        near = cv2.dilate(main, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (71, 71)))
        ink = ((c_only | pink) & roi & (near > 0))
    mask = badge_from_ink(ink, dilate=dilate)
    if mask.sum() == 0:
        return mask
    # Clip to a padded bbox of the actual ink, not the whole search ROI
    ys, xs = np.where(ink)
    if len(xs):
        clip = np.zeros((h, w), np.uint8)
        clip[max(0, int(ys.min()) - 32): min(h, int(ys.max()) + 32),
             max(0, int(xs.min()) - 32): min(w, int(xs.max()) + 32)] = 255
        mask = cv2.bitwise_and(mask, clip)
    # Absorb leftover outline flecks (grey-cyan hooks) next to the badge
    pad = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    ring = (pad > 0) & (mask == 0)
    med = cv2.medianBlur(img, 31)
    dist = np.linalg.norm(img.astype(np.float32) - med.astype(np.float32), axis=2)
    mask = np.where(ring & (dist > 12), 255, mask).astype(np.uint8)
    if 'clip' in dir() and clip is not None and clip.sum():
        mask = cv2.bitwise_and(mask, clip)
    return protect_garment(img, mask)


def detect_text_block(img: np.ndarray, x0, y0, x1, y1, on_fabric: bool = False) -> np.ndarray:
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    roi = np.zeros((h, w), np.uint8)
    roi[y0:y1, x0:x1] = 255
    # Ink is near-black; on white bg sat is low. On fabric, still low-sat black overlay.
    ink = (gray < 90) & (hsv[:, :, 1] < 90) & (roi > 0)
    if ink.sum() < 30:
        ink = (gray < 110) & (hsv[:, :, 1] < 110) & (roi > 0)
    if ink.sum() < 20:
        return np.zeros((h, w), np.uint8)
    u8 = ink.astype(np.uint8) * 255
    u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(u8, 8)
    keep = np.zeros((h, w), np.uint8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 25:
            keep[labels == i] = 255
    if keep.sum() == 0:
        return keep
    near = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    edge = (gray < 150) & (near > 0) & (roi > 0)
    mask = np.where(edge | (keep > 0), 255, 0).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    bx0, bx1 = max(0, int(xs.min()) - 14), min(w, int(xs.max()) + 16)
    by0, by1 = max(0, int(ys.min()) - 12), min(h, int(ys.max()) + 14)
    block = np.zeros((h, w), np.uint8)
    cv2.rectangle(block, (bx0, by0), (bx1, by1), 255, thickness=-1)
    # stay inside the search roi
    block = cv2.bitwise_and(block, roi)
    return block


def fill_fabric(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Clone nearby fabric over a small logo/text overlay on a garment."""
    if mask.sum() == 0:
        return img
    h, w = img.shape[:2]
    ys, xs = np.where(mask > 0)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    # Search band around the hole, excluding the hole
    search = np.zeros((h, w), np.uint8)
    search[max(0, y0 - 160): min(h, y1 + 80), max(0, x0 - 40): min(w, x1 + 160)] = 255
    search[mask > 0] = 0
    # Prefer mid-tones of the same garment, not walls
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # score shifts of the same-sized block
    best = None
    for dy in range(-140, 80, 2):
        for dx in range(-20, 120, 2):
            sy0, sy1 = y0 + dy, y1 + dy
            sx0, sx1 = x0 + dx, x1 + dx
            if sy0 < 0 or sx0 < 0 or sy1 > h or sx1 > w:
                continue
            if mask[sy0:sy1, sx0:sx1].any():
                continue
            if search[sy0:sy1, sx0:sx1].mean() < 200:
                continue
            src = img[sy0:sy1, sx0:sx1].astype(np.float32)
            dest = img[max(0, y0 - 8): min(h, y1 + 8), max(0, x0 - 8): min(w, x1 + 8)]
            # compare a 8px halo if we can
            # simpler: mean color similarity + avoid very dark leftover
            if src.mean() < 40:
                continue
            # Must look like the same garment, not a nearby table or floor
            hole_ring = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
            hole_ring = (hole_ring > 0) & (mask == 0)
            if hole_ring.sum() > 20:
                ref = img[hole_ring].reshape(-1, 3).mean(axis=0)
                if np.linalg.norm(src.reshape(-1, 3).mean(axis=0) - ref) > 28:
                    continue
            # border score against dest immediately outside the hole
            band_pad = 8
            by0, bx0 = max(0, y0 - band_pad), max(0, x0 - band_pad)
            by1, bx1 = min(h, y1 + band_pad), min(w, x1 + band_pad)
            d_band = img[by0:by1, bx0:bx1].astype(np.float32)
            s_band = img[by0 + dy: by1 + dy, bx0 + dx: bx1 + dx]
            if s_band.shape != d_band.shape:
                continue
            local = np.ones(d_band.shape[:2], dtype=bool)
            local[y0 - by0: y1 - by0, x0 - bx0: x1 - bx0] = False
            if local.sum() < 20:
                continue
            d = float(((d_band - s_band.astype(np.float32))[local] ** 2).mean())
            if best is None or d < best[0]:
                best = (d, dy, dx)
    if best is None:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return cv2.cvtColor(cv2.inpaint(bgr, mask, 5, cv2.INPAINT_TELEA), cv2.COLOR_BGR2RGB)
    _, dy, dx = best
    pasted = img.copy()
    pasted[y0:y1, x0:x1] = img[y0 + dy: y1 + dy, x0 + dx: x1 + dx]
    feather = cv2.GaussianBlur(mask.astype(np.float32), (21, 21), 0)
    interior = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    alpha = np.clip(np.maximum(feather, interior.astype(np.float32)), 0, 1)[..., None]
    return (img.astype(np.float32) * (1 - alpha) + pasted.astype(np.float32) * alpha).astype(np.uint8)


def fill_surface(img: np.ndarray, mask: np.ndarray, allow_split: bool = True, forbid: np.ndarray | None = None) -> np.ndarray:
    """Shift-clone a similar surface (rug, parquet, or plain) over the text block."""
    if mask.sum() == 0:
        return img
    h, w = img.shape[:2]
    ys, xs = np.where(mask > 0)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    # If the block crosses two surfaces (parquet then rug), fill each band
    if allow_split and (y1 - y0) > 40:
        row_mean = img[y0:y1, x0:x1].reshape(y1 - y0, -1, 3).mean(axis=1)
        jumps = np.linalg.norm(np.diff(row_mean, axis=0), axis=1)
        # ignore tiny spikes; want a sustained surface change
        if jumps.max() > 18:
            split = int(np.argmax(jumps)) + 1
            if 8 < split < (y1 - y0) - 8:
                midy = y0 + split
                top = mask.copy(); top[midy:, :] = 0
                bot = mask.copy(); bot[:midy, :] = 0
                if top.sum() > 20 and bot.sum() > 20:
                    out = fill_surface(img, top, allow_split=False, forbid=forbid)
                    return fill_surface(out, bot, allow_split=False, forbid=forbid)
    ring = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    ring = (ring > 0) & (mask == 0)
    if ring.sum() < 20:
        return fill_plain(img, mask)
    ref = img[ring].reshape(-1, 3).mean(axis=0)
    # Very uniform bright surface (white wall / paper): just paint it
    if ref.mean() > 230 and img[ring].std() < 8:
        return fill_plain(img, mask)

    best = None
    band_pad = 16
    bx0, by0 = max(0, x0 - band_pad), max(0, y0 - band_pad)
    bx1, by1 = min(w, x1 + band_pad), min(h, y1 + band_pad)
    blocked = mask.copy()
    if forbid is not None:
        blocked = np.maximum(blocked, (forbid > 0).astype(np.uint8) * 255)
    bh, bw = y1 - y0, x1 - x0
    shifts = []
    # From above (typical floor/wall text)
    for dy in range(20, min(260, y0 - band_pad - 2), 3):
        for dx in range(-36, 37, 4):
            shifts.append((dy, dx))
    # Horizontal: same surface just to the right or left (rug, parquet)
    for dy in range(-20, 21, 5):
        for dx in list(range(max(8, bw // 2), min(w - x1, bw + 200), 6)) + list(range(-min(x0, bw + 200), -max(8, bw // 2), 6)):
            shifts.append((dy, dx))
    for dy, dx in shifts:
            if y0 - dy < 0 or y1 - dy > h or x0 + dx < 0 or x1 + dx > w:
                continue
            src_block = img[y0 - dy: y1 - dy, x0 + dx: x1 + dx]
            if src_block.shape[0] != (y1 - y0) or src_block.shape[1] != (x1 - x0):
                continue
            if blocked[y0 - dy: y1 - dy, x0 + dx: x1 + dx].any():
                continue
            if np.linalg.norm(src_block.reshape(-1, 3).mean(axis=0) - ref) > 14:
                continue
            if src_block.reshape(-1, 3).std() > img[ring].reshape(-1, 3).std() * 1.6 + 8:
                continue
            d_band = img[by0:by1, bx0:bx1].astype(np.float32)
            s_band = img[by0 - dy: by1 - dy, bx0 + dx: bx1 + dx]
            if s_band.shape != d_band.shape or mask[by0 - dy: by1 - dy, bx0 + dx: bx1 + dx].any():
                continue
            local = np.ones(d_band.shape[:2], dtype=bool)
            local[y0 - by0: y1 - by0, x0 - bx0: x1 - bx0] = False
            if local.sum() < 20:
                continue
            d = float(((d_band - s_band.astype(np.float32))[local] ** 2).mean())
            if best is None or d < best[0]:
                best = (d, dy, dx)
    if best is None:
        # Last resort: inpaint, but seed with a same-colour plane first so Telea
        # does not smear dark leftover ink into the wood/rug.
        seeded = fill_plain(img, mask, ref)
        bgr = cv2.cvtColor(seeded, cv2.COLOR_RGB2BGR)
        return cv2.cvtColor(cv2.inpaint(bgr, mask, 4, cv2.INPAINT_NS), cv2.COLOR_BGR2RGB)
    _, dy, dx = best
    pasted = img.copy()
    pasted[y0:y1, x0:x1] = img[y0 - dy: y1 - dy, x0 + dx: x1 + dx]
    feather = cv2.GaussianBlur(mask.astype(np.float32), (21, 21), 0)
    interior = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    alpha = np.clip(np.maximum(feather, interior.astype(np.float32)), 0, 1)[..., None]
    return (img.astype(np.float32) * (1 - alpha) + pasted.astype(np.float32) * alpha).astype(np.uint8)



def fill_leftover_logo(img: np.ndarray, out: np.ndarray, y0, x0, y1, x1) -> np.ndarray:
    """Second pass: any remaining cyan/pink badge ink in a corner ROI."""
    h, w = out.shape[:2]
    cyan, pink, _ = cyan_mask(out)
    roi = np.zeros((h, w), dtype=bool)
    roi[max(0, y0): min(h, y1), max(0, x0): min(w, x1)] = True
    ink = (cyan | pink) & roi
    if ink.sum() < 40:
        return out
    mask = badge_from_ink(ink, dilate=15)
    if mask.sum() < 40:
        mask = cv2.dilate(ink.astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    return fill_clean(out, mask, extra_dilate=9)


def clean_lifestyle(img: np.ndarray) -> np.ndarray:
    logo = lifestyle_logo_mask(img)
    text = lifestyle_text_mask(img)
    out = inpaint_smooth(img, logo)
    out = fill_surface(out, text)
    return out, logo, text


def clean_flat(img: np.ndarray) -> tuple:
    h, w = img.shape[:2]
    logo = detect_badge_in_roi(img, 0, 0, int(w * 0.32), int(h * 0.28), dilate=20)
    text = detect_text_block(img, 0, int(h * 0.72), int(w * 0.36), h)
    out = fill_from_neighbor(img, logo)
    out = fill_leftover_logo(img, out, 0, 0, int(h * 0.30), int(w * 0.38))
    out = fill_from_neighbor(out, text)
    return out, logo, text


def _smooth_backdrop(img: np.ndarray) -> np.ndarray:
    """True on the plain table / white paper, False on stitched fabric."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    tex = cv2.GaussianBlur(np.abs(cv2.Laplacian(gray, cv2.CV_32F)), (5, 5), 0)
    return tex < 8.0


def clean_lining(img: np.ndarray) -> tuple:
    h, w = img.shape[:2]
    # Logo bottom-left, code bottom-right, both on a plain cream/white table
    logo = detect_badge_in_roi(img, 0, int(h * 0.70), int(w * 0.40), h, dilate=13)
    smooth = _smooth_backdrop(img)
    cyan, pink, _ = cyan_mask(img)
    core = cv2.dilate(((cyan | pink) & (np.arange(h)[:, None] > int(h * 0.70)) & (np.arange(w)[None, :] < int(w * 0.40))).astype(np.uint8) * 255,
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    logo[(~smooth) & (core == 0)] = 0
    # Small halo on the table only — never grow onto the stitched hem
    if logo.sum():
        halo = cv2.dilate(logo, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        logo = np.where(smooth | (core > 0), halo, 0).astype(np.uint8)
    # "Code: 447" is a small label on the table; never a box over the print
    text = detect_text_block(img, int(w * 0.50), int(h * 0.90), w, h)
    lum = img.astype(np.float32).mean(axis=2)
    table = smooth & (lum > 226)
    text = np.where(table, text, 0).astype(np.uint8)
    text[: int(h * 0.90), :] = 0
    if text.sum() > 20000:
        # Solid rect climbed into the garment — keep only a tight halo around ink
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        ink = (gray < 90) & (hsv[:, :, 1] < 90) & (text > 0) & (np.arange(h)[:, None] > int(h * 0.90))
        text = cv2.dilate(ink.astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        text = np.where(table, text, 0).astype(np.uint8)
    # fill_plain only (no expand_outline grow on a textured table)
    out = fill_plain(img, logo)
    out = fill_leftover_logo(img, out, int(h * 0.70), 0, h, int(w * 0.42))
    out = fill_plain(out, text)
    return out, logo, text


def clean_mid(img: np.ndarray) -> tuple:
    h, w = img.shape[:2]
    cyan, pink, _ = cyan_mask(img)
    tl = int(cyan[: int(h * 0.30), : int(w * 0.32)].sum())
    bl = int(cyan[int(h * 0.65):, : int(w * 0.35)].sum())
    logo = np.zeros((h, w), np.uint8)
    text = np.zeros((h, w), np.uint8)
    out = img
    # Logo on the wall (top-left) vs on the garment (bottom-left)
    if tl >= 80 and tl >= bl:
        # Proven solid-hull mask (no letter-counter holes) + gradient inpaint
        logo = lifestyle_logo_mask(img)
        if logo.sum() == 0:
            logo = detect_badge_in_roi(img, 0, 0, int(w * 0.30), int(h * 0.28), dilate=17)
            logo[:, int(w * 0.32) :] = 0
            logo[int(h * 0.32) :, :] = 0
        out = inpaint_smooth(img, logo) if logo.sum() else img
    else:
        logo = detect_badge_in_roi(img, 0, int(h * 0.58), int(w * 0.36), int(h * 0.88), dilate=13)
        # Smooth bokeh floor/table vs printed fabric
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        tex = cv2.GaussianBlur(np.abs(cv2.Laplacian(gray, cv2.CV_32F)), (5, 5), 0)
        if logo.sum():
            ring = cv2.dilate(logo, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
            ring = (ring > 0) & (logo == 0)
            if ring.sum() and tex[ring].mean() < 6:
                out = fill_from_neighbor(img, logo)
            else:
                out = fill_fabric(img, logo)
        else:
            out = img
    # "Code: 447" sits under the badge on the floor, not on the garment
    text = detect_text_block(img, 0, int(h * 0.80), int(w * 0.24), h, on_fabric=False)
    if logo.sum() and text.sum():
        text[cv2.dilate(logo, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))) > 0] = 0
    if text.sum():
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        tex = cv2.GaussianBlur(np.abs(cv2.Laplacian(gray, cv2.CV_32F)), (5, 5), 0)
        ys, xs = np.where(text > 0)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0 = int(xs.min())
        # Clip the box at the left edge of the garment (texture jump)
        maxx = min(w, int(w * 0.45), int(xs.max()) + 1)
        if y1 > y0 and maxx > x0 + 10:
            col = tex[y0:y1, x0:maxx].mean(axis=0)
            # baseline is the leftmost 40px of the box (floor around "Code")
            base = float(col[: min(40, len(col))].mean())
            cut = None
            run = 0
            for i, v in enumerate(col):
                if v > base * 1.8 + 3:
                    run += 1
                    if run >= 6:
                        cut = x0 + i - run + 1
                        break
                else:
                    run = 0
            if cut is not None and cut > x0 + 20:
                text[:, cut:] = 0
    if text.sum():
        out = fill_clean(out, text, extra_dilate=6)
    return out, logo, text


def export(img: np.ndarray, width: int = 1080) -> np.ndarray:
    img = crop_to_aspect(img, 4 / 5)
    img = resize_export(img, width)
    target_h = int(round(width * 5 / 4))
    if img.shape[0] != target_h or img.shape[1] != width:
        img = cv2.resize(img, (width, target_h), interpolation=cv2.INTER_AREA)
    return img


def process_one(src: Path, dst: Path, debug_dir: Path | None = None) -> dict:
    kind = classify(src)
    img = load(src)
    if kind == "flat":
        out, logo, text = clean_flat(img)
    elif kind == "lining":
        out, logo, text = clean_lining(img)
    elif kind == "mid":
        out, logo, text = clean_mid(img)
    elif kind == "lifestyle":
        out, logo, text = clean_lifestyle(img)
    else:
        raise ValueError(f"unsupported kind {kind} for {src}")
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        overlay = img.copy()
        overlay[logo > 0] = (255, 0, 0)
        overlay[text > 0] = (0, 255, 0)
        stem = src.stem
        save_jpg(debug_dir / f"{stem}-overlay.jpg", overlay, 85)
        Image.fromarray(logo).save(debug_dir / f"{stem}-logo.png")
        Image.fromarray(text).save(debug_dir / f"{stem}-text.png")
        save_jpg(debug_dir / f"{stem}-fullres.jpg", out, 93)
    out = export(out)
    save_jpg(dst, out)
    return {
        "src": src.name,
        "dst": dst.name,
        "kind": kind,
        "logo_px": int(logo.sum() // 255),
        "text_px": int(text.sum() // 255),
        "size": out.shape[1::-1],
    }


def main():
    src_dir = Path("/workspace/bambelle-media/rand-baby/src")
    out_dir = Path("/workspace/bambelle-media/rand-baby")
    debug = Path("/tmp/rand-baby-work/debug")
    # Map rb-<name>.jpg → <name>.jpg  (drop rb- prefix)
    files = sorted(src_dir.glob("rb-*.jpg"))
    files = [p for p in files if "group" not in p.stem]
    only = set(sys.argv[1:]) if len(sys.argv) > 1 else None
    for p in files:
        if only and p.stem not in only and p.name not in only:
            continue
        dst = out_dir / (p.name.replace("rb-", "", 1))
        info = process_one(p, dst, debug)
        print(info)


if __name__ == "__main__":
    main()
