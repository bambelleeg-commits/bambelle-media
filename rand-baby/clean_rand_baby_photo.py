#!/usr/bin/env python3
"""Clean Rand Baby supplier photos: remove logo (top-left) and Code/Size/Package text (bottom-left).

Reusable for the 11 cuffed velvet set designs. Keeps the garment/print untouched.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def detect_logo_mask(img: np.ndarray) -> np.ndarray:
    """Mask the Rand Baby badge in the top-left wall, never the model or garment."""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Tight search window: the supplier badge lives in the upper-left wall,
    # well left of a typical model. 28% x 22% stays off the face/garment.
    roi = np.zeros((h, w), np.uint8)
    roi[: int(h * 0.22), : int(w * 0.28)] = 255

    # The baby icon and "Rand" strokes are a distinctive cyan/teal.
    cyan = (hue >= 82) & (hue <= 115) & (sat > 50) & (val > 80) & (roi > 0)
    if cyan.sum() < 80:
        return np.zeros((h, w), np.uint8)

    # Keep cyan only in reasonably large blobs (ignore single-pixel noise)
    cyan_u8 = cyan.astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cyan_u8, 8)
    keep = np.zeros((h, w), np.uint8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 40:
            keep[labels == i] = 255
    if keep.sum() == 0:
        return np.zeros((h, w), np.uint8)

    ys, xs = np.where(keep > 0)
    # Expand to the right to catch "Rand Baby" lettering (pink + cyan),
    # and a little around the icon for the white outline.
    x0 = max(0, int(xs.min()) - 20)
    x1 = min(w, int(xs.max()) + 30)  # will widen if pink is found just to the right
    y0 = max(0, int(ys.min()) - 20)
    y1 = min(h, int(ys.max()) + 20)

    # Pink/magenta lettering only if it sits next to the cyan badge (not cheeks)
    band = np.zeros((h, w), np.uint8)
    # Allow up to ~180px to the right of the cyan bbox for the wordmark
    band[y0: min(h, y1 + 10), x0: min(w, int(xs.max()) + 190)] = 255
    pink = ((hue >= 145) | (hue <= 12)) & (sat > 50) & (val > 80) & (band > 0)
    # Keep pink components close to cyan
    pink_u8 = pink.astype(np.uint8) * 255
    if pink_u8.sum():
        near_cyan = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61)))
        pink_u8 = cv2.bitwise_and(pink_u8, near_cyan)
        n2, lab2, st2, _ = cv2.connectedComponentsWithStats(pink_u8, 8)
        for i in range(1, n2):
            if st2[i, cv2.CC_STAT_AREA] >= 20:
                keep[lab2 == i] = 255
        ys, xs = np.where(keep > 0)
        x0 = max(0, int(xs.min()) - 18)
        x1 = min(w, int(xs.max()) + 22)
        y0 = max(0, int(ys.min()) - 18)
        y1 = min(h, int(ys.max()) + 20)

    # White badge backing / outline: pixels inside a dilation of the ink that
    # are clearly whiter than the surrounding wall, clipped to the bbox.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23))
    dil = cv2.dilate(keep, kernel, iterations=1)

    # Wall sample: rings around the badge, excluding the badge itself
    ring = cv2.dilate(dil, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)))
    ring = (ring > 0) & (dil == 0)
    # Stay in the left wall, avoid the curtain (higher x)
    ring[:, int(w * 0.32) :] = False
    if ring.sum() < 50:
        wall_mean = img[8:50, 8:50].reshape(-1, 3).mean(axis=0)
    else:
        wall_mean = img[ring].reshape(-1, 3).mean(axis=0)

    diff = np.linalg.norm(img.astype(np.float32) - wall_mean.reshape(1, 1, 3), axis=2)
    whiter = (img[:, :, 0] > wall_mean[0] + 10) & (img[:, :, 1] > wall_mean[1] + 6)
    not_wall = diff > 16
    interior = (dil > 0) & (not_wall | whiter | (keep > 0))

    clip = np.zeros((h, w), np.uint8)
    clip[y0:y1, x0:x1] = 255
    mask = (interior.astype(np.uint8) * 255)
    mask = cv2.bitwise_and(mask, clip)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))

    # Solid convex hull so interiors (the swirl loop, white badge backing) are included.
    ys_m, xs_m = np.where(mask > 0)
    if len(xs_m) >= 8:
        hull = cv2.convexHull(np.stack([xs_m, ys_m], axis=1))
        filled = np.zeros_like(mask)
        cv2.fillConvexPoly(filled, hull, 255)
        mask = filled
    # Dilate generously to catch the soft anti-aliased halo and the white
    # outline's faded corners (these are gray-cyan, not pure white).
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)), iterations=1)

    # Absorb leftover outline flecks: any pixel near the badge that does not
    # match the neighbouring wall (white backing, cyan fringe, pink fringe).
    pad = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    ring = (pad > 0) & (mask == 0)
    # Estimate local wall with a wide blur of pixels that are currently unmasked
    wall_src = img.copy()
    # A wide median of the unmasked image approximates the wall
    unmasked = img.copy()
    med = cv2.medianBlur(unmasked, 31)
    dist = np.linalg.norm(img.astype(np.float32) - med.astype(np.float32), axis=2)
    fringe = ring & (dist > 12)
    mask = np.where(fringe, 255, mask).astype(np.uint8)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    # Final safety: never extend right of 30% width or below 24% height
    mask[:, int(w * 0.30) :] = 0
    mask[int(h * 0.24) :, :] = 0
    return mask


def detect_text_mask(img: np.ndarray) -> np.ndarray:
    """Mask the black Code/Size/Package block. Ignores dark garment prints."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Text sits in the extreme bottom-left; the lower pant leg is higher than this.
    roi = np.zeros((h, w), np.uint8)
    roi[int(h * 0.72) :, : int(w * 0.34)] = 255

    # Ink is near-black and low-saturation; garment prints are colorful.
    ink = (gray < 85) & (hsv[:, :, 1] < 80) & (roi > 0)
    if ink.sum() < 40:
        # slightly more lenient if the photo is softer
        ink = (gray < 100) & (hsv[:, :, 1] < 100) & (roi > 0)
    if ink.sum() < 40:
        return np.zeros((h, w), np.uint8)

    ink_u8 = ink.astype(np.uint8) * 255
    # Close small gaps between letters of a line
    ink_u8 = cv2.morphologyEx(ink_u8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink_u8, 8)
    keep = np.zeros((h, w), np.uint8)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        # Text fragments are wide-ish or part of a line, not tiny specks
        if area >= 30 and (bw >= 8 or area >= 80):
            keep[labels == i] = 255
    if keep.sum() == 0:
        return np.zeros((h, w), np.uint8)

    # Antialiased gray edges around the ink, but only next to kept ink
    near = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    edge = (gray < 145) & (near > 0) & (roi > 0)
    mask = np.where(edge | (keep > 0), 255, 0).astype(np.uint8)

    ys, xs = np.where(mask > 0)
    x0, x1 = max(0, int(xs.min()) - 12), min(w, int(xs.max()) + 28)
    y0, y1 = max(0, int(ys.min()) - 10), min(h, int(ys.max()) + 3)

    # Solid block: gaps between letters still hold anti-aliased ink and must go.
    # Stay left of a white sock/foot if one sits to the right of the text.
    sock_zone = img[int(h * 0.70):, int(w * 0.15): int(w * 0.55)]
    # If a very bright low-sat object (sock) is right of the text, clip x1
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sat = hsv[:, :, 1]
    sock = (gray > 210) & (sat < 40) & (np.arange(h)[:, None] > int(h * 0.72)) & (np.arange(w)[None, :] > x1 - 10)
    if sock.any():
        sock_x = int(np.where(sock)[1].min())
        x1 = min(x1, sock_x - 2) if sock_x - 2 > x1 - 20 else x1

    block = np.zeros((h, w), np.uint8)
    # Slightly rounded rectangle looks less stamped than a hard box after blending
    cv2.rectangle(block, (x0, y0), (x1, y1), 255, thickness=-1)
    mask = block
    # Safety: do not climb into the mid-frame (protect the foot / cuff)
    mask[: int(h * 0.70), :] = 0
    # If the bottom of the box steps onto a high-texture surface (jute rug),
    # trim those rows so a parquet clone is not pasted onto the rug.
    if mask.sum():
        ys, xs = np.where(mask > 0)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        if y1 - y0 > 30:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            tex = cv2.GaussianBlur(np.abs(cv2.Laplacian(gray, cv2.CV_32F)), (5, 5), 0)
            row_tex = tex[y0:y1, x0:x1].mean(axis=1)
            ref = float(row_tex[: max(8, (y1 - y0) // 3)].mean())
            for i in range(int((y1 - y0) * 0.60), y1 - y0):
                if row_tex[i] > ref * 2.2 + 4:
                    mask[y0 + i :, :] = 0
                    break
    return mask


def inpaint_smooth(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill a smooth wall by interpolating each column from clean wall above and below.

    These photos have a strong left-to-right lighting gradient (window on the right).
    A per-column lerp matches that gradient much more closely than a single plane,
    and never samples the curtain.
    """
    if mask.sum() == 0:
        return img
    h, w = img.shape[:2]
    out = img.copy().astype(np.float32)
    ys, xs = np.where(mask > 0)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    halo = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    # For each column, gather wall samples above and below the badge
    cols = {}
    for x in range(max(0, x0 - 4), min(w, x1 + 4)):
        above = img[0:y0, x]
        below = img[y1:min(h, y1 + 80), x]
        a_ok = halo[0:y0, x] == 0
        b_ok = halo[y1:min(h, y1 + 80), x] == 0
        samples = []
        if a_ok.any():
            samples.append(above[a_ok])
        if b_ok.any():
            samples.append(below[b_ok])
        if samples:
            pix = np.concatenate(samples, axis=0)
            cols[x] = np.median(pix, axis=0).astype(np.float32)

    if not cols:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return cv2.cvtColor(cv2.inpaint(bgr, mask, 7, cv2.INPAINT_TELEA), cv2.COLOR_BGR2RGB)

    # Smooth the 1-D color curve across x so neighbouring columns agree
    xs_k = np.array(sorted(cols))
    curve = np.stack([cols[x] for x in xs_k], axis=0)
    # Small moving average
    k = 7
    pad = np.pad(curve, ((k // 2, k // 2), (0, 0)), mode="edge")
    kernel = np.ones((k, 1), dtype=np.float32) / k
    # manual conv
    smooth = np.stack([np.convolve(curve[:, c], np.ones(k) / k, mode="same") for c in range(3)], axis=1)
    # For edges, convolve same already tapers; good enough.
    lut = {int(x): smooth[i] for i, x in enumerate(xs_k)}

    rng = np.random.default_rng(2)

    for y, x in zip(ys, xs):
        color = lut.get(x)
        if color is None:
            nearest = xs_k[np.argmin(np.abs(xs_k - x))]
            color = lut[int(nearest)]
        out[y, x] = color

    # Real wall residual from the clean strip ABOVE the badge (below it there
    # are often flowers / furniture). Tile if the strip is short.
    tex = None
    if y0 >= 12:
        src_y0 = 0
        src_y1 = y0
        tex = img[src_y0:src_y1, x0:x1].astype(np.float32)
    if tex is None or tex.shape[0] < 8:
        # fallback: left of the badge
        if x0 >= 16:
            tex = img[y0:y1, 0:x0].astype(np.float32)
    if tex is not None and tex.size:
        reps_y = int(np.ceil((y1 - y0) / tex.shape[0]))
        reps_x = int(np.ceil((x1 - x0) / tex.shape[1]))
        tex = np.tile(tex, (reps_y, reps_x, 1))[: y1 - y0, : x1 - x0]
        local = cv2.GaussianBlur(tex, (0, 0), 3)
        residual = (tex - local) * 0.85
        region = out[y0:y1, x0:x1]
        m = mask[y0:y1, x0:x1] > 0
        region[m] = np.clip(region[m] + residual[m], 0, 255)
        out[y0:y1, x0:x1] = region
    else:
        grain = rng.normal(0, 1.0, (y1 - y0, x1 - x0, 3)).astype(np.float32)
        region = out[y0:y1, x0:x1]
        m = mask[y0:y1, x0:x1] > 0
        region[m] = np.clip(region[m] + grain[m], 0, 255)
        out[y0:y1, x0:x1] = region

    # Feather the edge into the original so the seam disappears
    feather = cv2.GaussianBlur((mask > 0).astype(np.float32), (15, 15), 0)
    interior = cv2.erode((mask > 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    alpha = np.clip(np.maximum(feather, interior.astype(np.float32)), 0, 1)[..., None]
    blended = img.astype(np.float32) * (1 - alpha) + out * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)



def fill_rug_text(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace the whole text block with a weave-aligned copy of nearby clean rug."""
    if mask.sum() == 0:
        return img
    h, w = img.shape[:2]
    ys, xs = np.where(mask > 0)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    box_h, box_w = y1 - y0, x1 - x0

    # Score candidate shifts by how well the *source* ring matches the *destination*
    # ring just outside the block (top and right edges, where the rug continues).
    ring = np.zeros((box_h, box_w), dtype=bool)
    margin = 14
    # Use a halo just outside? We only have dest pixels outside the mask.
    # Expand the comparison to a 16px band around the box, in the full image.
    band_pad = 16
    bx0, by0 = max(0, x0 - band_pad), max(0, y0 - band_pad)
    bx1, by1 = min(w, x1 + band_pad), min(h, y1 + band_pad)

    best = None  # (score, dy, dx)
    for dy in range(40, min(280, y0 - band_pad - 2), 1):
        for dx in range(-8, 9, 2):
            sy0, sy1 = y0 - dy, y1 - dy
            sx0, sx1 = x0 + dx, x1 + dx
            if sy0 < 0 or sx0 < 0 or sx1 > w:
                continue
            # Source must be clean (no text mask, no very dark ink)
            if mask[sy0:sy1, sx0:sx1].any():
                continue
            # Compare the destination band around the box with the shifted source band
            d_band = img[by0:by1, bx0:bx1].astype(np.float32)
            s_band = img[by0 - dy: by1 - dy, bx0 + dx: bx1 + dx]
            if s_band.shape != d_band.shape:
                continue
            if mask[by0 - dy: by1 - dy, bx0 + dx: bx1 + dx].any():
                continue
            # Stay on the rug: reject a source whose colour looks like floor/wood
            src_block = img[y0 - dy: y1 - dy, x0 + dx: x1 + dx]
            ring = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
            ring = (ring > 0) & (mask == 0)
            if ring.sum() > 20 and src_block.size:
                ref = img[ring].reshape(-1, 3).mean(axis=0)
                if np.linalg.norm(src_block.reshape(-1, 3).mean(axis=0) - ref) > 24:
                    continue
            # Only score pixels *outside* the text block (the visible seam)
            local = np.ones(d_band.shape[:2], dtype=bool)
            local[y0 - by0: y1 - by0, x0 - bx0: x1 - bx0] = False
            if local.sum() < 30:
                continue
            d = float(((d_band - s_band.astype(np.float32))[local] ** 2).mean())
            if best is None or d < best[0]:
                best = (d, dy, dx)

    if best is None:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return cv2.cvtColor(cv2.inpaint(bgr, mask, 5, cv2.INPAINT_TELEA), cv2.COLOR_BGR2RGB)

    _, dy, dx = best
    src_block = img[y0 - dy: y1 - dy, x0 + dx: x1 + dx]
    pasted = img.copy()
    pasted[y0:y1, x0:x1] = src_block

    # Feather only the edge of the block; the interior stays 100% clean rug.
    feather = cv2.GaussianBlur(mask.astype(np.float32), (31, 31), 0)
    # Boost the interior to 1.0 so no original ink leaks through the middle
    interior = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    alpha = np.clip(np.maximum(feather, interior.astype(np.float32)), 0, 1)[..., None]
    out = (img.astype(np.float32) * (1 - alpha) + pasted.astype(np.float32) * alpha).astype(np.uint8)
    return out


def crop_to_aspect(img: np.ndarray, aspect: float = 4 / 5) -> np.ndarray:
    """Crop to 4:5, taking more from the bottom (spare rug) than from above the head."""
    h, w = img.shape[:2]
    current = w / h
    if abs(current - aspect) < 0.01:
        return img
    if current < aspect:
        new_h = int(round(w / aspect))
        extra = h - new_h
        # Prefer trimming spare rug at the bottom (often the former text block).
        # If the leftover still includes a lot of the bottom, we'll also allow a
        # width trim path below when current > aspect after a bottom-first crop.
        top = int(round(extra * 0.12))
        bot = extra - top
        return img[top: h - bot, :]
    new_w = int(round(h * aspect))
    extra = w - new_w
    left = extra // 2
    return img[:, left: left + new_w]


def resize_export(img: np.ndarray, width: int = 1080) -> np.ndarray:
    h, w = img.shape[:2]
    if w == width:
        return img
    new_h = int(round(h * (width / w)))
    return cv2.resize(img, (width, new_h), interpolation=cv2.INTER_AREA)


def clean_image(src: Path, dst: Path, width: int = 1080) -> dict:
    img = np.array(Image.open(src).convert("RGB"))
    logo = detect_logo_mask(img)
    text = detect_text_mask(img)
    out = inpaint_smooth(img, logo)
    out = fill_rug_text(out, text)
    out = crop_to_aspect(out, 4 / 5)
    out = resize_export(out, width)
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(dst, format="JPEG", quality=93, subsampling=1, optimize=True)
    return {
        "src": str(src),
        "dst": str(dst),
        "logo_px": int(logo.sum() // 255),
        "text_px": int(text.sum() // 255),
        "size": out.shape[1::-1],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--width", type=int, default=1080)
    p.add_argument("--debug-dir", default="")
    args = p.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    img = np.array(Image.open(src).convert("RGB"))
    logo = detect_logo_mask(img)
    text = detect_text_mask(img)
    if args.debug_dir:
        d = Path(args.debug_dir)
        d.mkdir(parents=True, exist_ok=True)
        Image.fromarray(logo).save(d / "debug-logo-mask.png")
        Image.fromarray(text).save(d / "debug-text-mask.png")
        overlay = img.copy()
        overlay[logo > 0] = (255, 0, 0)
        overlay[text > 0] = (0, 255, 0)
        Image.fromarray(overlay).save(d / "debug-overlay.jpg", quality=85)
        # also save a full-res cleaned frame before crop, for inspection
        out = inpaint_smooth(img, logo)
        out = fill_rug_text(out, text)
        Image.fromarray(out).save(d / "fullres-clean.jpg", quality=93, subsampling=1)
    info = clean_image(src, dst, args.width)
    print(info)


if __name__ == "__main__":
    main()
