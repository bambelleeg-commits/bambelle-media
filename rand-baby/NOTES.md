# Rand Baby cuffed-set photo cleanup

Reusable method for the 11 velvet cuffed matching-set supplier photos.
Goal: remove the Rand Baby logo (top-left wall) and the Code / Size / Package
text (bottom-left), keep the garment print, model, and room looking real, then
export a 4:5 store JPEG (1080×1350).

Do not restyle these into cream flat lays. These are cleaned lifestyle / hanging
/ lining photos.

## Run

Uses the venv at `/tmp/imgvenv` (opencv-python-headless + pillow + numpy), or
any environment with those packages.

```bash
/tmp/imgvenv/bin/python /workspace/bambelle-media/rand-baby/clean_rand_baby_photo.py \
  SRC.jpg DST.jpg --debug-dir /tmp/rand-baby-work
```

`--debug-dir` writes `debug-logo-mask.png`, `debug-text-mask.png`,
`debug-overlay.jpg` (logo=red, text=green) and `fullres-clean.jpg` (cleaned
frame before the 4:5 crop). Always inspect those before uploading.

## Method

### Logo (top-left wall)

1. Search only the upper-left ~28% × 22% of the frame, so the model’s face
   (pink cheeks) and the garment prints are never considered.
2. Detect the distinctive cyan/teal strokes of the baby icon and “Rand”, then
   nearby pink/magenta of “Baby”, as connected components.
3. Take the convex hull of that ink so the white badge interior and the swirl
   loop are solid, not left as bright ghosts.
4. Dilate, then absorb leftover anti-aliased outline flecks (these are often
   gray-cyan, not pure white) by comparing a ring around the hull to a median
   blur of the wall.
5. Fill with a **per-column color** sampled from clean wall *above and below*
   that column. These rooms have a strong left-to-right lighting gradient
   (window on the right); a single average color leaves a visible blotch.
6. Add film grain copied from the clean wall *above* the badge (or to its left).
   Never sample texture from below the logo — that strip often contains the
   flower arrangement and will ghost onto the wall.

Safety clamps: the mask is zeroed past 30% width and 24% height.

### Text (bottom-left)

1. Search only the extreme bottom-left (~18% height × 32% width) so dark
   garment prints on the lower pant leg are ignored.
2. Keep near-black, low-saturation ink components (the Code / Size / Package
   block), plus their antialiased edges.
3. Replace the *whole block* (a solid rectangle, not just the letter pixels)
   so gaps between letters do not leave ghost ink. Clip the right edge if a
   white sock sits beside the text.
4. Cover the block with a weave-aligned copy of clean rug from above, scored
   by how well the border matches, then feather only the edges so the interior
   is 100% clean rug.

### Crop and export

Source portraits are taller than 4:5. Crop with most of the trim taken from
the bottom (spare rug, often the former text block) and a little from the top.
Resize to 1080×1350, JPEG quality 93.

On the Garden Friends model shot this also crops away most of the inpainted
text area, which is ideal.

## What to watch on other designs

- **Model shots** (same template as Garden Friends): logo on the wall, text on
  the rug. The script is tuned for this.
- **Hanging / garment-only shots:** logo and text are usually in the same
  corners; confirm with the debug overlay that the mask does not touch the
  garment.
- **Lining close-ups:** may have little or no logo/text, or a tighter crop.
  If the detectors find nothing, the image is only cropped/resized. Check
  that a close-up of the cuff or fleece was not mistaken for the text block.
- If a logo sits closer to a curtain or a face than on Garden Friends, inspect
  the overlay before trusting the automatic mask.

## Source files (11 designs)

Only **Garden Friends** was on this box when the first preview was made. The
other hashes were listed in the chat but were not in the attachments folder.
Copy them into a working directory, then run the same script.

Girls

| Design | Kind | Attachment hash (prefix) |
|---|---|---|
| Bunny Confetti | model + lining close-up | `5d14a337…`, `45e799f7…` |
| Chick Hearts | model + lining close-up | `403d5bd6…`, `b29f452a…` |
| Garden Friends | model | `ce35d19e…` → `garden-friends-1.jpg` |
| Sunny Rainbow | hanging / still | `1cb51bf7…` |
| Blue Blossom Elephant | hanging / still | `64b0266f…` |
| Pastel Bunny Doodle | hanging / still | `d66b2d26…` |

Boys

| Design | Kind | Attachment hash (prefix) |
|---|---|---|
| Bow Tie Bunny | model + lining close-up | `f2ce4ed2…`, `3525fc9f…` |
| Teddy Stars | hanging / still | `23c07105…` |
| Buzzy Bees | hanging / still | `5a4130c3…` |
| Rainbow Pilots | hanging / still | `984ef564…` |
| Little Clouds | hanging / still | `9b016000…` |

Suggested output names: `<slug>-1.jpg` (hero: hanging/flat if it exists, else
model), `<slug>-2.jpg` (model if the hero was a hanging shot), `<slug>-lining.jpg`
for the close-up. Product cards should use: hanging/flat first, then model,
then lining.

## Group cover

Cleaned `group-cover.jpg` is 1040×2016 (tall crop, not 4:5 — a 4:5 crop would
clip either the raised fist or the sitting boys' feet). Forest Baby (top-right)
was cropped out rather than inpainted because he sits tight against Little
Clouds. Bow Tie Bunny (top-left, raised arm) and Little Clouds (center-top)
are fully in frame. Teddy Stars' right shoulder is slightly tight at the
right edge. The Rand Baby logo was filled on the curtain only, with the
raised fist restored from the original; the Code/Size/Package block was
removed with a tight box left of the sleeve. A slight bright patch remains
where the logo was. Do **not** upload the group cover to Shopify.

Do not commit `src/` (supplier-branded originals) or `__pycache__/`.

## Shopify (when all photos are ready)

Draft only, not published, no collections that would publish.

- Title: `<Design> Velvet Cuffed 2-Piece Set`
- Option Size: `3–4 Years` (price 335, cost 213, inventory 1, tracked, DENY)
  and `4–5 Years` (price 345, cost 225, inventory 1, tracked, DENY)
- Fabric in the description: soft velvet with cozy fleece lining; mention the
  cuffs; runs small (choose the bigger size if between sizes)
- Type `Kids Sets`, vendor Bambelle, tags `winter, velvet, kids, girls|boys, cuffed`
- Host cleaned JPEGs in this folder, `git pull --rebase origin main` then push,
  and use jsDelivr:
  `https://cdn.jsdelivr.net/gh/bambelleeg-commits/bambelle-media@main/rand-baby/<file>`
