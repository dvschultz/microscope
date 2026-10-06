"""Social preview cards (Open Graph / Twitter, 900x472 JPEG, laid out at 1200x630) for every viewer URL.

    python og_cards.py                 # all models, written into dist/media/og/ (run after build_site.py)
    python og_cards.py --only clip_rn50/layer4/13 --out /tmp/og   # one card, for checking the design

Unit card: the feature visualization on the left; on the right the network/layer/unit line, six top
dataset examples and the unit's auto label (or, for SAE features, its top ImageNet classes).
Layer, network and home cards are grids of feature visualizations. Paths mirror the site:
og/<model>/<layer>/<unit>.jpg, og/<model>/<layer>/layer.jpg, og/<model>/model.jpg, og/home.jpg;
SAE features live under og/inceptionv1_sae/<layer>/<f>.jpg. pages/functions/_middleware.js
points each page's og:image at the matching file. Incremental: existing cards are skipped.
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from microscope.files import write_json

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "site"
W, H = 1200, 630                  # layout canvas
OUT = (900, 472)                  # saved size: about a third smaller than 1200x630 and still a large preview on X/Facebook
BG, PANEL, TEXT, MUTED, ACCENT = (17, 18, 20), (30, 32, 36), (228, 229, 232), (138, 142, 151), (121, 169, 255)
FONT = "/System/Library/Fonts/HelveticaNeue.ttc"     # faces: 0 regular, 1 bold, 10 medium
SITE_NAME = "microscope.artificial-images.com"


@lru_cache(maxsize=None)
def font(size, face=0):
    try:
        return ImageFont.truetype(FONT, size, index=face)
    except OSError:
        return ImageFont.load_default(size)


def wrap(draw, text, fnt, width, max_lines):
    words, lines, line = text.split(), [], ""
    for w in words:
        trial = f"{line} {w}".strip()
        if draw.textlength(trial, font=fnt) <= width:
            line = trial
            continue
        if line:
            lines.append(line)
        line = w
        if len(lines) == max_lines:
            break
    if line and len(lines) < max_lines:
        lines.append(line)
    if len(lines) == max_lines and " ".join(lines) != " ".join(words):
        while lines[-1] and draw.textlength(lines[-1] + "…", font=fnt) > width:
            lines[-1] = lines[-1][:-1]
        lines[-1] = lines[-1].rstrip() + "…"
    return lines


def tile(path, size):
    im = Image.open(path).convert("RGB")
    return im.resize((size, size), Image.NEAREST if im.width * 3 <= size else Image.LANCZOS)


def footer(draw, x, y):
    draw.text((x, y), SITE_NAME, font=font(22, 10), fill=MUTED)


def unit_card(job):
    model_title, layer, unit, kind, vis, examples, title, out = job
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    pad, V = 40, H - 80
    if vis and Path(vis).exists():
        img.paste(tile(vis, V), (pad, pad))
    else:
        d.rectangle([pad, pad, pad + V, pad + V], fill=PANEL)
    x0, colw = pad + V + 36, W - (pad + V + 36) - pad
    d.text((x0, pad - 4), f"{model_title} · {layer}", font=font(26, 10), fill=MUTED)
    d.text((x0, pad + 28), f"{kind} {unit}", font=font(26, 1), fill=TEXT)
    # top dataset examples, 3 x 2
    gap, n_cols = 6, 3
    S = min((colw - gap * (n_cols - 1)) // n_cols, 150)   # leaves room for a two-line label
    gy = pad + 76
    for i, p in enumerate(examples[:6]):
        if Path(p).exists():
            img.paste(tile(p, S), (x0 + (i % n_cols) * (S + gap), gy + (i // n_cols) * (S + gap)))
    ty = gy + 2 * S + gap + 20
    for k, line in enumerate(wrap(d, title, font(40, 1), colw, 2)):
        d.text((x0, ty + k * 48), line, font=font(40, 1), fill=TEXT)
    footer(d, x0, H - pad - 22)
    save(img, out)


def grid_card(job):
    heading, sub, vis_paths, out = job
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    cols, rows, gap, top = 8, 3, 4, 150
    S = (W - 80 - gap * (cols - 1)) // cols
    for i, p in enumerate(vis_paths[:cols * rows]):
        if Path(p).exists():
            img.paste(tile(p, S), (40 + (i % cols) * (S + gap), top + (i // cols) * (S + gap)))
    d.text((40, 40), heading, font=font(52, 1), fill=TEXT)
    d.text((40, 102), sub, font=font(26, 0), fill=MUTED)
    footer(d, W - 40 - d.textlength(SITE_NAME, font=font(22, 10)), H - 40)
    save(img, out)


def save(img, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    img.resize(OUT, Image.LANCZOS).save(out, quality=78, optimize=True, progressive=True, subsampling=2)


def load(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def short(text, n=200):
    return text if len(text) <= n else text[:n - 1].rsplit(" ", 1)[0] + "…"


def jobs(out_root, only=None, meta_out=None):
    """Card jobs; also fills meta_out[(model, layer)] = {unit: [title, description]} for og.json."""
    meta_out = {} if meta_out is None else meta_out
    manifest = load(SITE / "manifest.json", {"models": []})
    sae_models = load(SITE / "manifest_sae.json", {"models": []})["models"]
    units, grids = [], []
    for m in manifest["models"]:
        name, mtitle = m["name"], m["title"]
        a = SITE / "assets" / name
        for l in m["layers"]:
            ln, d = l["name"], a / l["name"]
            labels, meta = load(d / "labels.json", {}), load(d / "units.json", {})
            page = meta_out.setdefault((name, ln), {})
            for u in range(l["channels"]):
                rows = meta.get(str(u), [])
                lab = labels.get(str(u))
                classes = ", ".join(dict.fromkeys(r["label"] for r in rows[:6]))
                title = lab["label"] if lab else classes or f"{ln} unit {u}"
                page[str(u)] = [lab["label"] if lab else "",
                                short(lab["description"] if lab else f"Top ImageNet examples: {classes}.")]
                units.append((mtitle, ln, u, "unit", d / "vis" / f"{u}.png",
                              [d / "examples" / str(u) / f"{r}.jpg" for r in range(6)], title,
                              out_root / name / ln / f"{u}.jpg"))
            grids.append((f"{mtitle} · {ln}", f"{l['channels']:,} units · {l['spatial'][0]}×{l['spatial'][1]}",
                          [d / "vis" / f"{u}.png" for u in range(min(24, l["channels"]))], out_root / name / ln / "layer.jpg"))
        mid = m["layers"][int(len(m["layers"]) * 0.6)]["name"]
        total = sum(l["channels"] for l in m["layers"])
        grids.append((mtitle, f"{len(m['layers'])} layers · {total:,} units · feature visualizations, dataset examples, circuits",
                      [a / mid / "vis" / f"{u}.png" for u in range(24)], out_root / name / "model.jpg"))
    for m in sae_models:
        name, base_title = m["name"], m["title"].split("·")[0].strip()
        a = SITE / "assets" / name
        for l in m["layers"]:
            ln, d = l["name"], a / l["name"]
            meta = load(d / "units.json", {})
            page = meta_out.setdefault((name, ln), {})
            for f in range(l["channels"]):
                rows = meta.get(str(f), [])
                classes = ", ".join(dict.fromkeys(r["label"] for r in rows[:6])) or f"{ln} SAE feature {f}"
                page[str(f)] = [classes, short(f"A sparse-autoencoder feature of {base_title} {ln}. Top ImageNet examples: {classes}.")]
                units.append((f"{base_title} SAE", ln, f, "feature", d / "vis" / f"{f}.png",
                              [d / "examples" / str(f) / f"{r}.jpg" for r in range(6)], classes,
                              out_root / name / ln / f"{f}.jpg"))
            grids.append((f"{base_title} · {ln} · SAE features", f"{l['channels']:,} features from a sparse autoencoder",
                          [d / "vis" / f"{u}.png" for u in range(24)], out_root / name / ln / "layer.jpg"))
    picks = [SITE / "assets" / m["name"] / m["layers"][int(len(m["layers"]) * 0.6)]["name"] / "vis" / f"{u}.png"
             for u in range(8) for m in manifest["models"]]
    grids.append(("Microscope", "Browse every unit of InceptionV1, VGG19 and CLIP ResNet-50", picks, out_root / "home.jpg"))
    if only:
        model, layer, unit = only.split("/")
        units = [j for j in units if j[7] == out_root / model / layer / f"{unit}.jpg"]
        grids = []
    return units, grids


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(ROOT / "dist" / "media" / "og"))
    p.add_argument("--site", default=str(ROOT / "dist" / "site"), help="where og.json page metadata goes (dist/site)")
    p.add_argument("--only", help="model/layer/unit, e.g. clip_rn50/layer4/13 (always re-rendered)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = p.parse_args()
    out = Path(args.out)
    page_meta = {}
    units, grids = jobs(out, args.only, page_meta)
    if not args.only:
        # small per-layer title/description files for pages/functions/_middleware.js (served as static JSON)
        for (model, layer), rows in page_meta.items():
            write_json(Path(args.site) / "assets" / model / layer / "og.json", rows, separators=(",", ":"))
    if not args.only:
        units = [j for j in units if not j[7].exists()]
        grids = [j for j in grids if not j[3].exists()]
    print(f"rendering {len(units)} unit cards, {len(grids)} grid cards -> {out}")
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(unit_card, units, chunksize=64))
        list(pool.map(grid_card, grids))
    n = sum(1 for _ in out.rglob("*.jpg"))
    size = sum(f.stat().st_size for f in out.rglob("*.jpg"))
    print(f"{n} cards, {size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
