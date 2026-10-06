"""Build a publishable copy of the viewer in dist/.

    python build_site.py                                   # dist/site + dist/media, media served at /media
    python build_site.py --asset-base https://media.example.com   # images on object storage (R2/S3)

dist/site/   static host (Cloudflare Pages etc.): index.html, app.js, style.css, manifests, all JSON, _redirects
dist/media/  images, upload to object storage:
               <model>/<layer>/vis/<unit>.webp           feature visualization (WebP q85)
               <model>/<layer>/examples/<unit>.webp      the unit's 16 dataset-example crops as one 4×4 sprite (128px cells, q80)

Incremental: an output is rebuilt only if missing or older than its inputs, so re-running after more
layers land in site/assets/ only processes the new units. Reads site/ and never modifies it. CPU only.
"""
import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "site"
DIST = ROOT / "dist"
COLS, CELL = 4, 128
N_EXAMPLES = COLS * COLS


def stale(out, inputs):
    if not out.exists():
        return True
    t = out.stat().st_mtime
    return any(p.stat().st_mtime > t for p in inputs)


def vis_job(args):
    src, out = args
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.webp")
    Image.open(src).convert("RGB").save(tmp, "WEBP", quality=85, method=4)
    os.replace(tmp, out)


def sprite_job(args):
    srcs, out = args
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet = Image.new("RGB", (COLS * CELL, COLS * CELL), (24, 25, 28))
    for i, src in enumerate(srcs):
        im = Image.open(src).convert("RGB").resize((CELL, CELL), Image.LANCZOS)
        sheet.paste(im, ((i % COLS) * CELL, (i // COLS) * CELL))
    tmp = out.with_suffix(".tmp.webp")
    sheet.save(tmp, "WEBP", quality=80, method=4)
    os.replace(tmp, out)


def collect_jobs(model, layer):
    src = SITE / "assets" / model / layer
    dst = DIST / "media" / model / layer
    vis, sprites = [], []
    for png in sorted((src / "vis").glob("*.png")) if (src / "vis").exists() else []:
        out = dst / "vis" / f"{png.stem}.webp"
        if stale(out, [png]):
            vis.append((png, out))
    for d in sorted((src / "examples").iterdir()) if (src / "examples").exists() else []:
        if not d.is_dir():
            continue
        crops = [d / f"{r}.jpg" for r in range(N_EXAMPLES)]
        crops = [c for c in crops if c.exists()]
        if not crops:
            continue
        out = dst / "examples" / f"{d.name}.webp"
        if stale(out, crops):
            sprites.append((crops, out))
    return vis, sprites


def rewrite_manifest(path, media_base):
    m = json.loads(path.read_text())
    for model in m.get("models", []):
        name = model["name"]
        model["paths"]["vis"] = f"{media_base}/{name}/{{layer}}/vis/{{unit}}.webp"
        model["paths"]["example"] = f"{media_base}/{name}/{{layer}}/examples/{{unit}}.webp"
        model["paths"]["sprite"] = {"cols": COLS, "cell": CELL}
    return m


def tree_stats(root, exts=None):
    n = size = 0
    for p in root.rglob("*"):
        if p.is_file() and (exts is None or p.suffix in exts):
            n += 1
            size += p.stat().st_size
    return n, size


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asset-base", default="/media",
                    help="URL prefix for images (default /media = same host; e.g. https://media.example.com for R2)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()
    base = args.asset_base.rstrip("/")

    site, media = DIST / "site", DIST / "media"
    site.mkdir(parents=True, exist_ok=True)
    media.mkdir(parents=True, exist_ok=True)

    manifests = [p for p in [SITE / "manifest.json", SITE / "manifest_sae.json"] if p.exists()]
    layers = []
    for mp in manifests:
        for model in json.loads(mp.read_text()).get("models", []):
            layers += [(model["name"], l["name"]) for l in model["layers"]]

    # images
    vis_jobs, sprite_jobs = [], []
    for model, layer in layers:
        v, s = collect_jobs(model, layer)
        vis_jobs += v
        sprite_jobs += s
    print(f"to build: {len(vis_jobs)} feature vis, {len(sprite_jobs)} example sprites ({args.workers} workers)")
    if vis_jobs or sprite_jobs:
        with ProcessPoolExecutor(args.workers) as pool:
            list(pool.map(vis_job, vis_jobs, chunksize=64))
            list(pool.map(sprite_job, sprite_jobs, chunksize=32))

    # JSON data (stays on the static host; the viewer fetches it from /assets/...)
    for p in (SITE / "assets").rglob("*.json"):
        out = site / p.relative_to(SITE)
        if stale(out, [p]):
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out)
    for mp in manifests:
        (site / mp.name).write_text(json.dumps(rewrite_manifest(mp, base), indent=1))
    for f in ["index.html", "app.js", "style.css"]:
        shutil.copy2(SITE / f, site / f)
    (site / "_redirects").write_text("/* /index.html 200\n")

    n_site, s_site = tree_stats(site)
    n_media, s_media = tree_stats(media)
    print(f"static site  dist/site   {n_site:>7} files  {s_site / 1e6:8.1f} MB")
    print(f"assets       dist/media  {n_media:>7} files  {s_media / 1e9:8.2f} GB   (images at {base}/...)")
    n_src, s_src = tree_stats(SITE / "assets", {".png", ".jpg"})
    print(f"source images site/assets {n_src:>6} files  {s_src / 1e9:8.2f} GB")


if __name__ == "__main__":
    main()
