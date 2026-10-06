"""Image pages: invert every units.json into ImageNet-val image -> the units it is a top-16 example for.

    python image_index.py                 # shards: site/assets/_images/<shard>.json
    python image_index.py --thumbs        # also 256px WebP thumbnails: site/assets/_images/thumbs/<id>.webp
    python image_index.py --stats         # count images/entries and estimate sizes, write nothing

Shard <s> holds val images s*SHARD .. (s+1)*SHARD-1:
    {"models": [...], "layers": [[model_idx, layer], ...],
     "images": {"<id>": {"label": <ImageNet class>, "e": [[layer_idx, unit, rank, act, top, left, size], ...]}}}
`layer_idx` indexes "layers"; top/left/size is the crop box in 256px image coordinates. Labels are not baked in:
the viewer reads each layer's labels.json, which other scripts regenerate. Layers come from the manifests, so the
index matches what the viewer knows about. CPU only; re-running rewrites the shards (a few seconds) and skips
thumbnails that already exist.
"""
import argparse
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "site"
OUT = SITE / "assets" / "_images"
THUMBS = OUT / "thumbs"
SHARD = 256          # images per shard -> 196 shards for 50k val images
N_TOP = 16
THUMB_QUALITY = 75


def manifest_layers():
    """(model, layer, units.json path) in manifest order: base networks first, then their SAE lenses."""
    out = []
    for mp in [SITE / "manifest.json", SITE / "manifest_sae.json"]:
        if not mp.exists():
            continue
        for model in json.loads(mp.read_text()).get("models", []):
            for l in model["layers"]:
                p = SITE / model["paths"]["units"].lstrip("/").format(layer=l["name"])
                if p.exists():
                    out.append((model["name"], l["name"], p))
    return out


def build_index():
    layers = manifest_layers()
    models = list(dict.fromkeys(m for m, _, _ in layers))
    keys = [[models.index(m), l] for m, l, _ in layers]
    images = {}
    for k, (_, _, p) in enumerate(layers):
        for unit, rows in json.loads(p.read_text()).items():
            for rank, r in enumerate(rows[:N_TOP]):
                img = images.setdefault(r["img"], {"label": r["label"], "e": []})
                img["e"].append([k, int(unit), rank, float(f"{r['act']:.4g}"), *r["box"]])
    for img in images.values():
        img["e"].sort(key=lambda e: (e[0], e[2], e[1]))
    return models, keys, images


def write_shards(models, keys, images):
    OUT.mkdir(parents=True, exist_ok=True)
    shards = {}
    for i in sorted(images):
        shards.setdefault(i // SHARD, {})[str(i)] = images[i]
    for old in OUT.glob("*.json"):
        if old.stem.isdigit() and int(old.stem) not in shards:
            old.unlink()
    sizes = []
    for s, imgs in shards.items():
        text = json.dumps({"models": models, "layers": keys, "images": imgs}, separators=(",", ":"))
        tmp = OUT / f"{s:03d}.json.tmp"
        tmp.write_text(text)
        os.replace(tmp, OUT / f"{s:03d}.json")
        sizes.append(len(text))
    return sizes


def write_thumbs(ids, workers):
    from microscope.examples import ImageNetVal
    THUMBS.mkdir(parents=True, exist_ok=True)
    todo = {i for i in ids if not (THUMBS / f"{i}.webp").exists()}
    print(f"thumbnails: {len(ids) - len(todo)} exist, {len(todo)} to write")
    if not todo:
        return

    def save(item):
        i, arr = item
        tmp = THUMBS / f"{i}.tmp.webp"
        Image.fromarray(arr).save(tmp, "WEBP", quality=THUMB_QUALITY, method=4)
        os.replace(tmp, THUMBS / f"{i}.webp")

    ds = ImageNetVal(ROOT / "data" / "imagenet-val-256")
    done = 0
    with ThreadPoolExecutor(workers) as pool:
        for start, imgs, _ in ds.iter_groups(todo):
            rows = [i for i in range(start, start + len(imgs)) if i in todo]
            list(pool.map(save, [(i, imgs[i - start]) for i in rows]))
            done += len(rows)
            if done and done % 5000 < len(rows):
                print(f"  {done}/{len(todo)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--thumbs", action="store_true", help="also write 256px WebP thumbnails of each image")
    ap.add_argument("--stats", action="store_true", help="only report counts and estimated sizes")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    models, keys, images = build_index()
    n_entries = sum(len(v["e"]) for v in images.values())
    most = max(images.items(), key=lambda kv: len(kv[1]["e"]))
    print(f"{len(keys)} layers over {len(models)} models, {n_entries} entries, {len(images)} unique images "
          f"(most: image {most[0]} with {len(most[1]['e'])})")
    if args.stats:
        est = len(json.dumps(images, separators=(",", ":")))
        print(f"index ~{est / 1e6:.1f} MB in {len({i // SHARD for i in images})} shards; "
              f"thumbnails ~{len(images) * 11.5e3 / 1e6:.0f} MB at 256px q{THUMB_QUALITY}")
        return
    sizes = write_shards(models, keys, images)
    print(f"shards: {len(sizes)} files, {sum(sizes) / 1e6:.1f} MB, largest {max(sizes) / 1e3:.0f} KB -> {OUT}")
    if args.thumbs:
        write_thumbs(images.keys(), args.workers)
        n = size = 0
        for p in THUMBS.glob("*.webp"):
            n += 1
            size += p.stat().st_size
        print(f"thumbnails: {n} files, {size / 1e6:.0f} MB -> {THUMBS}")


if __name__ == "__main__":
    main()
