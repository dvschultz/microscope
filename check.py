"""Contact sheet for spot-checking units: feature vis next to top dataset examples.

    python check.py --layer mixed4a --random 5          # 5 random units
    python check.py --layer mixed4a --units 379,385     # specific units
Writes cache/check/<layer>_<units>.jpg and prints simple image stats per vis
(low std = gray mush, high high-frequency energy = noise).
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent


def vis_stats(img):
    a = np.asarray(img, dtype=np.float32) / 255
    hf = np.abs(np.diff(a, axis=0)).mean() + np.abs(np.diff(a, axis=1)).mean()
    return dict(std=round(float(a.std()), 3), hf=round(float(hf), 3))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer", required=True)
    p.add_argument("--units")
    p.add_argument("--random", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=8, help="examples per row")
    args = p.parse_args()

    d = ROOT / "site" / "assets" / "inceptionv1" / args.layer
    units = json.loads((d / "units.json").read_text())
    if args.units:
        chosen = [int(u) for u in args.units.split(",")]
    else:
        chosen = sorted(random.Random(args.seed).sample(sorted(int(k) for k in units), args.random))

    V, E = 224, 112
    sheet = Image.new("RGB", (V + 4 + E * (args.n // 2), len(chosen) * (V + 4)), (17, 18, 20))
    draw = ImageDraw.Draw(sheet)
    for row, u in enumerate(chosen):
        y = row * (V + 4)
        vis = Image.open(d / "vis" / f"{u}.png").convert("RGB")
        sheet.paste(vis.resize((V, V)), (0, y))
        for i in range(args.n):
            ex = Image.open(d / "examples" / str(u) / f"{i}.jpg").resize((E, E))
            sheet.paste(ex, (V + 4 + (i % (args.n // 2)) * E, y + (i // (args.n // 2)) * E))
        draw.rectangle([0, y, 44, y + 14], fill=(0, 0, 0))
        draw.text((3, y + 1), str(u), fill=(255, 255, 255))
        labels = [r["label"] for r in units[str(u)][:args.n]]
        print(u, vis_stats(vis), "top act", units[str(u)][0]["act"], "|", ", ".join(labels))

    out = ROOT / "cache" / "check" / f"{args.layer}_{'-'.join(map(str, chosen))}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=90)
    print(out)


if __name__ == "__main__":
    main()
