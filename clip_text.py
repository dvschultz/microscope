"""CLIP text view: which written words drive each unit.

    python clip_text.py                      # layer4_1 and layer4 of clip_rn50
    python clip_text.py --layers layer4 --k 10

Renders ~5,000 common English words and ImageNet class names as black-on-white text
(3 font/size variants, centered on a 224px canvas), runs them through CLIP RN50's
image encoder, and scores each unit by its spatial-max activation averaged over the
variants. This mirrors Microscope's "multimodal neuron" view (Goh et al. 2021).

Writes site/assets/clip_rn50/<layer>/text.json   {"<unit>": [[word, act], ...]} (top-k)
and cache/clip_rn50/text_<layer>.npy ([words, C] float16) plus cache/clip_rn50/words.txt.
"""
import argparse
import functools
import json
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from microscope.files import write_json

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache" / "clip_rn50"
SITE = ROOT / "site" / "assets" / "clip_rn50"
DATA = ROOT / "data" / "imagenet-val-256"
WORDLIST = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-usa-no-swears.txt"

# (font file candidates (a tuple, so load_font can cache), point size, uppercase)
VARIANTS = [
    (("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"), 40, False),
    (("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf", "/System/Library/Fonts/Supplemental/Times New Roman.ttf"), 44, False),
    (("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"), 30, True),
]


def build_words(n_common):
    """Most frequent English words (len >= 3) + ImageNet class names, deduplicated."""
    path = CACHE / "words.txt"
    if path.exists():
        return path.read_text().splitlines()
    from microscope.examples import ImageNetVal
    common = urllib.request.urlopen(WORDLIST, timeout=30).read().decode().split()
    common = [w for w in common if w.isalpha() and len(w) >= 3][:n_common]
    classes = [c.strip() for c in ImageNetVal(DATA).class_names()]
    words, seen = [], set()
    for w in common + classes:
        if w.lower() not in seen and len(w) <= 24:
            seen.add(w.lower())
            words.append(w)
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(words))
    return words


@functools.lru_cache(maxsize=None)
def load_font(cands, size):
    for f in cands:
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


def render(word, variant, size=224):
    cands, pt, upper = variant
    text = word.upper() if upper else word
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    while True:
        font = load_font(cands, pt)
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        if r - l <= size - 24 or pt <= 12:
            break
        pt -= 2
    draw.text(((size - (r - l)) / 2 - l, (size - (b - t)) / 2 - t), text, fill=(0, 0, 0), font=font)
    return np.asarray(img)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layers", nargs="*", default=["layer4_1", "layer4"])
    p.add_argument("--common", type=int, default=4200, help="frequent English words before adding class names")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--device")
    p.add_argument("--sheet", action="store_true", help="also save a few rendered samples to cache/clip_rn50/")
    args = p.parse_args()

    from tqdm import tqdm
    from microscope.device import pick_device
    from microscope.model import LayerCapture
    from microscope.models import get_spec

    spec = get_spec("clip_rn50")
    device = pick_device(args.device)
    model = spec.load(device)
    words = build_words(args.common)
    print(f"{len(words)} words x {len(VARIANTS)} variants")
    if args.sheet:
        for i, w in enumerate(words[:3] + words[-3:]):
            Image.fromarray(np.concatenate([render(w, v) for v in VARIANTS], 1)).save(CACHE / f"text_sample_{i}.png")

    cap = LayerCapture(model, args.layers, spec.layers, spec.modules)
    acts = {l: [] for l in args.layers}
    items = [(w, v) for w in words for v in VARIANTS]
    for i in tqdm(range(0, len(items), args.batch), unit="batch"):
        x = np.stack([render(w, v) for w, v in items[i:i + args.batch]])
        x = torch.from_numpy(x).to(device).permute(0, 3, 1, 2).float() / 255
        out = cap.run(model, spec.preprocess(x))
        for l in args.layers:
            acts[l].append(out[l].amax(dim=(2, 3)).float().cpu())
    cap.close()

    for l in args.layers:
        a = torch.cat(acts[l]).reshape(len(words), len(VARIANTS), -1).mean(1).numpy()   # [W, C]
        np.save(CACHE / f"text_{l}.npy", a.astype(np.float16))
        order = np.argsort(-a, axis=0)[:args.k]                                          # [k, C]
        res = {str(u): [[words[i], round(float(a[i, u]), 3)] for i in order[:, u]] for u in range(a.shape[1])}
        (SITE / l).mkdir(parents=True, exist_ok=True)
        write_json(SITE / l / "text.json", res, separators=(",", ":"))
        print(f"wrote {SITE / l / 'text.json'}")


if __name__ == "__main__":
    main()
