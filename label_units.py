"""Natural-language labels for every unit, from a vision model reading contact sheets.

    python label_units.py --layers mixed4a --units 0-23     # pilot
    python label_units.py                                   # all layers; resumable
    python label_units.py --layers mixed3b,mixed4a --jobs 4 --batch 8
    python label_units.py --model clip_rn50                 # other models (microscope/models.py)

Each call to the Codex CLI gets one contact sheet: per unit a row with the unit id,
its feature vis and 12 dataset-example crops, plus the units' top ImageNet classes
in the prompt. Raw responses are cached per batch in cache/labels/[<model>/]<layer>/, and
site/assets/<model>/<layer>/labels.json is rebuilt from that cache:
    {"<unit>": {"label", "description", "confidence": "high|medium|low", "polysemantic"}}
"""
import argparse
import json
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from microscope.files import write_json

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache" / "labels"
MODEL = "gpt-6-astra"

NETWORK = {
    "inceptionv1": "InceptionV1 / GoogLeNet",
    "vgg19": "VGG19 (an ImageNet classifier)",
    "clip_rn50": "the CLIP ResNet-50 image encoder (trained contrastively on image-text pairs, so some units respond to written text and abstract concepts)",
}

DEPTHS = {}
DEPTHS["vgg19"] = {
    **{f"conv1_{i}": f"the {'first' if i == 1 else 'second'} conv layer ({3 + 2 * (i - 1)}px receptive field). Units detect colors, oriented edges and simple gradients, not objects." for i in (1, 2)},
    **{f"conv2_{i}": "an early conv layer (10-14px receptive field). Units detect edges, color contrasts and fine textures, not objects." for i in (1, 2)},
    **{f"conv3_{i}": "an early-middle conv layer (24-48px receptive field). Units detect textures, simple curves, corners and repeated patterns." for i in range(1, 5)},
    **{f"conv4_{i}": "a middle conv layer (~70-120px receptive field). Units detect complex textures, shapes and object parts (e.g. eyes, fur, wheels, text)." for i in range(1, 5)},
    **{f"conv5_{i}": "a late conv layer (~160-250px receptive field). Units detect object parts, whole objects and object categories." for i in range(1, 5)},
}
DEPTHS["clip_rn50"] = {
    "stem": "the stem output (13px receptive field). Units detect colors, oriented edges and simple gradients, not objects.",
    "layer1": "the end of the first residual stage (~37px receptive field). Units detect edges, color contrasts and low-level textures.",
    "layer2": "the end of the second residual stage (~97px receptive field). Units detect textures, curves, simple shapes and repeated patterns.",
    "layer3_2": "a middle block of the third stage. Units detect object parts, complex textures and shapes (e.g. eyes, fur, text, wheels).",
    "layer3": "the end of the third stage. Units detect object parts and part-groupings, sometimes letters or text.",
    "layer4_1": "a middle block of the last stage. Units detect objects, scenes, faces, text and sometimes abstract concepts that span images and words.",
    "layer4": "the last conv block, before attention pooling. Units detect objects, scenes, people, text and abstract concepts that span images and words.",
}

DEPTH = {
    "conv2d0": "the first conv layer (7x7 receptive field). Units detect oriented edges, color contrasts and simple frequency/Gabor patterns, not objects.",
    "conv2d1": "the second conv layer (11px receptive field). Units detect edges, color blobs and low-level textures, not objects.",
    "conv2d2": "the third conv layer (19px receptive field). Units detect lines, simple textures, color contrasts and early curve/corner fragments, not objects.",
    "mixed3a": "an early-middle layer (~59px receptive field). Units detect textures, curves, simple shapes, color-contrast boundaries and repeated patterns.",
    "mixed3b": "an early-middle layer. Units detect curves, simple shapes, textures, boundaries and early parts (e.g. eyes, fur texture).",
    "mixed4a": "a middle layer (~128px crops). Units detect complex textures, shapes and object parts (e.g. arcs/rims, spokes, eyes, fur, leaves, text).",
    "mixed4b": "a middle layer. Units detect object parts and complex patterns (e.g. eyes, snouts, wheels, fur, foliage, faces).",
    "mixed4c": "a middle-late layer. Units detect object parts and part-groupings (e.g. dog heads, wheels, car fronts, faces).",
    "mixed4d": "a late-middle layer. Units detect object parts and whole small objects (e.g. dog faces, cars, birds, text).",
    "mixed4e": "a late layer. Units detect objects and large object parts.",
    "mixed5a": "a late layer. Units detect whole objects and object categories, sometimes scene context.",
    "mixed5b": "the last conv layer. Units detect whole objects, object categories and scenes.",
}
DEPTHS["inceptionv1"] = DEPTH

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["units"],
    "properties": {"units": {"type": "array", "items": {
        "type": "object",
        "additionalProperties": False,
        "required": ["unit", "label", "description", "confidence", "polysemantic"],
        "properties": {
            "unit": {"type": "integer"},
            "label": {"type": "string"},
            "description": {"type": "string"},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "polysemantic": {"type": "boolean"},
        },
    }}},
}

PROMPT = """You are labeling neurons (channels) of {network} for an interpretability viewer like OpenAI Microscope.

Layer: {layer}, which is {depth}

The attached contact sheet has one row per unit. Each row starts with the unit id (white on black), then the unit's feature visualization (the large square on the left: a synthetic image optimized to maximally excite the unit), then 12 dataset-example crops (the ImageNet image regions that activate it most, strongest first). The crops are local patches centered on where the unit fires, so the unit usually responds to a part, texture or shape inside them, not to the whole ImageNet class.

Top ImageNet classes of each unit's 16 strongest examples (hint only; crops are local):
{classes}

For each unit, find what the feature visualization and the example crops have in common, and name it.
- label: a 2-6 word lowercase noun phrase naming the visual feature (e.g. "curve facing up-left", "dog snout", "radial spokes", "blue-orange color contrast", "high-frequency stripes"). Be specific and grounded in what you see; avoid vague words like "pattern" or "texture" on their own, and don't just repeat an ImageNet class unless the unit is really an object detector.
- description: one sentence on what visual pattern or object it responds to, mentioning what the examples share.
- confidence: high if the vis and examples clearly agree, medium if mostly, low if unclear.
- polysemantic: true only if the examples show two or more clearly unrelated things (say what they are in the description); otherwise false.

Units on this sheet, in row order: {units}. Return exactly one entry per unit. Do not run any tools; just look at the image and answer."""


def parse_units(spec, n):
    if not spec:
        return list(range(n))
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def font(size):
    for f in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def render_sheet(layer, units, out, n_ex=12):
    d = assets_dir() / layer
    V, E, L, G = 192, 96, 84, 6  # vis, example crop, id column, row gap
    cols = n_ex // 2
    sheet = Image.new("RGB", (L + V + 4 + E * cols, len(units) * (V + G) - G), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    f = font(26)
    for row, u in enumerate(units):
        y = row * (V + G)
        draw.rectangle([0, y, L - 4, y + V], fill=(0, 0, 0))
        draw.text((6, y + V // 2 - 14), str(u), fill=(255, 255, 255), font=f)
        sheet.paste(Image.open(d / "vis" / f"{u}.png").convert("RGB").resize((V, V)), (L, y))
        for i in range(n_ex):
            p = d / "examples" / str(u) / f"{i}.jpg"
            if p.exists():
                ex = Image.open(p).convert("RGB").resize((E, E), Image.BICUBIC)
                sheet.paste(ex, (L + V + 4 + (i % cols) * E, y + (i // cols) * E))
    sheet.save(out, quality=88)


def top_classes(rows, k=4):
    c = Counter(r["label"] for r in rows)
    return ", ".join(f"{name} x{n}" if n > 1 else name for name, n in c.most_common(k))


def label_batch(layer, units, meta, effort):
    d = cache_dir() / layer
    tag = f"{units[0]:04d}-{units[-1]:04d}"
    sheet, raw = d / f"{tag}.jpg", d / f"{tag}.json"
    if not sheet.exists():          # sheets may be pre-rendered elsewhere (--sheets-only)
        render_sheet(layer, units, sheet)
    prompt = PROMPT.format(
        network=NETWORK[ARGS_MODEL], layer=layer, depth=DEPTHS[ARGS_MODEL][layer], units=", ".join(map(str, units)),
        classes="\n".join(f"{u}: {top_classes(meta[str(u)])}" for u in units))
    schema = CACHE / "schema.json"
    for attempt in range(2):
        t = time.time()
        out = d / f"{tag}.out"
        out.unlink(missing_ok=True)
        r = subprocess.run(
            ["codex", "exec", "--skip-git-repo-check", "--ephemeral", "--ignore-user-config",
             "-s", "read-only", "-C", str(d), "-m", MODEL,
             "-c", f"model_reasoning_effort={effort}",
             "--output-schema", str(schema), "-o", str(out), "-i", str(sheet), "--", prompt],
            stdin=subprocess.DEVNULL, capture_output=True, text=True, errors="replace", timeout=600)
        dt = time.time() - t
        try:
            got = {e["unit"]: e for e in json.loads(out.read_text())["units"]}
            missing = [u for u in units if u not in got]
            if missing:
                raise ValueError(f"missing units {missing}")
        except Exception as e:  # noqa: BLE001 — bad JSON, missing file, missing units: retry once
            print(f"  {layer} {tag} attempt {attempt + 1} failed ({e}); rc={r.returncode} {r.stderr[-300:]}", flush=True)
            continue
        entries = {str(u): {k: got[u][k] for k in ("label", "description", "confidence", "polysemantic")} for u in units}
        raw.write_text(json.dumps({"units": units, "model": MODEL, "effort": effort, "seconds": round(dt, 1),
                                   "labels": entries}, indent=1))
        out.unlink(missing_ok=True)
        return tag, dt
    return tag, None


def collect(layer):
    labels = {}
    for f in sorted((cache_dir() / layer).glob("*.json")):
        labels.update(json.loads(f.read_text())["labels"])
    labels = dict(sorted(labels.items(), key=lambda kv: int(kv[0])))
    (assets_dir() / layer).mkdir(parents=True, exist_ok=True)
    write_json(assets_dir() / layer / "labels.json", labels, indent=1)
    return labels


ARGS_MODEL = "inceptionv1"


def assets_dir():
    return ROOT / "site" / "assets" / ARGS_MODEL


def cache_dir():
    # InceptionV1 keeps its original flat cache layout
    return CACHE if ARGS_MODEL == "inceptionv1" else CACHE / ARGS_MODEL


def main():
    global ARGS_MODEL
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="inceptionv1")
    p.add_argument("--layers", help="comma list; default all layers in manifest order")
    p.add_argument("--units", help="e.g. 0-23,50 (applies to every chosen layer)")
    p.add_argument("--batch", type=int, default=8, help="units per sheet")
    p.add_argument("--jobs", type=int, default=4, help="parallel codex processes")
    p.add_argument("--effort", default="medium", help="model_reasoning_effort")
    p.add_argument("--sheets-only", action="store_true",
                   help="only render the contact sheets (e.g. on the box that holds the assets) and exit")
    args = p.parse_args()

    ARGS_MODEL = args.model
    manifest = json.loads((ROOT / "site" / "manifest.json").read_text())
    entry = next(m for m in manifest["models"] if m["name"] == args.model)
    layers = {l["name"]: l["channels"] for l in entry["layers"]}
    chosen = args.layers.split(",") if args.layers else list(layers)
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "schema.json").write_text(json.dumps(SCHEMA))

    jobs = []
    for layer in chosen:
        (cache_dir() / layer).mkdir(parents=True, exist_ok=True)
        done = set(collect(layer))
        meta = json.loads((assets_dir() / layer / "units.json").read_text())
        todo = [u for u in parse_units(args.units, layers[layer]) if str(u) not in done]
        jobs += [(layer, todo[i:i + args.batch], meta) for i in range(0, len(todo), args.batch)]
    if args.sheets_only:
        for layer, units, _ in jobs:
            render_sheet(layer, units, cache_dir() / layer / f"{units[0]:04d}-{units[-1]:04d}.jpg")
        print(f"rendered {len(jobs)} sheets")
        return
    print(f"{len(jobs)} sheets to label", flush=True)

    t0, times, failed = time.time(), [], []
    with ThreadPoolExecutor(args.jobs) as pool:
        futs = {pool.submit(label_batch, l, u, m, args.effort): l for l, u, m in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            layer = futs[fut]
            tag, dt = fut.result()
            if dt is None:
                failed.append(f"{layer}/{tag}")
            else:
                times.append(dt)
            el = time.time() - t0
            print(f"[{i}/{len(jobs)}] {layer} {tag} {'FAILED' if dt is None else f'{dt:.0f}s'}"
                  f"  elapsed {el / 60:.1f}m  eta {el / i * (len(jobs) - i) / 60:.0f}m", flush=True)
    for layer in chosen:
        print(layer, len(collect(layer)), "labels")
    if times:
        print(f"mean {sum(times) / len(times):.1f}s/call over {len(times)} calls")
    if failed:
        print("failed:", " ".join(failed))


if __name__ == "__main__":
    main()
