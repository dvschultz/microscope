"""Precompute Microscope assets for one layer of a model (default InceptionV1).

    python precompute.py --layer mixed4a
    python precompute.py --model vgg19 --layer conv4_2 --batch 32

Writes, under site/assets/<model>/<layer>/:
    vis/<c>.png              channel-objective feature visualization (224px)
    examples/<c>/<rank>.jpg  top dataset-example crops from ImageNet val
    units.json               per-channel example metadata (image, activation, label, box)
and refreshes site/manifest.json. Resumable: channels whose files already exist
are skipped. The first run also does one pass over ImageNet val that caches the
top-k for every layer of the model (cache/[<model>/]topk_<layer>.npz), so later
layers skip it. Models are defined in microscope/models.py.
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from microscope.files import locked_update, save_npz, write_atomic

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "site"
CACHE = ROOT / "cache"
DATA = ROOT / "data" / "imagenet-val-256"
TIMINGS = CACHE / "timings.json"


def parse_channels(spec, n):
    if not spec:
        return list(range(n))
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return [c for c in out if c < n]


def layer_dir(model_name, layer):
    return SITE / "assets" / model_name / layer


def record_timing(layer, spec=None, **kv):
    key = layer if spec is None or spec.name == "inceptionv1" else f"{spec.name}/{layer}"
    locked_update(TIMINGS, lambda t: t.setdefault(key, {}).update(kv), indent=2)


def load_topk(layer, cache=CACHE):
    f = np.load(cache / f"topk_{layer}.npz")
    return {k: f[k] for k in f.files}


def do_examples(spec, model, device, layer, channels, n_show, recrop, pass_batch, max_px):
    from microscope.examples import ImageNetVal, crop_size, save_crops, topk_pass

    dataset = ImageNetVal(DATA)
    cache = spec.asset_cache(CACHE)
    missing = [l for l in spec.layers if not (cache / f"topk_{l}.npz").exists()]
    if missing:
        print(f"dataset pass over {dataset.n} images for {len(missing)} layers")
        t = time.time()
        result = topk_pass(model, device, dataset, layers=missing, batch=pass_batch, spec=spec)
        cache.mkdir(parents=True, exist_ok=True)
        for l, r in result.items():
            save_npz(cache / f"topk_{l}.npz", **r)
        record_timing("_dataset_pass", spec, seconds=round(time.time() - t, 1), layers=missing, images=dataset.n)

    out = layer_dir(spec.name, layer)
    units_path = out / "units.json"
    units = json.loads(units_path.read_text()) if units_path.exists() else {}
    if recrop:
        for c in channels:
            shutil.rmtree(out / "examples" / str(c), ignore_errors=True)
            units.pop(str(c), None)
    todo = [c for c in channels
            if str(c) not in units
            or not all((out / "examples" / str(c) / f"{r}.jpg").exists() for r in range(n_show))]
    if not todo:
        print("dataset examples: all present")
        return
    size = crop_size(spec.receptive_field[layer])
    print(f"dataset examples: cropping {len(todo)} channels at {size}px")
    t = time.time()
    meta = save_crops(dataset, load_topk(layer, cache), out / "examples", size, n_show, todo, max_px=max_px)
    units.update({str(c): rows for c, rows in meta.items()})
    write_atomic(units_path, json.dumps(dict(sorted(units.items(), key=lambda kv: int(kv[0])))))
    record_timing(layer, spec, crops_seconds=round(time.time() - t, 1), crop_size=size)


def gray_units(out, channels, thresh=0.02):
    from PIL import Image
    return [c for c in channels if (out / f"{c}.png").exists()
            and (np.asarray(Image.open(out / f"{c}.png"), dtype=np.float32) / 255).std() < thresh]


def do_fix_gray(spec, model, layer, channels, steps, batch):
    """Re-render flat-gray vis (unit dead on the random init) with the pre-ReLU objective."""
    from PIL import Image
    from microscope.featurevis import render_channels

    if spec.pre_relu is None:
        raise SystemExit(f"--fix-gray: no pre-ReLU modules defined for {spec.name}")
    out = layer_dir(spec.name, layer) / "vis"
    todo = gray_units(out, channels)
    print(f"fix gray: {len(todo)} flat renders")
    record = out.parent / "pre_relu.json"      # units whose vis used the pre-ReLU objective
    done = set(json.loads(record.read_text())) if record.exists() else set()
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        imgs = render_channels(model, layer, chunk, steps=steps, seed=chunk[0], spec=spec, pre_relu=True)
        for c, img in zip(chunk, imgs):
            Image.fromarray(img).save(out / f"{c}.png", optimize=True)
        done |= set(chunk)
        write_atomic(record, json.dumps(sorted(done)))
    left = gray_units(out, todo)
    record_timing(layer, spec, gray_fixed=len(todo) - len(left), gray_left=len(left))
    print(f"fix gray: {len(todo) - len(left)} fixed, {len(left)} still flat")


def do_vis(spec, model, layer, channels, steps, batch):
    from PIL import Image
    from tqdm import tqdm
    from microscope.featurevis import render_channels

    out = layer_dir(spec.name, layer) / "vis"
    out.mkdir(parents=True, exist_ok=True)
    todo = [c for c in channels if not (out / f"{c}.png").exists()]
    if not todo:
        print("feature vis: all present")
        return
    print(f"feature vis: {len(todo)} channels, {steps} steps, batch {batch}")
    t = time.time()
    for i in tqdm(range(0, len(todo), batch), unit="batch"):
        chunk = todo[i:i + batch]
        imgs = render_channels(model, layer, chunk, steps=steps, seed=chunk[0], spec=spec)
        for c, img in zip(chunk, imgs):
            Image.fromarray(img).save(out / f"{c}.png", optimize=True)
    dt = time.time() - t
    record_timing(layer, spec, vis_seconds=round(dt, 1), vis_channels=len(todo), steps=steps, batch=batch,
                  seconds_per_channel=round(dt / len(todo), 2), device=str(next(model.parameters()).device))
    print(f"feature vis: {dt / 60:.1f} min, {dt / len(todo):.2f} s/channel")


def model_entry(spec, shapes, n_show, model):
    from microscope.examples import crop_size

    layers = []
    for name in spec.layers:
        C, H, W = shapes[name]
        d = layer_dir(spec.name, name)
        vis = sum(1 for _ in (d / "vis").glob("*.png")) if (d / "vis").exists() else 0
        units = json.loads((d / "units.json").read_text()) if (d / "units.json").exists() else {}
        rf = spec.receptive_field[name]
        entry = dict(name=name, channels=C, spatial=[H, W], receptive_field=rf,
                     crop_size=crop_size(rf), rendered=vis, examples=len(units),
                     circuits=(d / "circuits.json").exists(),
                     circuits_ablation=(d / "circuits_ablation.json").exists(),
                     branches=spec.branches(model, name) if spec.branches else None)
        if spec.groups:
            entry["group"] = spec.groups[name]
            if name in spec.notes:
                entry["note"] = spec.notes[name]
            entry["text"] = (d / "text.json").exists()
            entry["example_px"] = min(crop_size(rf), spec.max_crop_px or 10 ** 6)
            pr = d / "pre_relu.json"
            entry["pre_relu_units"] = json.loads(pr.read_text()) if pr.exists() else []
            entry["labels"] = (d / "labels.json").exists()
        layers.append(entry)
    m = spec.name
    entry = dict(
        name=m, title=spec.title, description=spec.description, dataset=spec.dataset,
        examples_per_unit=n_show, layers=layers,
        paths=dict(vis=f"/assets/{m}/{{layer}}/vis/{{unit}}.png",
                   example=f"/assets/{m}/{{layer}}/examples/{{unit}}/{{rank}}.jpg",
                   units=f"/assets/{m}/{{layer}}/units.json",
                   circuits=f"/assets/{m}/{{layer}}/circuits.json"),
    )
    if spec.max_crop_px:
        entry["example_max_px"] = spec.max_crop_px
        entry["example_note"] = (f"Crops are cut at crop_size px from the 256px image (box in units.json) and "
                                 f"saved at most {spec.max_crop_px}px wide (example_px per layer).")
    if spec.pre_relu:
        entry["pre_relu_note"] = ("Units in a layer's pre_relu_units were zero on the random starting image, so the "
                                  "post-ReLU channel objective had no gradient; their feature vis maximizes the same "
                                  "channel before its final ReLU.")
    if spec.architecture:
        entry["architecture"] = spec.architecture
        entry["paths"].update(labels=f"/assets/{m}/{{layer}}/labels.json",
                              text=f"/assets/{m}/{{layer}}/text.json")
        entry["circuits_note"] = CIRCUITS_NOTE.get(m)
    return entry


CIRCUITS_NOTE = {
    "vgg19": "Circuits connect adjacent conv layers (a 2x2 maxpool sits between blocks).",
    "clip_rn50": "Circuits connect the chosen layers, not adjacent blocks: blocks between them are skipped over, "
                 "and residual connections carry part of each input straight through, so an entry measures the "
                 "total first-order effect of an earlier-layer unit through everything in between.",
}


def write_manifest(n_show, current=None):
    """Rewrite site/manifest.json with every registered model that has assets.
    current: (spec, model, shapes) already loaded, reused instead of reloading."""
    from microscope.models import SPECS

    models = []
    for spec in SPECS.values():
        if spec.name != "inceptionv1" and not (SITE / "assets" / spec.name).exists():
            continue
        if current and current[0].name == spec.name:
            _, model, shapes = current
        else:
            device = torch.device("cpu")
            model = spec.load(device, pretrained=False)   # shapes don't need weights
            shapes = layer_shapes(spec, model, device)
        models.append(model_entry(spec, shapes, n_show, model))
    write_atomic(SITE / "manifest.json", json.dumps(dict(models=models), indent=1))


@torch.no_grad()
def layer_shapes(spec, model, device, size=224):
    from microscope.model import LayerCapture
    cap = LayerCapture(model, spec.layers, spec.layers, spec.modules)
    out = cap.run(model, spec.preprocess(torch.zeros(1, 3, size, size, device=device)))
    cap.close()
    return {name: tuple(out[name].shape[1:]) for name in spec.layers}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="inceptionv1", help="inceptionv1, vgg19 or clip_rn50 (microscope/models.py)")
    p.add_argument("--layer", required=True, help="e.g. mixed4a, or 'manifest' to only rewrite the manifest")
    p.add_argument("--channels", help="subset, e.g. 0-31,40 (default: all)")
    p.add_argument("--steps", type=int, default=512)
    p.add_argument("--batch", type=int, default=32, help="channels optimized together")
    p.add_argument("--examples", type=int, default=16, help="dataset examples saved per channel")
    p.add_argument("--device", help="mps, cpu, or cuda (default: best available)")
    p.add_argument("--skip-vis", action="store_true")
    p.add_argument("--skip-examples", action="store_true")
    p.add_argument("--fix-gray", action="store_true", help="only re-render flat-gray vis with the pre-ReLU objective")
    p.add_argument("--recrop", action="store_true", help="redo example crops (e.g. after changing crop size)")
    p.add_argument("--pass-batch", type=int, default=200, help="images per batch in the dataset top-k pass")
    p.add_argument("--max-crop-px", type=int, help="downscale saved crops above this size "
                   "(default: the model's max_crop_px, 128 for vgg19/clip_rn50; 0 = off)")
    args = p.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    from microscope.device import patch_lucent, pick_device
    from microscope.models import get_spec

    spec = get_spec(args.model)
    if args.layer == "manifest":
        write_manifest(args.examples)
        print(f"manifest: {SITE / 'manifest.json'}")
        return
    device = pick_device(args.device)
    patch_lucent(device)
    model = spec.load(device)
    for prm in model.parameters():
        prm.requires_grad_(False)
    shapes = layer_shapes(spec, model, device)
    print(f"device: {device}")

    if args.layer not in spec.layers:
        p.error(f"--layer must be one of {', '.join(spec.layers)}")
    max_px = args.max_crop_px if args.max_crop_px is not None else spec.max_crop_px
    channels = parse_channels(args.channels, shapes[args.layer][0])
    if not args.skip_examples:
        do_examples(spec, model, device, args.layer, channels, args.examples, args.recrop, args.pass_batch, max_px)
    if args.fix_gray:
        do_fix_gray(spec, model, args.layer, channels, args.steps, args.batch)
    elif not args.skip_vis:
        do_vis(spec, model, args.layer, channels, args.steps, args.batch)
    write_manifest(args.examples, current=(spec, model, shapes))
    print(f"manifest: {SITE / 'manifest.json'}")


if __name__ == "__main__":
    main()
