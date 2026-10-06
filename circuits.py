"""Circuits: which units in the previous layer drive each unit, measured on real images.

    python circuits.py                      # all layers (needs cache/topk_*.npz)
    python circuits.py --layers mixed4a mixed4b
    python circuits.py --model clip_rn50          # any model in microscope/models.py

P is the previous layer in the model's layer list. For residual nets (CLIP RN50) the
chosen layers are not adjacent blocks, so this measures P's total first-order effect
through everything in between, skip connections included.

For unit u in layer L with previous layer P, take u's top-n dataset images and the
spatial position where u peaked in each. Attribution of P-channel c is
    sum over space of  act_P[c] * d act_L[u, y, x] / d act_P[c]
averaged over the n images and divided by u's mean peak activation, so values
read as "fraction of this unit's activation contributed by c" (first-order;
fractions don't sum to exactly 1 because of biases and nonlinearity).

Writes cache/[<model>/]attr_<L>.npy ([C_L, C_P] float32) and, per layer,
site/assets/<model>/<layer>/circuits.json:
    {u: {"in":  {"pos": [[c, frac], ...], "neg": [[c, frac], ...]},   # from P
         "out": {"pos": [[v, frac], ...], "neg": [[v, frac], ...]}}}  # into next layer
Units in P/next layer are referred to by channel index; the manifest names the layers.
For models other than InceptionV1 it also writes site/assets/<model>/circuits_summary.json
(layer pairs, n, and a note on what the numbers mean for that architecture).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from microscope.model import StopForward
from microscope.models import get_spec
from microscope.files import write_json

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
SITE = ROOT / "site"
DATA = ROOT / "data" / "imagenet-val-256"


class PairCapture:
    """Make P's output a leaf that requires grad, record L's output, stop after L."""

    def __init__(self, model, prev, layer, spec):
        self.out = {}
        self.layer = layer
        self.handles = [
            spec.module(model, prev).register_forward_hook(self._prev),
            spec.module(model, layer).register_forward_hook(self._layer),
        ]
        self.prev = prev

    def _prev(self, module, inputs, output):
        leaf = output.detach().requires_grad_(True)
        self.out[self.prev] = leaf
        return leaf

    def _layer(self, module, inputs, output):
        self.out[self.layer] = output
        raise StopForward

    def run(self, model, x):
        self.out = {}
        try:
            model(x)
        except StopForward:
            pass
        return self.out

    def close(self):
        for h in self.handles:
            h.remove()


def attribute_layer(spec, model, device, images, layer, prev, n, units_per_batch):
    top = dict(np.load(spec.asset_cache(CACHE) / f"topk_{layer}.npz"))
    C = top["act"].shape[1]
    cap = PairCapture(model, prev, layer, spec)
    rows = []
    for start in range(0, C, units_per_batch):
        units = list(range(start, min(C, start + units_per_batch)))
        idx = np.array([[top["img"][r, u] for r in range(n)] for u in units]).reshape(-1)
        ys = torch.tensor(np.array([[top["y"][r, u] for r in range(n)] for u in units]).reshape(-1), device=device)
        xs = torch.tensor(np.array([[top["x"][r, u] for r in range(n)] for u in units]).reshape(-1), device=device)
        us = torch.tensor(np.repeat(units, n), device=device)
        x = torch.from_numpy(images[idx]).to(device).permute(0, 3, 1, 2).float() / 255
        out = cap.run(model, spec.preprocess(x))
        a_prev, a_layer = out[prev], out[layer]
        b = torch.arange(len(idx), device=device)
        target = a_layer[b, us, ys, xs]                                  # [B]
        target.sum().backward()
        contrib = (a_prev.grad * a_prev.detach()).sum(dim=(2, 3))       # [B, C_P]
        contrib = contrib.reshape(len(units), n, -1).mean(1)
        peak = target.detach().reshape(len(units), n).mean(1).clamp_min(1e-6)
        rows.append((contrib / peak[:, None]).cpu().numpy())
    cap.close()
    return np.concatenate(rows).astype(np.float32)


def top_pairs(vec, k, sign):
    order = np.argsort(-vec if sign > 0 else vec)[:k]
    return [[int(c), round(float(vec[c]), 4)] for c in order if sign * vec[c] > 0]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="inceptionv1")
    p.add_argument("--layers", nargs="*", help="layers to attribute (default: all but the first)")
    p.add_argument("--n", type=int, default=16, help="top dataset images per unit")
    p.add_argument("--units-per-batch", type=int, default=8)
    p.add_argument("--k", type=int, default=8, help="inputs/outputs kept per direction")
    p.add_argument("--device")
    args = p.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    from microscope.device import patch_lucent, pick_device
    from microscope.examples import ImageNetVal

    spec = get_spec(args.model)
    LAYERS, cache = spec.layers, spec.asset_cache(CACHE)
    device = pick_device(args.device)
    patch_lucent(device)
    model = spec.load(device)
    for prm in model.parameters():
        prm.requires_grad_(False)

    todo = [l for l in (args.layers or LAYERS[1:]) if l != LAYERS[0] and not (cache / f"attr_{l}.npy").exists()]
    if todo:
        wanted = set()
        for l in todo:
            wanted |= set(np.load(cache / f"topk_{l}.npz")["img"][:args.n].reshape(-1).tolist())
        print(f"loading {len(wanted)} images")
        images = ImageNetVal(DATA).load(wanted)
        for l in todo:
            t = time.time()
            prev = LAYERS[LAYERS.index(l) - 1]
            attr = attribute_layer(spec, model, device, images, l, prev, args.n, args.units_per_batch)
            np.save(cache / f"attr_{l}.npy", attr)
            print(f"{prev} -> {l}: {attr.shape}, {time.time() - t:.0f}s", flush=True)

    # circuits.json for every layer that has attr data on either side
    for i, l in enumerate(LAYERS):
        f_in = cache / f"attr_{l}.npy"
        f_out = cache / f"attr_{LAYERS[i + 1]}.npy" if i + 1 < len(LAYERS) else None
        a_in = np.load(f_in) if f_in.exists() else None
        a_out = np.load(f_out) if f_out is not None and f_out.exists() else None
        if a_in is None and a_out is None:
            continue
        C = a_in.shape[0] if a_in is not None else a_out.shape[1]
        res = {}
        for u in range(C):
            r = {}
            if a_in is not None:
                r["in"] = dict(pos=top_pairs(a_in[u], args.k, 1), neg=top_pairs(a_in[u], args.k, -1))
            if a_out is not None:
                r["out"] = dict(pos=top_pairs(a_out[:, u], args.k, 1), neg=top_pairs(a_out[:, u], args.k, -1))
            res[str(u)] = r
        d = SITE / "assets" / spec.name / l
        d.mkdir(parents=True, exist_ok=True)
        write_json(d / "circuits.json", res, separators=(",", ":"))
        print(f"wrote {d / 'circuits.json'}")

    if spec.name != "inceptionv1":
        from precompute import CIRCUITS_NOTE
        pairs = [[LAYERS[i - 1], l] for i, l in enumerate(LAYERS) if i and (cache / f"attr_{l}.npy").exists()]
        summary = dict(model=spec.name, pairs=pairs, n_images=args.n, k=args.k,
                       method="activation x gradient at the unit's peak, mean over its top-n dataset images, "
                              "divided by the unit's mean peak activation",
                       note=CIRCUITS_NOTE.get(spec.name))
        write_json(SITE / "assets" / spec.name / "circuits_summary.json", summary, indent=1)


if __name__ == "__main__":
    main()
