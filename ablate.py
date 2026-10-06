"""Ablation check of the circuits: does knocking out an input channel move the unit?

    python ablate.py                         # all layers (needs cache/topk_*.npz, cache/attr_*.npy)
    python ablate.py --layers mixed4a        # one layer
    python ablate.py --verify                # batched vs naive full-model ablation on a few units
    python ablate.py --curves                # mixed4a curve units: ablate every mixed3b channel
    python ablate.py --json-only             # rebuild JSON + summary from cache/ablate_*.npz

circuits.py estimates, to first order (act x grad), how much each channel c of the
previous layer P contributes to unit u of layer L. Here we measure it: for each of
u's top-n dataset images, set channel c of P to zero everywhere (P is post-ReLU, so
zero is "off" and is the baseline act x grad linearizes around), recompute P -> L,
and read u at its peak position. effect = mean over images of
(act_clean - act_ablated) / act_clean; positive means c excites u.

Candidates per unit: the top-12 and bottom-8 channels of attr_<L>[u], plus 4 random
channels as a control. P's activations are computed once per unit group; only the
P -> L sub-network is re-run, with (candidates x images) stacked in the batch.

Writes cache/ablate_<L>.npz:
    units [U], cand [U,K], kind [U,K] (0 grad-pos, 1 grad-neg, 2 random),
    grad [U,K] (attr value), clean [U,n], ablated [U,K,n] (u's act at its peak),
    effect [U,K], runtime (s), clean_err (max rel. diff of clean act vs topk act)
site/assets/inceptionv1/<L>/circuits_ablation.json (same schema as circuits.json,
values are ablation effects) and site/assets/inceptionv1/circuits_ablation_summary.json.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from microscope.model import LAYERS, MODEL, LayerCapture, StopForward, preprocess
from microscope.files import write_json

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
SITE = ROOT / "site"
DATA = ROOT / "data" / "imagenet-val-256"

N_POS, N_NEG, N_RAND = 12, 8, 4
CURVE_4A = [26, 50, 58, 62, 95, 168, 403, 406, 407, 415, 424, 428, 436, 437]
from microscope.model import CURVE_DETECTORS_3B as CURVE_3B
VERIFY_LAYERS = ["conv2d1", "conv2d2", "mixed3a", "mixed4a", "mixed5a", "mixed5b"]


# ---------------------------------------------------------------- P -> L sub-network
# Mirrors lucent's InceptionV1.forward between consecutive Microscope layers.

def lrn(x):
    # TF semantics (alpha not divided by size), as microscope.model._fix_lucent_lrn
    return F.local_response_norm(x, size=9, alpha=9.99999974738e-05 * 9, beta=0.5, k=1)


def maxpool_s2(m, name, x):
    return getattr(m, name)(F.pad(x, (0, 1, 0, 1), value=float("-inf")),
                            kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)


def mixed(m, name, x):
    g = lambda s: getattr(m, f"{name}_{s}")
    b1 = g("1x1")(g("1x1_pre_relu_conv")(x))
    b3 = g("3x3_bottleneck")(g("3x3_bottleneck_pre_relu_conv")(x))
    b3 = g("3x3")(g("3x3_pre_relu_conv")(F.pad(b3, (1, 1, 1, 1))))
    b5 = g("5x5_bottleneck")(g("5x5_bottleneck_pre_relu_conv")(x))
    b5 = g("5x5")(g("5x5_pre_relu_conv")(F.pad(b5, (2, 2, 2, 2))))
    bp = g("pool")(F.pad(x, (1, 1, 1, 1), value=float("-inf")),
                   kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
    bp = g("pool_reduce")(g("pool_reduce_pre_relu_conv")(bp))
    return getattr(m, name)((b1, b3, b5, bp), 1)


def forward_from(m, layer, a):
    """Output of `layer` given the output of the layer before it."""
    if layer == "conv2d1":
        return m.conv2d1(m.conv2d1_pre_relu_conv(lrn(maxpool_s2(m, "maxpool0", a))))
    if layer == "conv2d2":
        return m.conv2d2(m.conv2d2_pre_relu_conv(F.pad(a, (1, 1, 1, 1))))
    if layer == "mixed3a":
        a = maxpool_s2(m, "maxpool1", lrn(a))
    elif layer == "mixed4a":
        a = maxpool_s2(m, "maxpool4", a)
    elif layer == "mixed5a":
        a = maxpool_s2(m, "maxpool10", a)
    return mixed(m, layer, a)


# ---------------------------------------------------------------- ablation

@torch.no_grad()
def ablate_group(model, a_prev, layer, units, ys, xs, chans, max_items):
    """a_prev [G*n, C, H, W]: P activations of each unit's n images (unit-major).
    chans: list of G channel lists. Returns clean [G, n] and a list of [K_g, n] arrays."""
    G = len(units)
    n = a_prev.shape[0] // G
    A = a_prev.view(G, n, *a_prev.shape[1:])
    jobs = [(g, -1) for g in range(G)] + [(g, c) for g in range(G) for c in chans[g]]
    vals = []
    per_chunk = max(1, max_items // n)
    for s in range(0, len(jobs), per_chunk):
        jg = torch.tensor([j[0] for j in jobs[s:s + per_chunk]], device=a_prev.device)
        jc = torch.tensor([j[1] for j in jobs[s:s + per_chunk]], device=a_prev.device)
        batch = A[jg].clone()                                            # [J, n, C, H, W]
        hit = (jc >= 0).nonzero().squeeze(1)
        batch[hit, :, jc[hit]] = 0
        out = forward_from(model, layer, batch.flatten(0, 1))            # [J*n, C_L, H_L, W_L]
        out = out.view(len(jg), n, *out.shape[1:])
        b = torch.arange(n, device=out.device)
        u = units[jg][:, None].expand(-1, n)
        vals.append(out[torch.arange(len(jg), device=out.device)[:, None], b[None], u,
                        ys[jg], xs[jg]])                                 # [J, n]
    vals = torch.cat(vals).cpu().numpy().astype(np.float32)
    clean, rest = vals[:G], vals[G:]
    abl, i = [], 0
    for g in range(G):
        abl.append(rest[i:i + len(chans[g])])
        i += len(chans[g])
    return clean, abl


def effects(clean, abl):
    """clean [..., n], abl [..., K, n] -> mean over images of (clean - abl) / clean."""
    c = clean[..., None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(c > 1e-6, (c - abl) / c, np.nan)
        e = np.nanmean(r, axis=-1)
    return np.nan_to_num(e).astype(np.float32)


def candidates(attr_row, layer_i, u):
    order = np.argsort(-attr_row)
    pos, neg = order[:N_POS], order[::-1][:N_NEG]
    rest = np.setdiff1d(np.arange(len(attr_row)), np.concatenate([pos, neg]))
    rnd = np.random.default_rng(layer_i * 100003 + u).choice(rest, N_RAND, replace=False)
    cand = np.concatenate([pos, neg, rnd])
    kind = np.repeat([0, 1, 2], [N_POS, N_NEG, N_RAND])
    return cand, kind


class Runner:
    def __init__(self, model, device, images, n, budget):
        self.model, self.device, self.images, self.n, self.budget = model, device, images, n, budget

    def prev_acts(self, prev, idx):
        cap = LayerCapture(self.model, [prev])
        x = torch.from_numpy(self.images[idx]).to(self.device).permute(0, 3, 1, 2).float() / 255
        with torch.no_grad():
            a = cap.run(self.model, preprocess(x))[prev]
        cap.close()
        return a

    def run(self, layer, units, chans):
        """Ablate chans[i] (channel list) of P for unit units[i]. Returns clean [U,n],
        list of [K_i, n] ablated acts, topk act [U, n]."""
        prev = LAYERS[LAYERS.index(layer) - 1]
        top = dict(np.load(CACHE / f"topk_{layer}.npz"))
        n = self.n
        shape = None
        clean_all, abl_all = [], []
        # group size: as many units as fit one P-activation tensor of ~budget/8 items
        probe = self.prev_acts(prev, top["img"][:1, 0])
        item = probe[0].numel()
        max_items = max(n, int(self.budget // (item * 4 * 6)))
        G = max(1, min(64, max_items // (n * (1 + max(len(c) for c in chans)))))
        for s in range(0, len(units), G):
            us = np.asarray(units[s:s + G])
            idx = top["img"][:n, us].T.reshape(-1)
            a = self.prev_acts(prev, idx)
            t = lambda v: torch.from_numpy(np.ascontiguousarray(v[:n, us].T)).to(self.device)
            c, ab = ablate_group(self.model, a, layer, torch.from_numpy(us).to(self.device),
                                 t(top["y"]), t(top["x"]), chans[s:s + G], max_items)
            clean_all.append(c)
            abl_all += ab
        return np.concatenate(clean_all), abl_all, top["act"][:n, units].T


def ablate_layer(runner, layer):
    t0 = time.time()
    attr = np.load(CACHE / f"attr_{layer}.npy")
    li = LAYERS.index(layer)
    units = np.arange(attr.shape[0])
    ck = [candidates(attr[u], li, u) for u in units]
    cand = np.stack([c for c, _ in ck])
    kind = np.stack([k for _, k in ck])
    clean, abl, topact = runner.run(layer, units, list(cand))
    abl = np.stack(abl)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    runtime = time.time() - t0
    err = float(np.max(np.abs(clean - topact) / np.maximum(np.abs(topact), 1e-3)))
    np.savez(CACHE / f"ablate_{layer}.npz", units=units, cand=cand, kind=kind,
             grad=np.take_along_axis(attr, cand, 1).astype(np.float32),
             clean=clean, ablated=abl, effect=effects(clean, abl),
             runtime=runtime, clean_err=err)
    return runtime, err


# ---------------------------------------------------------------- checks

def verify(runner, model, device, layers, n_units=3, n_cands=4):
    """Naive path: full model, forward hook on P zeroing one channel, one image set at a time."""
    worst = 0.0
    for layer in layers:
        prev = LAYERS[LAYERS.index(layer) - 1]
        attr = np.load(CACHE / f"attr_{layer}.npy")
        top = dict(np.load(CACHE / f"topk_{layer}.npz"))
        rng = np.random.default_rng(1)
        units = rng.choice(attr.shape[0], n_units, replace=False)
        chans = [candidates(attr[u], LAYERS.index(layer), u)[0][[0, 1, N_POS, N_POS + N_NEG]][:n_cands]
                 for u in units]
        clean_b, abl_b, _ = runner.run(layer, units, chans)
        for i, u in enumerate(units):
            idx = top["img"][:runner.n, u]
            x = torch.from_numpy(runner.images[idx]).to(device).permute(0, 3, 1, 2).float() / 255
            ys, xs = top["y"][:runner.n, u], top["x"][:runner.n, u]
            for j, c in enumerate([-1] + list(chans[i])):
                def zero(mod, inp, out, c=c):
                    if c >= 0:
                        out = out.clone()
                        out[:, c] = 0
                    return out
                h = getattr(model, prev).register_forward_hook(zero)
                cap = LayerCapture(model, [layer])
                with torch.no_grad():
                    o = cap.run(model, preprocess(x))[layer]
                cap.close()
                h.remove()
                naive = o[torch.arange(len(idx)), u, torch.as_tensor(ys), torch.as_tensor(xs)].cpu().numpy()
                batched = clean_b[i] if c < 0 else abl_b[i][j - 1]
                d = float(np.max(np.abs(naive - batched)) / max(1e-3, float(np.abs(naive).max())))
                worst = max(worst, d)
                print(f"verify {layer}:{u} c={c:4d} naive={naive[:3].round(4)} batched={batched[:3].round(4)} rel={d:.2e}")
    print(f"verify worst rel diff {worst:.2e}")
    return worst


def curves(runner, n_random=50):
    """mixed4a: ablate every mixed3b channel for the 14 curve units + random units."""
    attr = np.load(CACHE / "attr_mixed4a.npy")
    others = np.setdiff1d(np.arange(attr.shape[0]), CURVE_4A)
    rnd = np.random.default_rng(0).choice(others, n_random, replace=False)
    units = np.concatenate([CURVE_4A, rnd])
    allc = np.arange(attr.shape[1])
    t0 = time.time()
    clean, abl, _ = runner.run("mixed4a", units, [allc] * len(units))
    eff = effects(clean, np.stack(abl))
    np.savez(CACHE / "ablate_curves_mixed4a.npz", units=units, effect=eff,
             grad=attr[units].astype(np.float32), n_curve=len(CURVE_4A), runtime=time.time() - t0)
    return curve_report()


def curve_report():
    d = np.load(CACHE / "ablate_curves_mixed4a.npz")
    units, eff, grad, nc = d["units"], d["effect"], d["grad"], int(d["n_curve"])
    site = np.load(CACHE / "ablate_mixed4a.npz")
    top8 = lambda v: [int(c) for c in np.argsort(-v)[:8] if v[c] > 0]
    rows = []
    for i, u in enumerate(units[:nc]):
        u = int(u)
        cand_eff = np.full(eff.shape[1], -np.inf)
        cand_eff[site["cand"][u]] = site["effect"][u]
        rows.append(dict(
            unit=u,
            grad_top8=[c for c in top8(grad[i]) if c in CURVE_3B],
            ablation_top8_candidates=[c for c in top8(cand_eff) if c in CURVE_3B],
            ablation_top8_all=[c for c in top8(eff[i]) if c in CURVE_3B],
            curve_effects={c: round(float(eff[i, c]), 4) for c in CURVE_3B},
            curve_ranks={c: int((eff[i] > eff[i, c]).sum()) + 1 for c in CURVE_3B},
        ))
    frac = lambda arr, sl: float(np.mean([bool(set(top8(v)) & set(CURVE_3B)) for v in arr[sl]]))
    i50 = int(np.where(units == 50)[0][0])
    res = dict(
        units=rows,
        curve_units_with_curve_in_grad_top8=int(sum(bool(r["grad_top8"]) for r in rows)),
        curve_units_with_curve_in_ablation_top8_all=int(sum(bool(r["ablation_top8_all"]) for r in rows)),
        curve_units_with_curve_in_ablation_top8_candidates=int(sum(bool(r["ablation_top8_candidates"]) for r in rows)),
        random_units=len(units) - nc,
        random_frac_grad_top8=frac(grad, slice(nc, None)),
        random_frac_ablation_top8=frac(eff, slice(nc, None)),
        unit50={c: dict(effect=round(float(eff[i50, c]), 4), rank=int((eff[i50] > eff[i50, c]).sum()) + 1,
                        grad=round(float(grad[i50, c]), 4)) for c in (406, 385, 379)},
        # full-sweep top-8 ablation inputs that fall inside the 12+8 gradient candidates
        candidate_coverage=float(np.mean([
            np.mean([c in set(site["cand"][int(u)][:N_POS + N_NEG]) for c in top8(eff[i])] or [1])
            for i, u in enumerate(units)])),
        runtime=round(float(d["runtime"]), 1),
    )
    return res


# ---------------------------------------------------------------- outputs

def rank(v):
    r = np.empty(len(v))
    r[np.argsort(v, kind="stable")] = np.arange(len(v))
    return r


def spearman(a, b):
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def pairs(chans, vals, k, sign):
    order = np.argsort(-vals if sign > 0 else vals)[:k]
    return [[int(chans[i]), round(float(vals[i]), 4)] for i in order if sign * vals[i] > 0]


def write_outputs(k=8):
    data = {l: dict(np.load(CACHE / f"ablate_{l}.npz")) for l in LAYERS if (CACHE / f"ablate_{l}.npz").exists()}
    summary = {"method": "zero-ablation of one P channel (all positions) on the unit's top-16 images; "
                         "effect = mean_i (clean_i - ablated_i) / clean_i at the unit's peak",
               "layers": {}}
    for i, l in enumerate(LAYERS):
        a_in = data.get(l)
        a_out = data.get(LAYERS[i + 1]) if i + 1 < len(LAYERS) else None
        if a_in is None and a_out is None:
            continue
        C = len(a_in["units"]) if a_in is not None else int(np.load(CACHE / f"attr_{LAYERS[i + 1]}.npy").shape[1])
        outs = {u: ([], []) for u in range(C)}
        if a_out is not None:
            for v, (cs, es) in enumerate(zip(a_out["cand"], a_out["effect"])):
                for c, e in zip(cs, es):
                    outs[int(c)][0].append(v)
                    outs[int(c)][1].append(e)
        res = {}
        for u in range(C):
            r = {}
            if a_in is not None:
                cs, es = a_in["cand"][u], a_in["effect"][u]
                r["in"] = dict(pos=pairs(cs, es, k, 1), neg=pairs(cs, es, k, -1))
            if a_out is not None:
                vs, es = np.array(outs[u][0], dtype=int), np.array(outs[u][1], dtype=np.float32)
                r["out"] = dict(pos=pairs(vs, es, k, 1), neg=pairs(vs, es, k, -1))
            res[str(u)] = r
        d = SITE / "assets" / MODEL / l
        d.mkdir(parents=True, exist_ok=True)
        write_json(d / "circuits_ablation.json", res, separators=(",", ":"))

        if a_in is not None:
            g, e, kind = a_in["grad"], a_in["effect"], a_in["kind"]
            nr = kind < 2
            # gradient top-8 by sign, as in circuits.json (candidates hold top-12 / bottom-8)
            pos = (kind == 0) & (np.arange(g.shape[1]) < 8) & (g > 0)
            neg = (kind == 1) & (g < 0)
            conf_p = (e > 0.01) & pos
            conf_n = (e < -0.01) & neg
            summary["layers"][l] = dict(
                prev=LAYERS[i - 1], units=int(len(g)), candidates=int(g.size),
                spearman=round(spearman(g.ravel(), e.ravel()), 3),
                spearman_nonrandom=round(spearman(g[nr], e[nr]), 3),
                spearman_per_unit_median=round(float(np.nanmedian(
                    [spearman(g[u], e[u]) if np.std(e[u]) > 0 else np.nan for u in range(len(g))])), 3),
                confirmed=round(float((conf_p.sum() + conf_n.sum()) / max(1, pos.sum() + neg.sum())), 3),
                confirmed_pos=round(float(conf_p.sum() / max(1, pos.sum())), 3),
                confirmed_neg=round(float(conf_n.sum() / max(1, neg.sum())), 3),
                mean_abs_effect_grad_top8_pos=round(float(np.abs(e[pos]).mean()), 4),
                mean_abs_effect_random=round(float(np.abs(e[kind == 2]).mean()), 4),
                mean_abs_grad_random=round(float(np.abs(g[kind == 2]).mean()), 4),
                runtime_s=round(float(a_in["runtime"]), 1),
                clean_vs_topk_max_rel_err=float(a_in["clean_err"]),
            )
    if (CACHE / "ablate_curves_mixed4a.npz").exists():
        summary["curves_mixed4a"] = curve_report()
    write_json(SITE / "assets" / MODEL / "circuits_ablation_summary.json", summary, indent=1)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layers", nargs="*", default=LAYERS[1:])
    p.add_argument("--n", type=int, default=16, help="top dataset images per unit")
    p.add_argument("--budget-gb", type=float, default=6, help="rough GPU memory budget per batch")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--curves", action="store_true")
    p.add_argument("--json-only", action="store_true")
    p.add_argument("--force", action="store_true", help="recompute layers that already have ablate_<L>.npz")
    p.add_argument("--device")
    args = p.parse_args()

    if not args.json_only:
        import warnings
        warnings.filterwarnings("ignore")
        from microscope.device import patch_lucent, pick_device
        from microscope.examples import ImageNetVal
        from microscope.model import load_model

        # effects are thresholded at 1%: keep convs in full fp32
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        device = pick_device(args.device)
        patch_lucent(device)
        model = load_model(device)

        todo = [l for l in args.layers if l != "conv2d0" and (args.force or not (CACHE / f"ablate_{l}.npz").exists())]
        need = set(todo) | ({"mixed4a"} if args.curves else set()) | (set(VERIFY_LAYERS) if args.verify else set())
        wanted = set()
        for l in need:
            wanted |= set(np.load(CACHE / f"topk_{l}.npz")["img"][:args.n].reshape(-1).tolist())
        print(f"loading {len(wanted)} images", flush=True)
        images = ImageNetVal(DATA).load(wanted)
        runner = Runner(model, device, images, args.n, args.budget_gb * 1e9)

        if args.verify:
            verify(runner, model, device, VERIFY_LAYERS)
        for l in todo:
            rt, err = ablate_layer(runner, l)
            print(f"{LAYERS[LAYERS.index(l) - 1]} -> {l}: {rt:.0f}s, clean vs topk max rel err {err:.1e}", flush=True)
        if args.curves:
            print(json.dumps(curves(runner), indent=1), flush=True)

    s = write_outputs()
    for l, r in s["layers"].items():
        print(l, r)


if __name__ == "__main__":
    main()
