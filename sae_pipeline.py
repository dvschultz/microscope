"""Sparse-autoencoder features for InceptionV1, as a second "model" in the viewer.

    python sae_pipeline.py collect                       # sample activations (both layers, one pass)
    python sae_pipeline.py train --layer mixed4a --k 32  # TopK SAE -> cache/sae/mixed4a_k32.pt
    python sae_pipeline.py examples --layer mixed4a --k 32
    python sae_pipeline.py render --layer mixed4a --budget-min 80
    python sae_pipeline.py curves --layer mixed3b --k 32 # synthetic-curve tuning, SAE vs channels
    python sae_pipeline.py manifest                      # features.json rendered flags + site/manifest_sae.json

Later layers (disk budget: assets only for the top-N features by energy, crops stored at 128px):
    python sae_pipeline.py collect --layers mixed4b,mixed4c,mixed4d,mixed4e,mixed5a,mixed5b
    python sae_pipeline.py examples --layer mixed4b --k 32 --expose 2000 --example-px 128
    python sae_pipeline.py manifest --layers mixed4b --example-px 128   # other layers' entries kept as-is

Activations: post-ReLU concat output of the layer on ImageNet val at 256px (same
preprocess as everything else), POSITIONS random spatial positions per image.
Images >= HOLDOUT are held out for evaluation. Each layer gets one scale factor so
the mean L2 norm of a training vector is sqrt(d); it is stored with the weights.

Outputs, under site/assets/inceptionv1_sae/<layer>/ (never inceptionv1/):
    vis/<f>.png, examples/<f>/<rank>.jpg, units.json   same schema as inceptionv1/
    features.json  {f: {orig, density, energy, top_channels: [[c, w], ...], rendered}}
Features that never fire on the 50k images are dropped; the rest are re-indexed
by energy share (f=0 carries the most), `orig` is the SAE's own index.
"""
import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from microscope.files import locked_update, write_json

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "site"
CACHE = ROOT / "cache" / "sae"
DATA = ROOT / "data" / "imagenet-val-256"
SHM = Path(os.environ.get("SAE_SHM", "/dev/shm/microscope_sae"))   # RAM, not disk
MODEL_NAME = "inceptionv1_sae"
SAE_LAYERS = ["mixed3b", "mixed4a"]
POSITIONS = 32
HOLDOUT = 47_500
N_SHOW = 16
from microscope.model import CURVE_DETECTORS_3B as CURVE_CHANNELS


def setup(device=None):
    import warnings
    warnings.filterwarnings("ignore")
    from microscope.device import patch_lucent, pick_device
    from microscope.model import load_model
    device = pick_device(device)
    patch_lucent(device)
    return device, load_model(device)




def layer_out(layer):
    return SITE / "assets" / MODEL_NAME / layer


# ---------------------------------------------------------------- collect

@torch.no_grad()
def cmd_collect(args):
    from tqdm import tqdm
    from microscope.examples import ImageNetVal
    from microscope.model import LayerCapture, preprocess

    device, model = setup(args.device)
    ds = ImageNetVal(DATA)
    n_img = min(ds.n, args.images or ds.n)
    layers = args.layers.split(",") if args.layers else SAE_LAYERS
    cap = LayerCapture(model, layers)
    g = torch.Generator(device=device).manual_seed(0)
    out = {l: [] for l in layers}
    idx_all = []
    bar = tqdm(total=n_img, unit="img")
    for start, imgs, _ in ds.iter_groups():
        if start >= n_img:
            break
        imgs = imgs[: n_img - start]
        for i in range(0, len(imgs), args.batch):
            x = torch.from_numpy(imgs[i:i + args.batch]).to(device).permute(0, 3, 1, 2).float() / 255
            acts = cap.run(model, preprocess(x))
            B = x.shape[0]
            for l in layers:
                a = acts[l]
                _, C, H, W = a.shape
                pos = torch.rand(B, H * W, device=device, generator=g).argsort(1)[:, :POSITIONS]
                v = a.reshape(B, C, H * W).gather(2, pos[:, None].expand(B, C, POSITIONS))
                out[l].append(v.permute(0, 2, 1).reshape(-1, C).half().cpu())
            idx_all.append(np.repeat(np.arange(start + i, start + i + B), POSITIONS))
            bar.update(B)
    bar.close()
    cap.close()
    SHM.mkdir(parents=True, exist_ok=True)
    np.save(SHM / "img_idx.npy", np.concatenate(idx_all))
    for l in layers:
        X = torch.cat(out[l]).numpy()
        np.save(SHM / f"acts_{l}.npy", X)
        print(l, X.shape, f"{X.nbytes / 1e9:.2f} GB in {SHM}")


# ---------------------------------------------------------------- train

def load_acts(layer, device):
    X = torch.from_numpy(np.load(SHM / f"acts_{layer}.npy"))
    img = np.load(SHM / "img_idx.npy")
    hold = torch.from_numpy(img >= (HOLDOUT if img.max() >= HOLDOUT else np.quantile(img, 0.95)))
    return X[~hold].to(device), X[hold].to(device)


@torch.no_grad()
def evaluate(sae, X, chunk=65536):
    """FVE (vs. per-dim mean), mean L0, fire counts. X is scaled, fp16."""
    sse, sst, l0 = 0.0, 0.0, 0.0
    fired = torch.zeros(sae.n, device=X.device)
    mu = X.float().mean(0)
    for i in range(0, len(X), chunk):
        x = X[i:i + chunk].float()
        z = sae.encode(x)
        sse += (sae.decode(z) - x).pow(2).sum().item()
        sst += (x - mu).pow(2).sum().item()
        l0 += (z > 0).sum().item()
        fired += (z > 0).sum(0)
    return dict(fve=1 - sse / sst, l0=l0 / len(X)), fired


def cmd_train(args):
    from microscope.sae import TopKSAE

    device = torch.device(args.device or "cuda")
    torch.manual_seed(0)
    Xtr, Xte = load_acts(args.layer, device)
    d = Xtr.shape[1]
    scale = math.sqrt(d) / Xtr.float().norm(dim=1).mean().item()
    Xtr.mul_(scale)
    Xte.mul_(scale)
    n = args.n or 8 * d
    sae = TopKSAE(d, n, args.k).to(device)
    sae.scale.fill_(scale)
    with torch.no_grad():
        sae.b_dec.copy_(Xtr.float().mean(0))
    opt = torch.optim.Adam(sae.parameters(), lr=args.lr, betas=(0.9, 0.999))
    var = (Xtr.float() - Xtr.float().mean(0)).pow(2).sum(1).mean()
    k_aux = args.k_aux or d // 2
    since_fired = torch.zeros(n, device=device)          # samples since each feature last fired
    dead_after = args.dead_after
    log = []
    t0 = time.time()
    for step in range(args.steps):
        lr = args.lr * min(1.0, (step + 1) / 500, (args.steps - step) / (0.2 * args.steps))
        for gp in opt.param_groups:
            gp["lr"] = lr
        x = Xtr[torch.randint(len(Xtr), (args.batch,), device=device)].float()
        pre = sae.preact(x)
        z = sae.topk(pre)
        x_hat = sae.decode(z)
        e = x - x_hat
        loss = e.pow(2).sum(1).mean() / var
        fired = (z > 0).any(0)
        since_fired += len(x)
        since_fired[fired] = 0
        dead = since_fired > dead_after
        aux = torch.tensor(0.0, device=device)
        if dead.any():
            pre_dead = pre.masked_fill(~dead, -float("inf"))
            kk = min(k_aux, int(dead.sum()))
            v, i = pre_dead.topk(kk, dim=1)
            za = torch.zeros_like(pre).scatter_(1, i, v.relu())
            e_hat = za @ sae.W_dec
            aux = (e.detach() - e_hat).pow(2).sum(1).mean() / var
        opt.zero_grad()
        (loss + args.aux_coef * aux).backward()
        sae.project_decoder_grad()
        opt.step()
        sae.normalize_decoder()
        if step % 1000 == 0 or step == args.steps - 1:
            row = dict(step=step, nmse=round(loss.item(), 4), aux=round(aux.item(), 4),
                       dead_frac=round(dead.float().mean().item(), 4), t=round(time.time() - t0, 1))
            log.append(row)
            print(row, flush=True)

    sae.eval()
    te, _ = evaluate(sae, Xte)
    tr, fired = evaluate(sae, Xtr)
    stats = dict(layer=args.layer, d=d, n=n, k=args.k, k_aux=k_aux, aux_coef=args.aux_coef, lr=args.lr,
                 batch=args.batch, steps=args.steps, dead_after=dead_after, scale=scale,
                 train_vectors=len(Xtr), test_vectors=len(Xte), positions_per_image=POSITIONS,
                 fve_test=round(te["fve"], 4), l0_test=round(te["l0"], 2), fve_train=round(tr["fve"], 4),
                 dead_frac_train=round((fired == 0).float().mean().item(), 4),
                 seconds=round(time.time() - t0, 1), log=log)
    CACHE.mkdir(parents=True, exist_ok=True)
    path = sae_path(args.layer, args.k, args.n)
    sae.save(path, stats=stats)
    write_json(path.with_suffix(".json"), stats, indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k != "log"}))


# ---------------------------------------------------------------- examples

def sae_path(layer, k, n=None):
    """Checkpoint for an SAE trained with --k (and a non-default --n)."""
    return CACHE / (f"{layer}_k{k}" + (f"_n{n}" if n else "") + ".pt")


@torch.no_grad()
def cmd_examples(args):
    from tqdm import tqdm
    from microscope.examples import ImageNetVal, crop_size, save_crops
    from microscope.model import LayerCapture, RECEPTIVE_FIELD, preprocess
    from microscope.sae import TopKSAE

    device, model = setup(args.device)
    sae, ck = TopKSAE.load(sae_path(args.layer, args.k, args.n), device)
    ds = ImageNetVal(DATA)
    cap = LayerCapture(model, [args.layer])
    n = sae.n
    top_act = top_img = top_pos = None
    fires = torch.zeros(n, device=device, dtype=torch.float64)
    energy = torch.zeros(n, device=device, dtype=torch.float64)
    positions = 0
    bar = tqdm(total=ds.n, unit="img")
    for start, imgs, _ in ds.iter_groups():
        for i in range(0, len(imgs), args.batch):
            x = torch.from_numpy(imgs[i:i + args.batch]).to(device).permute(0, 3, 1, 2).float() / 255
            a = cap.run(model, preprocess(x))[args.layer]
            B, C, H, W = a.shape
            z = sae.encode(a.permute(0, 2, 3, 1).reshape(-1, C) * sae.scale)     # [B*HW, n]
            fires += (z > 0).sum(0)
            energy += z.pow(2).sum(0)
            positions += z.shape[0]
            vals, pos = z.view(B, H * W, n).max(dim=1)                             # [B, n]
            img = torch.arange(start + i, start + i + B, device=device)[:, None].expand(B, n)
            if top_act is not None:
                vals, pos, img = torch.cat([top_act, vals]), torch.cat([top_pos, pos]), torch.cat([top_img, img])
            top_act, order = vals.topk(min(N_SHOW, len(vals)), dim=0)
            top_pos, top_img = pos.gather(0, order), img.gather(0, order)
            hw = (H, W)
            bar.update(B)
    bar.close()
    cap.close()

    fires, energy = fires.cpu().numpy(), energy.cpu().numpy()
    act, img, pos = top_act.cpu().numpy(), top_img.cpu().numpy(), top_pos.cpu().numpy()
    live = np.where(fires > 0)[0]
    live = live[np.argsort(-energy[live], kind="stable")]               # new index = energy rank
    print(f"{args.layer}: {len(live)}/{n} live, {n - len(live)} never fire on {ds.n} images")
    W_dec = sae.W_dec.detach().cpu().numpy()
    exposed = live[: args.expose] if args.expose else live
    np.savez(CACHE / f"{args.layer}_features.npz", live=live, fires=fires, energy=energy,
             positions=positions, act=act, img=img, pos=pos, hw=np.array(hw), k=args.k, n=args.n or 0, exposed=len(exposed))
    print(f"exposing {len(exposed)} features, {energy[exposed].sum() / energy.sum():.1%} of total energy")

    out = layer_out(args.layer)
    # features are re-indexed by energy on every run: if the mapping changed (e.g. after retraining),
    # existing crops and vis belong to other features, and the skip-if-exists writers would keep them
    old = out / "features.json"
    if old.exists():
        prev = {f: v["orig"] for f, v in json.loads(old.read_text()).items()}
        if prev != {str(f): int(o) for f, o in enumerate(exposed)}:
            print("feature index changed since the last run: clearing old examples/ and vis/")
            shutil.rmtree(out / "examples", ignore_errors=True)
            shutil.rmtree(out / "vis", ignore_errors=True)
    feats = {}
    for f, o in enumerate(exposed):
        row = W_dec[o]
        tc = np.argsort(-np.abs(row))[:8]
        feats[str(f)] = dict(orig=int(o), density=float(f"{fires[o] / positions:.3g}"),
                             energy=float(f"{energy[o] / energy.sum():.3g}"),
                             top_channels=[[int(c), round(float(row[c]), 3)] for c in tc],
                             rendered=(out / "vis" / f"{f}.png").exists())
    write_json(out / "features.json", feats)

    e = exposed
    top = dict(act=act[:, e], img=img[:, e], y=pos[:, e] // hw[1], x=pos[:, e] % hw[1], hw=hw)
    size = crop_size(RECEPTIVE_FIELD[args.layer])
    t = time.time()
    meta = save_crops(ds, top, out / "examples", size, N_SHOW, max_px=args.example_px)
    write_json(out / "units.json", {str(c): rows for c, rows in sorted(meta.items())})
    print(f"crops at {size}px: {time.time() - t:.0f} s")


# ---------------------------------------------------------------- render

def cmd_render(args):
    from PIL import Image
    from microscope.sae import TopKSAE, render_features

    device, model = setup(args.device)
    meta = np.load(CACHE / f"{args.layer}_features.npz")
    sae, _ = TopKSAE.load(sae_path(args.layer, int(meta["k"]), int(meta["n"]) if "n" in meta else None), device)
    live = meta["live"][: int(meta["exposed"])] if "exposed" in meta.files else meta["live"]
    out = layer_out(args.layer) / "vis"
    out.mkdir(parents=True, exist_ok=True)
    todo = [f for f in range(len(live)) if not (out / f"{f}.png").exists()]
    if args.only:
        only = [int(f) for f in args.only.split(",")]
        todo = [f for f in only if f in set(todo)]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{args.layer}: {len(todo)} to render, batch {args.batch}, budget {args.budget_min} min", flush=True)
    t0 = time.time()
    done = 0
    for i in range(0, len(todo), args.batch):
        if args.budget_min and time.time() - t0 > args.budget_min * 60:
            break
        chunk = todo[i:i + args.batch]
        imgs = render_features(model, args.layer, sae, torch.from_numpy(live[chunk]).to(device),
                               steps=args.steps, seed=chunk[0])
        for f, im in zip(chunk, imgs):
            Image.fromarray(im).save(out / f"{f}.png", optimize=True)
        done += len(chunk)
        dt = time.time() - t0
        print(f"{done}/{len(todo)}  {dt / 60:.1f} min  {dt / done:.2f} s/feature", flush=True)
    dt = time.time() - t0
    run = dict(rendered=done, seconds=round(dt, 1), batch=args.batch, steps=args.steps,
               s_per_feature=round(dt / max(done, 1), 3), remaining=len(todo) - done)
    locked_update(CACHE / "render_timings.json", lambda t: t.setdefault(args.layer, []).append(run), indent=1)


# ---------------------------------------------------------------- curves

def curve_images(size=224, radii=(24, 48), n_orient=36, width=4, span=math.pi / 2):
    """Black arcs on white through the image center. Orientation theta = direction
    from the curve's midpoint to its circle center (so the arc's convex side points
    at theta + 180 deg). Also straight lines at the same orientations as a control."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) + 0.5
    c = size / 2
    imgs, meta = [], []
    for r in radii:
        for j in range(n_orient):
            th = 2 * math.pi * j / n_orient
            cx, cy = c + r * math.cos(th), c + r * math.sin(th)
            d = np.hypot(xx - cx, yy - cy)
            ang = np.arctan2(yy - cy, xx - cx)
            mid = th + math.pi
            dang = np.angle(np.exp(1j * (ang - mid)))
            mask = (np.abs(d - r) < width / 2) & (np.abs(dang) < span / 2)
            imgs.append(np.where(mask, 0.0, 1.0))
            meta.append(("curve", r, j))
    for j in range(n_orient // 2):
        th = math.pi * j / (n_orient // 2)
        # line through center, perpendicular to theta, same length as the larger arc
        u = (xx - c) * math.cos(th) + (yy - c) * math.sin(th)
        v = -(xx - c) * math.sin(th) + (yy - c) * math.cos(th)
        L = max(radii) * span
        mask = (np.abs(u) < width / 2) & (np.abs(v) < L / 2)
        imgs.append(np.where(mask, 0.0, 1.0))
        meta.append(("line", 0, j))
    x = torch.from_numpy(np.stack(imgs)).float()[:, None].expand(-1, 3, -1, -1)
    return x, meta


@torch.no_grad()
def cmd_curves(args):
    """Radial tuning on synthetic curves, for SAE features and raw channels.
    Response = max over the central 3x3 cells. SAE features use ReLU(preact)
    (no TopK) so a feature's own response is not hidden by competition."""
    from microscope.model import LayerCapture, preprocess
    from microscope.sae import TopKSAE

    device, model = setup(args.device)
    sae, _ = TopKSAE.load(sae_path(args.layer, args.k, args.n), device)
    meta_f = np.load(CACHE / f"{args.layer}_features.npz")
    live = meta_f["live"]
    x, meta = curve_images()
    cap = LayerCapture(model, [args.layer])
    a = torch.cat([cap.run(model, preprocess(x[i:i + 64].to(device)))[args.layer] for i in range(0, len(x), 64)])
    cap.close()
    N, C, H, W = a.shape
    cy, cx = H // 2, W // 2
    win = a[:, :, cy - 1:cy + 2, cx - 1:cx + 2]                               # [N, C, 3, 3]
    ch_resp = win.amax(dim=(2, 3)).cpu().numpy()                              # [N, C]
    v = win.permute(0, 2, 3, 1).reshape(-1, C) * sae.scale
    f_resp = sae.preact(v).relu().view(N, 9, -1).amax(1).cpu().numpy()[:, live]  # [N, F_live]

    kinds = np.array([m[0] for m in meta])
    radii = np.array([m[1] for m in meta])
    orient = np.array([m[2] for m in meta])
    n_or = orient[kinds == "curve"].max() + 1

    def summarize(resp, ref_top):
        """resp [N, U]; ref_top [U] = mean top-16 dataset activation of each unit."""
        cur = resp[kinds == "curve"].reshape(-1, n_or, resp.shape[1]).max(0)       # [36, U] best radius
        line = resp[kinds == "line"].max(0)
        peak = cur.max(0)
        theta = 2 * np.pi * np.arange(n_or) / n_or
        vec = (cur * np.exp(1j * theta)[:, None]).sum(0) / np.maximum(cur.sum(0), 1e-9)
        return dict(peak=peak, rel=peak / np.maximum(ref_top, 1e-9), line_ratio=peak / np.maximum(line, 1e-9),
                    selectivity=np.abs(vec), pref=(np.degrees(np.angle(vec)) % 360), tuning=cur)

    ch_top = np.load(ROOT / "cache" / f"topk_{args.layer}.npz")["act"][:N_SHOW].mean(0)
    f_top = meta_f["act"][:, live].mean(0)
    S_ch, S_f = summarize(ch_resp, ch_top), summarize(f_resp, f_top)

    def is_curve(S, i):
        return S["rel"][i] > args.rel and S["line_ratio"][i] > args.line_ratio and S["selectivity"][i] > args.sel

    W_dec = sae.W_dec.detach().cpu().numpy()[live]
    curve_mass = (W_dec[:, CURVE_CHANNELS] ** 2).sum(1)                        # unit rows: share in [0,1]
    res = dict(criteria=dict(rel=args.rel, line_ratio=args.line_ratio, selectivity=args.sel),
               channels={}, features={})
    for c in range(C):
        if is_curve(S_ch, c) or c in CURVE_CHANNELS:
            res["channels"][c] = dict(curve=bool(is_curve(S_ch, c)), known=c in CURVE_CHANNELS,
                                      **{k: round(float(S_ch[k][c]), 3) for k in ("rel", "line_ratio", "selectivity", "pref")},
                                      tuning=[round(float(t), 2) for t in S_ch["tuning"][:, c]])
    for f in range(len(live)):
        if is_curve(S_f, f):
            tc = np.argsort(-np.abs(W_dec[f]))[:5]
            res["features"][f] = dict(orig=int(live[f]), curve_channel_mass=round(float(curve_mass[f]), 3),
                                      top_channels=[[int(c), round(float(W_dec[f, c]), 3)] for c in tc],
                                      **{k: round(float(S_f[k][f]), 3) for k in ("rel", "line_ratio", "selectivity", "pref")},
                                      tuning=[round(float(t), 2) for t in S_f["tuning"][:, f]])
    write_json(CACHE / f"curves_{args.layer}.json", res, indent=1)
    print(f"channels flagged: {sorted(int(c) for c, r in res['channels'].items() if r['curve'])}")
    print(f"known curve channels passing: {[c for c in CURVE_CHANNELS if is_curve(S_ch, c)]}")
    print(f"SAE features flagged: {len(res['features'])}")
    for f, r in sorted(res["features"].items(), key=lambda kv: kv[1]["pref"]):
        print(f, r["orig"], "pref", r["pref"], "sel", r["selectivity"], "rel", r["rel"],
              "curve_mass", r["curve_channel_mass"], r["top_channels"][:3])


# ---------------------------------------------------------------- manifest

def cmd_manifest(args):
    from microscope.examples import crop_size
    from microscope.model import RECEPTIVE_FIELD

    from microscope.model import LAYERS

    base = {l["name"]: l for l in json.loads((SITE / "manifest.json").read_text())["models"][0]["layers"]}
    path = SITE / "manifest_sae.json"
    # layers not named in --layers keep their existing entries untouched
    refresh = args.layers.split(",") if args.layers else SAE_LAYERS
    kept = {l["name"]: l for l in json.loads(path.read_text())["models"][0]["layers"]} if path.exists() else {}
    layers = []
    for name in LAYERS:
        if name not in refresh:
            if name in kept:
                layers.append(kept[name])
            continue
        d = layer_out(name)
        if not (d / "features.json").exists():
            continue
        feats = json.loads((d / "features.json").read_text())
        for f, r in feats.items():
            r["rendered"] = (d / "vis" / f"{f}.png").exists()
        write_json(d / "features.json", feats)
        units = json.loads((d / "units.json").read_text()) if (d / "units.json").exists() else {}
        stats_path = CACHE / f"{name}_features.npz"
        st = np.load(stats_path) if stats_path.exists() else None
        n_total = int(st["fires"].shape[0]) if st is not None else len(feats)
        n_live = len(st["live"]) if st is not None else len(feats)
        entry = dict(name=name, channels=len(feats), spatial=base[name]["spatial"],
                     receptive_field=RECEPTIVE_FIELD[name], crop_size=crop_size(RECEPTIVE_FIELD[name]),
                     rendered=sum(r["rendered"] for r in feats.values()), examples=len(units),
                     circuits=False, branches=None, base_layer=name, dead_removed=n_total - n_live)
        if len(feats) < n_live:
            entry.update(total_features=n_live, exposed=f"top {len(feats)} by energy")
        if args.example_px:
            entry["example_px"] = args.example_px
        layers.append(entry)
    p = f"/assets/{MODEL_NAME}"
    manifest = dict(models=[dict(
        name=MODEL_NAME, title="InceptionV1 · SAE features",
        description="TopK sparse autoencoder features (8x expansion) trained on InceptionV1 layer "
                    "activations; each feature is shown like a unit.",
        dataset="ImageNet ILSVRC2012 validation, 50,000 images, center-cropped to 256px",
        examples_per_unit=N_SHOW, layers=layers,
        paths=dict(vis=f"{p}/{{layer}}/vis/{{unit}}.png",
                   example=f"{p}/{{layer}}/examples/{{unit}}/{{rank}}.jpg",
                   units=f"{p}/{{layer}}/units.json",
                   features=f"{p}/{{layer}}/features.json"),
    )])
    write_json(path, manifest, indent=1)
    print(json.dumps([l for l in layers if l["name"] in refresh], indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("collect"); s.add_argument("--images", type=int)
    s.add_argument("--layers", help=f"comma-separated (default {','.join(SAE_LAYERS)})"); s.add_argument("--batch", type=int, default=100)
    s = sub.add_parser("train")
    s.add_argument("--layer", required=True); s.add_argument("--k", type=int, default=32)
    s.add_argument("--n", type=int, help="latents (default 8*d)")
    s.add_argument("--steps", type=int, default=15000); s.add_argument("--batch", type=int, default=4096)
    s.add_argument("--lr", type=float, default=1e-3); s.add_argument("--k-aux", type=int)
    s.add_argument("--aux-coef", type=float, default=1 / 32)
    s.add_argument("--dead-after", type=int, default=1_000_000, help="samples without firing -> dead (AuxK)")
    s = sub.add_parser("examples")
    s.add_argument("--layer", required=True); s.add_argument("--k", type=int, required=True)
    s.add_argument("--n", type=int, help="latents, if trained with a non-default --n")
    s.add_argument("--batch", type=int, default=50)
    s.add_argument("--expose", type=int, default=0, help="only the top-N features by energy get assets (0 = all)")
    s.add_argument("--example-px", type=int, default=0, help="store crops downscaled to this size (0 = native)")
    s = sub.add_parser("render")
    s.add_argument("--layer", required=True); s.add_argument("--batch", type=int, default=64)
    s.add_argument("--steps", type=int, default=512); s.add_argument("--budget-min", type=float, default=0)
    s.add_argument("--limit", type=int); s.add_argument("--only", help="comma-separated feature ids (new index)")
    s = sub.add_parser("curves")
    s.add_argument("--layer", default="mixed3b"); s.add_argument("--k", type=int, required=True)
    s.add_argument("--n", type=int, help="latents, if trained with a non-default --n")
    s.add_argument("--rel", type=float, default=0.3, help="peak curve response / mean top-16 dataset act")
    s.add_argument("--line-ratio", type=float, default=1.2, help="peak curve / peak straight line")
    s.add_argument("--sel", type=float, default=0.3, help="orientation selectivity (1 - circular variance)")
    s = sub.add_parser("manifest")
    s.add_argument("--layers", help=f"layers to refresh; others keep their entries (default {','.join(SAE_LAYERS)})")
    s.add_argument("--example-px", type=int, default=0, help="record example_px on the refreshed layers")
    for s in sub.choices.values():
        s.add_argument("--device")
    args = p.parse_args()
    dict(collect=cmd_collect, train=cmd_train, examples=cmd_examples, render=cmd_render,
         curves=cmd_curves, manifest=cmd_manifest)[args.cmd](args)


if __name__ == "__main__":
    main()
