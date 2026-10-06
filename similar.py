"""Similar units across networks, measured by behavior on shared images.

    python similar.py collect --models inceptionv1 --sae --device cuda:0   # GPU: spatial-max activations
    python similar.py collect --models vgg19,clip_rn50 --device cuda:1
    python similar.py match --out <root>                                   # GPU: correlations -> <root>/<model>/<layer>/similar/<shard>.json
    python similar.py summary --stats <root>/stats.json                    # Mac, JSON only: sanity checks -> cache/similar_summary.json

Method: every network sees the same N ImageNet val images (a fixed-seed random subset, 256px,
each network with its own preprocessing). Each unit's response to an image is the spatial max of
its post-ReLU activation (the statistic its dataset examples are ranked by), so each unit is a
length-N vector. Similarity is the Pearson correlation of two units' vectors across the images.
SAE features (inceptionv1_sae, --sae) are the TopK SAE codes at every position, spatial max, in
the exposed index order of site/assets/inceptionv1_sae/<layer>/features.json (`orig` -> f);
features not exposed in the viewer are left out.

Output, sharded so a unit page loads 256 units' worth: site/assets/<model>/<layer>/similar/<unit // 256>.json
    {"<unit>": {"<other_model>": [[layer, unit, r], ...], ...}}
An entry has a 4th element, 1, when the matched unit is active (spatial max > 0) on < 1% of the images:
its r rests on few images and is noisy (in practice only SAE features).
Keys are model names (inceptionv1, vgg19, clip_rn50, inceptionv1_sae). Lists are sorted by r,
highest first, r rounded to 3 decimals: the top 8 in every other model and, under the unit's own
model name, the top 5 in its own model's other layers (same layer excluded). inceptionv1 and
inceptionv1_sae count as different models, so a channel's list of SAE features includes its own layer.
Units with zero variance on the subset (never active) get no entry.

Activations live in RAM (/dev/shm/microscope_similar, ~1.5 GB for 8,000 images); only the JSON is written to disk.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "site"
SHM = Path(os.environ.get("SIMILAR_SHM", "/dev/shm/microscope_similar"))
SAE = "inceptionv1_sae"
K_OTHER, K_OWN = 8, 5
SHARD = 256          # units per similar/<k>.json
RARE = 0.01          # active on fewer images than this: flagged as noisy


def subset(n_total, n, seed):
    return np.sort(np.random.default_rng(seed).choice(n_total, n, replace=False))


# ---------------------------------------------------------------- collect

@torch.no_grad()
def cmd_collect(args):
    import warnings
    warnings.filterwarnings("ignore")
    from tqdm import tqdm
    from microscope.device import patch_lucent, pick_device
    from microscope.examples import ImageNetVal
    from microscope.model import LayerCapture
    from microscope.models import get_spec
    from microscope.sae import TopKSAE

    device = pick_device(args.device)
    patch_lucent(device)
    ds = ImageNetVal(args.data)
    idx = subset(ds.n, args.n, args.seed)
    t0 = time.time()
    images = np.zeros((len(idx), 256, 256, 3), np.uint8)
    k = 0
    for start, imgs, _ in ds.iter_groups(idx):
        rows = idx[(idx >= start) & (idx < start + len(imgs))] - start
        images[k:k + len(rows)] = imgs[rows]
        k += len(rows)
    assert k == len(idx)
    print(f"decoded {len(idx)} images in {time.time() - t0:.0f} s", flush=True)
    SHM.mkdir(parents=True, exist_ok=True)

    for name in args.models.split(","):
        t0 = time.time()
        spec = get_spec(name)
        model = spec.load(device)
        cap = LayerCapture(model, spec.layers, spec.layers, spec.modules)
        saes = {}
        if args.sae and name == "inceptionv1":
            for layer in spec.layers:
                ck, fj = Path(args.sae_dir) / f"{layer}_k32.pt", Path(args.sae_features) / layer / "features.json"
                if ck.exists() and fj.exists():
                    sae, _ = TopKSAE.load(ck, device)
                    feats = json.loads(fj.read_text())
                    orig = torch.tensor([feats[str(f)]["orig"] for f in range(len(feats))], device=device)
                    saes[layer] = (sae, orig)
        out = {l: [] for l in spec.layers}
        sae_out = {l: [] for l in saes}
        for s in tqdm(range(0, len(idx), args.batch), desc=name, mininterval=10):
            x = torch.from_numpy(images[s:s + args.batch]).to(device).permute(0, 3, 1, 2).float() / 255
            acts = cap.run(model, spec.preprocess(x))
            for l in spec.layers:
                a = acts[l]
                out[l].append(a.flatten(2).amax(2).cpu())
                if l in saes:
                    sae, orig = saes[l]
                    B, C, H, W = a.shape
                    z = sae.encode(a.permute(0, 2, 3, 1).reshape(-1, C) * sae.scale)[:, orig]
                    sae_out[l].append(z.view(B, H * W, -1).amax(1).cpu())
        cap.close()
        save(name, spec.layers, out, idx)
        if saes:
            save(SAE, list(saes), sae_out, idx)
        del model, saes
        torch.cuda.empty_cache()
        print(f"{name}: {time.time() - t0:.0f} s", flush=True)


def save(name, layers, out, idx):
    mats = [torch.cat(out[l]).numpy().astype(np.float32) for l in layers]
    np.savez(SHM / f"{name}.npz", acts=np.concatenate(mats, 1), img=idx, layers=np.array(layers),
             layer_of=np.concatenate([np.full(m.shape[1], i) for i, m in enumerate(mats)]),
             unit_of=np.concatenate([np.arange(m.shape[1]) for m in mats]))
    print(f"saved {name}: {sum(m.shape[1] for m in mats)} units", flush=True)


# ---------------------------------------------------------------- match

def standardize(X):
    """[N, U] -> zero-mean, unit-norm columns (so Z_a . Z_b is Pearson r); constant columns -> 0."""
    X = X - X.mean(0)
    norm = X.norm(dim=0)
    dead = norm < 1e-6 * max(1.0, norm.max().item())
    return X / norm.clamp_min(1e-12) * (~dead), dead


def top_matches(A, B, groups, a_layer=None, chunk=2048):
    """For every column of A, its top matches among columns of B, per group.
    A, B: standardized [N, *]; groups: {key: dict(cols=slice of B, k=int, layer=layer ids of those cols,
    mask=bool)}; with mask, columns in the same layer as the source (a_layer) are excluded.
    Returns {key: (r [UA, k], column index into B [UA, k])}."""
    U = A.shape[1]
    res = {g: (torch.empty(U, v["k"]), torch.empty(U, v["k"], dtype=torch.long)) for g, v in groups.items()}
    for s in range(0, U, chunk):
        e = min(U, s + chunk)
        C = A[:, s:e].T @ B                                                  # [chunk, UB]
        for g, v in groups.items():
            block = C[:, v["cols"]]
            if v.get("mask"):
                block = block.masked_fill(a_layer[s:e, None] == v["layer"][None, :], -2)
            val, col = block.topk(v["k"], dim=1)
            res[g][0][s:e], res[g][1][s:e] = val.cpu(), col.cpu() + v["cols"].start
    return res


@torch.no_grad()
def cmd_match(args):
    t0 = time.time()
    device = torch.device(args.device)
    names = [n for n in ["inceptionv1", "vgg19", "clip_rn50", SAE] if (SHM / f"{n}.npz").exists()]
    data = {n: np.load(SHM / f"{n}.npz") for n in names}
    img = data[names[0]]["img"]
    assert all((d["img"] == img).all() for d in data.values()), "models saw different images"
    span, start = {}, 0
    for n in names:
        u = data[n]["acts"].shape[1]
        span[n] = slice(start, start + u)
        start += u
    X = torch.from_numpy(np.concatenate([data[n]["acts"] for n in names], 1)).to(device)
    N, U = X.shape
    print(f"{N} images x {U} units ({', '.join(f'{n} {span[n].stop - span[n].start}' for n in names)})", flush=True)
    layer_of = torch.cat([torch.from_numpy(data[n]["layer_of"]) + 100 * i for i, n in enumerate(names)]).to(device)
    model_of = np.concatenate([np.full(span[n].stop - span[n].start, i) for i, n in enumerate(names)])
    active = (X > 0).float().mean(0).cpu().numpy()

    Z, dead = standardize(X)
    # SAE features and channels of the same network: matching each other is not "own network, other layers"
    family = {n: ("inceptionv1" if n == SAE else n) for n in names}

    def groups_for(src, k_other=K_OTHER):
        g = {}
        for n in names:
            if n == src:
                g[n] = dict(cols=span[n], k=K_OWN, layer=layer_of[span[n]], mask=True)
            else:
                g[n] = dict(cols=span[n], k=k_other, layer=layer_of[span[n]])
        return g

    out_root = Path(args.out)
    stats = dict(n_images=int(N), seed=args.seed, method="Pearson r of per-image spatial-max activation", models={})
    for src in names:
        sl = span[src]
        res = top_matches(Z[:, sl], Z, groups_for(src), layer_of[sl])
        d = data[src]
        layers = [str(l) for l in d["layers"]]
        files = {l: {} for l in layers}
        for i in range(sl.stop - sl.start):
            if dead[sl.start + i]:
                continue
            entry = {}
            for n, (vals, cols) in res.items():
                ent = []
                for v, c in zip(vals[i].tolist(), cols[i].tolist()):
                    if dead[c] or v <= -1.5:
                        continue
                    j = c - span[n].start
                    ent.append([str(data[n]["layers"][data[n]["layer_of"][j]]), int(data[n]["unit_of"][j]), round(v, 3)]
                               + ([1] if active[c] < RARE else []))
                entry[n] = ent
            files[layers[d["layer_of"][i]]][str(int(d["unit_of"][i]))] = entry
        for l, f in files.items():
            shards = {}
            for u, e in f.items():
                shards.setdefault(int(u) // SHARD, {})[u] = e
            (out_root / src / l / "similar").mkdir(parents=True, exist_ok=True)
            for k, sh in shards.items():
                (out_root / src / l / "similar" / f"{k}.json").write_text(json.dumps(sh, separators=(",", ":")))
        # best cross-network r per unit, for the summary
        best = {n: res[n][0][:, 0].numpy() for n in names if family[n] != family[src]}
        ms = dict(units=int(sl.stop - sl.start), dead=int(dead[sl].sum().item()),
                  rarely_active=int((active[sl] < RARE).sum()),
                  best_r_median={n: round(float(np.median(b[~dead[sl].cpu().numpy()])), 3) for n, b in best.items()})
        stats["models"][src] = ms
        print(f"{src}: wrote {len(layers)} layers; {ms}", flush=True)

    # split-half stability: top-1 match in each other network from two disjoint halves of the images
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(1)).to(device)
    halves = [standardize(X[perm[: N // 2]])[0], standardize(X[perm[N // 2:]])[0]]
    stats["split_half"] = {}
    for src in [n for n in names if n != SAE]:
        sl = span[src]
        g = {n: v for n, v in groups_for(src).items() if family[n] != family[src]}
        ra, rb = (top_matches(h[:, sl], h, g) for h in halves)
        live = ~dead[sl].cpu().numpy()
        stats["split_half"][src] = {n: dict(
            top1_same=round(float((ra[n][1][:, 0] == rb[n][1][:, 0]).numpy()[live].mean()), 3),
            top1_in_top8=round(float((ra[n][1][:, :1] == rb[n][1]).any(1).numpy()[live].mean()), 3))
            for n in g}
    # null: best r against the other network with its images shuffled (no shared content)
    stats["null_best_r_median"] = {}
    for src in [n for n in names if n != SAE]:
        sl = span[src]
        for n in names:
            if family[n] == family[src]:
                continue
            shuffled = Z[torch.randperm(N, generator=torch.Generator().manual_seed(2)).to(device)][:, span[n]]
            r = top_matches(Z[:, sl], shuffled, {n: dict(cols=slice(0, shuffled.shape[1]), k=1)})[n][0][:, 0]
            stats["null_best_r_median"][f"{src}->{n}"] = round(float(r.median()), 3)
    stats["seconds_match"] = round(time.time() - t0, 1)
    (out_root / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


# ---------------------------------------------------------------- summary (no GPU)

CHECKS = [
    ("curve detectors (Distill)", "inceptionv1", "mixed3b", [379, 406, 385]),
]


def cmd_summary(args):
    assets = SITE / "assets"
    labels_cache = {}

    def label(model, layer, unit):
        if (model, layer) not in labels_cache:
            p = assets / model / layer / "labels.json"
            labels_cache[model, layer] = json.loads(p.read_text()) if p.exists() else {}
        return labels_cache[model, layer].get(str(unit), {}).get("label")

    def show(model, layer, unit, n=5):
        sim = json.loads((assets / model / layer / "similar" / f"{unit // SHARD}.json").read_text()).get(str(unit), {})
        return dict(unit=f"{model}/{layer}/{unit}", label=label(model, layer, unit), matches={
            m: [dict(unit=f"{e[0]}/{e[1]}", r=e[2], label=label(m, e[0], e[1])) for e in lst[:n]] for m, lst in sim.items()})

    checks = []
    for title, model, layer, units in CHECKS:
        checks.append(dict(check=title, units=[show(model, layer, u) for u in units]))
    # units whose labels mention a term, a few per network, matched across networks
    for term in args.terms.split(","):
        found = []
        for model in ["inceptionv1", "vgg19", "clip_rn50"]:
            hits = []
            for p in sorted((assets / model).glob("*/labels.json")):
                for u, v in json.loads(p.read_text()).items():
                    if term in v["label"].lower():
                        hits.append((p.parent.name, int(u)))
            found += [show(model, l, u) for l, u in hits[-2:]]           # deepest two
        checks.append(dict(check=f"label contains '{term}'", units=found))

    def overlap(term, c):
        """Share of cross-network top-3 matches whose label also contains the term."""
        tot = hit = 0
        for u in c["units"]:
            for m, lst in u["matches"].items():
                if m == u["unit"].split("/")[0]:
                    continue
                for e in lst[:3]:
                    tot += 1
                    hit += term in (e["label"] or "").lower()
        return round(hit / tot, 3) if tot else None
    for c, term in zip(checks[len(CHECKS):], args.terms.split(",")):
        c["cross_network_top3_label_hit_rate"] = overlap(term, c)

    sizes, files = {}, []
    for p in assets.glob("*/*/similar/*.json"):
        m = p.parts[-4]
        sizes[m] = sizes.get(m, 0) + p.stat().st_size
        files.append((p.stat().st_size, str(p.relative_to(assets))))
    stats = json.loads(Path(args.stats).read_text()) if args.stats else {}
    # SAE entries use features.json's orig -> f mapping; if a layer is re-indexed, re-run collect --sae + match
    import hashlib
    sae_md5 = {p.parent.name: hashlib.md5(p.read_bytes()).hexdigest() for p in sorted((assets / SAE).glob("*/features.json"))}
    out = dict(stats=stats, bytes_per_model=sizes, files=len(files), largest_shard=max(files, default=None), sae_features_md5=sae_md5, checks=checks)
    (ROOT / "cache" / "similar_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1)[:6000])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--models", default="inceptionv1,vgg19,clip_rn50")
    c.add_argument("--sae", action="store_true", help="also SAE features (with inceptionv1)")
    c.add_argument("--device")
    c.add_argument("--n", type=int, default=8000)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--batch", type=int, default=32)
    c.add_argument("--data", default=str(ROOT / "data" / "imagenet-val-256"))
    c.add_argument("--sae-dir", default=str(ROOT / "cache" / "sae"))
    c.add_argument("--sae-features", default=str(SITE / "assets" / SAE), help="dir with <layer>/features.json")
    m = sub.add_parser("match")
    m.add_argument("--out", default=str(SITE / "assets"))
    m.add_argument("--device", default="cuda:0")
    m.add_argument("--seed", type=int, default=0, help="recorded only; must match collect")
    s = sub.add_parser("summary")
    s.add_argument("--stats", help="stats.json written by match")
    s.add_argument("--terms", default="dog,curve,face,text,eye")
    args = p.parse_args()
    {"collect": cmd_collect, "match": cmd_match, "summary": cmd_summary}[args.cmd](args)


if __name__ == "__main__":
    main()
