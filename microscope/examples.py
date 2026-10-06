"""Dataset examples: the ImageNet val crops that activate each channel most.

One streaming pass over the 50k images keeps a running top-k per channel of
(spatial-max activation, image index, y, x) for every layer at once, so later
layers never need another pass. Crops are centered on the argmax position.
"""
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

from .model import LAYERS, LayerCapture, preprocess

IMAGE_SIZE = 256  # the parquet images are center-cropped squares at 256px


class ImageNetVal:
    def __init__(self, root):
        self.files = sorted(Path(root).glob("*.parquet"))
        if not self.files:
            raise FileNotFoundError(f"no parquet files in {root}")
        # global index -> (file, row group, offset)
        self.groups = []
        for f in self.files:
            meta = pq.ParquetFile(f).metadata
            for g in range(meta.num_row_groups):
                self.groups.append((f, g, meta.row_group(g).num_rows))
        self.n = sum(g[2] for g in self.groups)

    def class_names(self):
        meta = pq.ParquetFile(self.files[0]).schema_arrow.metadata[b"huggingface"]
        names = json.loads(meta)["info"]["features"]["label"]["names"]
        return [n.split(",")[0] for n in names]

    def iter_groups(self, wanted=None):
        """Yield (start_index, images uint8 [N,256,256,3], labels) per row group.
        If `wanted` is a set of global indices, only groups containing one are read and only
        the wanted rows are decoded; the other rows of such a group are left as zeros."""
        wanted = None if wanted is None else np.unique(np.fromiter(wanted, dtype=np.int64))
        files = {f: pq.ParquetFile(f) for f in self.files}
        start = 0
        with ThreadPoolExecutor(8) as pool:
            for f, g, n in self.groups:
                if wanted is None:
                    rows = None
                else:
                    lo, hi = np.searchsorted(wanted, [start, start + n])
                    rows = wanted[lo:hi] - start
                if rows is None or len(rows):
                    t = files[f].read_row_group(g, columns=["image", "label"])
                    blobs = [r["bytes"] for r in t.column("image").to_pylist()]
                    if rows is None:
                        imgs = np.stack(list(pool.map(_decode, blobs)))
                    else:
                        imgs = np.zeros((n, IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8)
                        imgs[rows] = np.stack(list(pool.map(_decode, [blobs[i] for i in rows])))
                    yield start, imgs, t.column("label").to_numpy()
                start += n

    def load(self, wanted):
        """Dense uint8 [n_images,256,256,3] with only `wanted` rows filled (untouched pages cost no RAM)."""
        images = np.zeros((self.n, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        for s, ims, _ in self.iter_groups(wanted):
            images[s:s + len(ims)] = ims
        return images


def _decode(b):
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))


@torch.no_grad()
def topk_pass(model, device, dataset, layers=LAYERS, k=32, batch=200, progress=True, spec=None):
    """Return {layer: dict(act [k,C], img [k,C], y [k,C], x [k,C], hw (H,W))}.
    spec: a microscope.models.ModelSpec (default InceptionV1)."""
    from tqdm import tqdm

    cap = LayerCapture(model, layers) if spec is None else LayerCapture(model, layers, spec.layers, spec.modules)
    prep = preprocess if spec is None else spec.preprocess
    state = {}
    pending = []

    def flush(chunk_imgs, chunk_idx):
        x = torch.from_numpy(chunk_imgs).to(device).permute(0, 3, 1, 2).float() / 255
        out = cap.run(model, prep(x))
        idx = torch.from_numpy(chunk_idx).to(device)
        for name in layers:
            a = out[name]
            B, C, H, W = a.shape
            vals, pos = a.reshape(B, C, H * W).max(dim=2)          # [B, C]
            img = idx[:, None].expand(B, C)
            if name in state:
                s = state[name]
                vals = torch.cat([s["act"], vals])
                pos = torch.cat([s["pos"], pos])
                img = torch.cat([s["img"], img])
            top, order = vals.topk(min(k, vals.shape[0]), dim=0)    # [k, C]
            state[name] = dict(act=top, pos=pos.gather(0, order), img=img.gather(0, order), hw=(H, W))

    buf, buf_idx = [], []
    bar = tqdm(total=dataset.n, unit="img", disable=not progress, desc="dataset pass")
    for start, imgs, _ in dataset.iter_groups():
        buf.append(imgs)
        buf_idx.append(np.arange(start, start + len(imgs)))
        if sum(len(b) for b in buf) >= batch:
            # row groups can be much larger than `batch`; flush in slices of at most `batch`
            ims, ids = np.concatenate(buf), np.concatenate(buf_idx)
            for s in range(0, len(ims) - len(ims) % batch, batch):
                flush(ims[s:s + batch], ids[s:s + batch])
            rest = len(ims) - len(ims) % batch
            buf, buf_idx = ([ims[rest:]], [ids[rest:]]) if rest < len(ims) else ([], [])
            bar.update(rest)
    if buf:
        flush(np.concatenate(buf), np.concatenate(buf_idx))
        bar.update(sum(len(b) for b in buf))
    bar.close()
    cap.close()

    result = {}
    for name, s in state.items():
        H, W = s["hw"]
        pos = s["pos"].cpu().numpy()
        result[name] = dict(
            act=s["act"].cpu().numpy().astype(np.float32),
            img=s["img"].cpu().numpy().astype(np.int64),
            y=pos // W, x=pos % W, hw=np.array([H, W]),
        )
    return result


def crop_size(rf):
    """Crop side in 256px image pixels. The theoretical receptive field
    overstates what a unit responds to (the effective RF is roughly Gaussian and
    much smaller), so take ~3/4 of it, clamped to [32, 224]."""
    return int(min(224, max(32, round(0.75 * rf / 8) * 8)))


def crop_box(y, x, hw, size):
    """Center a size x size box on feature-map cell (y, x), kept inside the image."""
    H, W = hw
    cy = (y + 0.5) * IMAGE_SIZE / H
    cx = (x + 0.5) * IMAGE_SIZE / W
    top = int(round(min(max(cy - size / 2, 0), IMAGE_SIZE - size)))
    left = int(round(min(max(cx - size / 2, 0), IMAGE_SIZE - size)))
    return top, left, size


def save_crops(dataset, top, out_dir, size, n_show, channels=None, max_px=None):
    """Write out_dir/<c>/<rank>.jpg for each channel; return per-channel metadata.
    max_px: downscale saved crops larger than this (box metadata stays in image pixels)."""
    C = top["act"].shape[1]
    channels = range(C) if channels is None else channels
    names = dataset.class_names()
    # image index -> [(channel, rank)], so each image is decoded once and cropped on the spot
    need = {}
    for c in channels:
        for r in range(n_show):
            need.setdefault(int(top["img"][r, c]), []).append((c, r))

    meta = {c: [None] * n_show for c in channels}
    for start, imgs, labels in dataset.iter_groups(need.keys()):
        for j in range(len(imgs)):
            for c, r in need.get(start + j, ()):
                t, l, s = crop_box(int(top["y"][r, c]), int(top["x"][r, c]), top["hw"], size)
                path = Path(out_dir) / str(c) / f"{r}.jpg"
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    im = Image.fromarray(imgs[j][t:t + s, l:l + s])
                    if max_px and s > max_px:
                        im = im.resize((max_px, max_px), Image.LANCZOS)
                    im.save(path, quality=90)
                meta[c][r] = dict(img=start + j, act=round(float(top["act"][r, c]), 3),
                                  label=names[int(labels[j])], box=[t, l, s])
    return meta
