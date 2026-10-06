"""Sanity check that a loaded model classifies sensibly, on a fixed ImageNet val subset.

    python evaluate.py --model vgg19         # top-1 / top-5 on every 100th val image (500)
    python evaluate.py --model clip_rn50     # zero-shot, "a photo of a {class}"

The 256px center-cropped val images are center-cropped to 224 (VGG19, torchvision's
resize-256 + crop-224) or resized to 224 (CLIP: its resize-224 + crop on a square).
Class names for CLIP are the first synonym of each ImageNet class.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from microscope.files import write_json

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "imagenet-val-256"


def load_subset(n, stride):
    from microscope.examples import ImageNetVal
    ds = ImageNetVal(DATA)
    wanted = set(range(0, stride * n, stride))
    imgs, labels = [], []
    for s, ims, labs in ds.iter_groups(wanted):
        for j in range(len(ims)):
            if s + j in wanted:
                imgs.append(ims[j])
                labels.append(int(labs[j]))
    return np.stack(imgs), np.array(labels), ds.class_names()


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=["vgg19", "clip_rn50"])
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--stride", type=int, default=100)
    p.add_argument("--device")
    p.add_argument("--batch", type=int, default=50)
    args = p.parse_args()

    from microscope.device import pick_device
    from microscope.models import get_spec, load_clip

    device = pick_device(args.device)
    spec = get_spec(args.model)
    imgs, labels, names = load_subset(args.n, args.stride)

    if args.model == "clip_rn50":
        import open_clip
        clip = load_clip(device)
        tok = open_clip.get_tokenizer("RN50-quickgelu")
        text = torch.cat([clip.encode_text(tok([f"a photo of a {c}." for c in names[i:i + 200]]).to(device))
                          for i in range(0, len(names), 200)])
        text = F.normalize(text, dim=-1)
        model = lambda x: F.normalize(clip.encode_image(x), dim=-1) @ text.T
    else:
        model = spec.load(device)

    logits = []
    for i in range(0, len(imgs), args.batch):
        x = torch.from_numpy(imgs[i:i + args.batch]).to(device).permute(0, 3, 1, 2).float() / 255
        if args.model == "clip_rn50":
            x = F.interpolate(x, size=224, mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        else:
            x = x[:, :, 16:240, 16:240]
        logits.append(model(spec.preprocess(x)).float().cpu())
    logits = torch.cat(logits)
    y = torch.from_numpy(labels)
    top5 = logits.topk(5, dim=1).indices
    res = dict(model=args.model, n=len(y), top1=round((top5[:, 0] == y).float().mean().item(), 4),
               top5=round((top5 == y[:, None]).any(1).float().mean().item(), 4),
               protocol="zero-shot, 'a photo of a {class}.'" if args.model == "clip_rn50" else "supervised")
    print(json.dumps(res))
    out = ROOT / "cache" / args.model
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "eval.json", res, indent=1)


if __name__ == "__main__":
    main()
