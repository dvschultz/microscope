"""Model registry: everything the pipeline needs to know per network.

    from microscope.models import get_spec
    spec = get_spec("clip_rn50")
    model = spec.load(device)            # eval mode, ReLUs not in-place
    x = spec.preprocess(img01)           # [0,1] RGB -> what the network expects

Each spec names its Microscope layers in forward order and maps each to the module
whose output is the layer (post-activation). InceptionV1 keeps the module names of
lucent's port, so `modules` is the identity there.
"""
from dataclasses import dataclass, field
from typing import Callable

import torch

from . import model as inception


@dataclass
class ModelSpec:
    name: str
    title: str
    description: str
    layers: list
    modules: dict                     # layer -> module path (model.get_submodule)
    receptive_field: dict             # layer -> theoretical RF in input pixels
    load: Callable                    # (device, pretrained=True) -> nn.Module
    preprocess: Callable              # [0,1] RGB tensor -> network input
    pre_relu: dict = None             # layer -> module whose *input* is the pre-ReLU tensor
    groups: dict = field(default_factory=dict)        # layer -> architecture stage
    notes: dict = field(default_factory=dict)         # layer -> short note for the viewer
    architecture: dict = None         # top-level summary for a generic architecture strip
    branches: Callable = None         # (model, layer) -> branch slices or None
    max_crop_px: int = None           # saved example crops larger than this are downscaled (the web export shows 128px cells)
    dataset: str = "ImageNet ILSVRC2012 validation, 50,000 images, center-cropped to 256px"

    def module(self, model, layer):
        return model.get_submodule(self.modules[layer])

    def asset_cache(self, root):
        """Cache dir for top-k / attribution files. InceptionV1 keeps the flat layout."""
        return root if self.name == inception.MODEL else root / self.name


def receptive_fields(ops_by_layer):
    """ops_by_layer: ordered [(layer or None, [(kernel, stride), ...])]; the deepest
    path through each block. Returns {layer: RF in input pixels}."""
    rf, jump, out = 1, 1, {}
    for layer, ops in ops_by_layer:
        for k, s in ops:
            rf += (k - 1) * jump
            jump *= s
        if layer:
            out[layer] = rf
    return out


def _no_inplace(model):
    for m in model.modules():
        if isinstance(m, torch.nn.ReLU):
            m.inplace = False
    return model


# ---------------------------------------------------------------- InceptionV1

def _inception_branches(model, layer):
    """Concat slices of an inception block, in channel order, with bottleneck widths."""
    if not layer.startswith("mixed"):
        return None
    conv = lambda n: getattr(model, f"{layer}_{n}_pre_relu_conv")
    out, start = [], 0
    for name, last, neck in [("1x1", "1x1", None), ("3x3", "3x3", "3x3_bottleneck"),
                             ("5x5", "5x5", "5x5_bottleneck"), ("pool", "pool_reduce", None)]:
        c = conv(last).out_channels
        out.append(dict(name=name, start=start, stop=start + c,
                        bottleneck=conv(neck).out_channels if neck else None))
        start += c
    return out


INCEPTIONV1 = ModelSpec(
    name=inception.MODEL, title="InceptionV1",
    description="GoogLeNet (Szegedy et al. 2014), TF inception5h weights via lucent.",
    layers=inception.LAYERS, modules={l: l for l in inception.LAYERS},
    receptive_field=inception.RECEPTIVE_FIELD,
    load=lambda device, pretrained=True: inception.load_model(device),
    preprocess=inception.preprocess, branches=_inception_branches,
)


# ---------------------------------------------------------------- VGG19

_VGG_BLOCKS = [(1, 2), (2, 2), (3, 4), (4, 4), (5, 4)]
VGG_LAYERS, _vgg_modules, _vgg_ops, _i = [], {}, [], 0
for _b, _n in _VGG_BLOCKS:
    for _j in range(1, _n + 1):
        _name = f"conv{_b}_{_j}"
        VGG_LAYERS.append(_name)
        _vgg_modules[_name] = f"features.{_i + 1}"   # the ReLU after conv _i
        _vgg_ops.append((_name, [(3, 1)]))
        _i += 2
    _vgg_ops.append((None, [(2, 2)]))                 # maxpool
    _i += 1

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _normalizer(mean, std):
    """(x - mean) / std, with the constants moved to each device once instead of on every call."""
    cache = {}

    def f(x):
        if x.device not in cache:
            cache[x.device] = (mean.to(x.device), std.to(x.device))
        m, s = cache[x.device]
        return (x - m) / s
    return f


_vgg_preprocess = _normalizer(_IMAGENET_MEAN, _IMAGENET_STD)


def _load_vgg19(device, pretrained=True):
    import torchvision
    w = torchvision.models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None
    model = _no_inplace(torchvision.models.vgg19(weights=w)).to(device).eval()
    assert all(isinstance(model.get_submodule(m), torch.nn.ReLU) for m in _vgg_modules.values())
    return model


VGG19 = ModelSpec(
    name="vgg19", title="VGG19",
    description="VGG19 (Simonyan & Zisserman 2014), torchvision ImageNet weights. All 16 conv layers, after ReLU. "
                "Sanity check: 67.7% top-1 on 2,000 val images of our 256px pre-resized copy "
                "(torchvision reports 72.4% on full-resolution val; the gap is likely the pre-resize and re-encoding).",
    layers=VGG_LAYERS, modules=_vgg_modules, pre_relu=_vgg_modules, receptive_field=receptive_fields(_vgg_ops),
    load=_load_vgg19, preprocess=_vgg_preprocess, max_crop_px=128,
    groups={l: f"block{l[4]}" for l in VGG_LAYERS},
    notes={f"conv{b}_1": "after 2x2 maxpool" for b in range(2, 6)},
    architecture=dict(
        kind="sequential",
        summary="16 3x3 conv layers in 5 blocks, each block ends in a 2x2 maxpool; then 3 fully connected layers (not shown).",
        stages=[dict(name=f"block{b}", layers=[f"conv{b}_{j}" for j in range(1, n + 1)],
                     note=f"{n} conv 3x3, {64 * 2 ** min(b - 1, 3)} channels, then maxpool")
                for b, n in _VGG_BLOCKS],
    ),
)


# ---------------------------------------------------------------- CLIP ResNet-50

_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


_clip_preprocess = _normalizer(_CLIP_MEAN, _CLIP_STD)


def load_clip(device, pretrained=True):
    """The full open_clip model (image + text towers), fp32."""
    import open_clip
    model = open_clip.create_model("RN50-quickgelu", pretrained="openai" if pretrained else None,
                                   precision="fp32", device=device)
    return _no_inplace(model).eval()


def _load_clip_visual(device, pretrained=True):
    return load_clip(device, pretrained).visual


# (name, block module, blocks in stage, stride)
_RN50_STAGES = [("layer1", 3, 1), ("layer2", 4, 2), ("layer3", 6, 2), ("layer4", 3, 2)]
CLIP_LAYERS = ["stem", "layer1", "layer2", "layer3_2", "layer3", "layer4_1", "layer4"]
_clip_modules = {"stem": "avgpool"}
_clip_ops = [("stem", [(3, 2), (3, 1), (3, 1), (2, 2)])]
for _s, _n, _stride in _RN50_STAGES:
    for _b in range(_n):
        # deepest path through a CLIP bottleneck: 1x1, 3x3, avgpool(stride) on the first block, 1x1
        _clip_ops.append((f"{_s}_{_b}", [(1, 1), (3, 1)] + ([(2, 2)] if _b == 0 and _stride > 1 else []) + [(1, 1)]))
        _clip_modules[f"{_s}_{_b}"] = f"{_s}.{_b}"
    _clip_modules[_s] = f"{_s}.{_n - 1}"
_clip_rf = receptive_fields(_clip_ops)
for _s, _n, _ in _RN50_STAGES:
    _clip_rf[_s] = _clip_rf[f"{_s}_{_n - 1}"]

CLIP_RN50 = ModelSpec(
    name="clip_rn50", title="CLIP ResNet-50",
    description="OpenAI CLIP image encoder, modified ResNet-50 (Radford et al. 2021), via open_clip. "
                "Stem output plus the output of the last block of each stage, and one middle block of layer3 and layer4. "
                "Sanity check: 49.0% zero-shot top-1 on 500 val images of our 256px pre-resized copy with a single "
                "\"a photo of a {class}.\" prompt and raw WordNet names (OpenAI's 59.6% used curated names and 80 prompts).",
    layers=CLIP_LAYERS, modules={l: _clip_modules[l] for l in CLIP_LAYERS},
    pre_relu={l: "act3" if l == "stem" else f"{_clip_modules[l]}.act3" for l in CLIP_LAYERS},
    receptive_field={l: _clip_rf[l] for l in CLIP_LAYERS},
    load=_load_clip_visual, preprocess=_clip_preprocess, max_crop_px=128,
    groups={"stem": "stem", "layer1": "layer1", "layer2": "layer2", "layer3_2": "layer3",
            "layer3": "layer3", "layer4_1": "layer4", "layer4": "layer4"},
    notes={"stem": "3 conv 3x3 + 2x2 avgpool", "layer1": "block 3 of 3", "layer2": "block 4 of 4",
           "layer3_2": "block 3 of 6", "layer3": "block 6 of 6", "layer4_1": "block 2 of 3",
           "layer4": "block 3 of 3, before attention pooling"},
    architecture=dict(
        kind="residual",
        summary="Modified ResNet-50: 3-conv stem with avgpool, 4 stages of bottleneck blocks (3, 4, 6, 3) "
                "with residual connections and anti-aliased (avgpool) downsampling, then attention pooling "
                "into the 1024-d CLIP embedding (not shown).",
        stages=[dict(name="stem", layers=["stem"], note="conv 3x3 x3, avgpool 2x2, 64 channels")] +
               [dict(name=s, blocks=n, layers=[l for l in CLIP_LAYERS if l.split("_")[0] == s],
                     note=f"{n} bottleneck blocks, {256 * 2 ** i} channels")
                for i, (s, n, _) in enumerate(_RN50_STAGES)],
    ),
)

SPECS = {s.name: s for s in [INCEPTIONV1, VGG19, CLIP_RN50]}


def get_spec(name):
    try:
        return SPECS[name]
    except KeyError:
        raise SystemExit(f"--model must be one of {', '.join(SPECS)}") from None
