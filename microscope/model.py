"""InceptionV1 (lucent port of TF inception5h) plus layer metadata."""
import torch

MODEL = "inceptionv1"

# Layers Microscope showed for InceptionV1, in forward order.
LAYERS = [
    "conv2d0", "conv2d1", "conv2d2",
    "mixed3a", "mixed3b",
    "mixed4a", "mixed4b", "mixed4c", "mixed4d", "mixed4e",
    "mixed5a", "mixed5b",
]

# The 10 mixed3b curve detectors from Distill's "Curve Detectors" (Cammarata et al. 2020).
CURVE_DETECTORS_3B = [379, 406, 385, 343, 342, 388, 340, 330, 349, 324]

# Theoretical receptive field in input pixels (stride / kernel arithmetic through
# the deepest branch: 5x5 conv inside each mixed block, 3x3/2 maxpools between).
RECEPTIVE_FIELD = {
    "conv2d0": 7, "conv2d1": 11, "conv2d2": 19,
    "mixed3a": 59, "mixed3b": 91,
    "mixed4a": 171, "mixed4b": 235, "mixed4c": 299, "mixed4d": 363, "mixed4e": 427,
    "mixed5a": 587, "mixed5b": 715,
}


def load_model(device, fix_lrn=True):
    from lucent.modelzoo import inceptionv1
    if fix_lrn:
        _fix_lucent_lrn()
    return inceptionv1(pretrained=True).to(device).eval()


def _fix_lucent_lrn():
    """lucent's InceptionV1 port passes TF's LRN alpha (1e-4) straight to
    F.local_response_norm, but PyTorch divides alpha by `size` and TF does not,
    so the LRN is 9x too weak. Mixed4a activations come out ~3x larger than
    Microscope's published values. Scaling alpha by size restores TF semantics:
    activations then match Microscope's channel_<u>.json within ~10%, with
    unchanged ImageNet accuracy. Patches only the `F` that module sees."""
    import sys
    import types
    import torch.nn.functional as F

    mod = sys.modules["lucent.modelzoo.inceptionv1.InceptionV1"]
    if getattr(mod.F, "_lrn_fixed", False):
        return
    proxy = types.SimpleNamespace(**{k: getattr(F, k) for k in dir(F) if not k.startswith("__")})
    proxy.local_response_norm = lambda x, size, alpha, beta=0.75, k=1.0: \
        F.local_response_norm(x, size, alpha * size, beta, k)
    proxy._lrn_fixed = True
    mod.F = proxy


def preprocess(x):
    """[0,1] RGB -> the [-117, 138] range the TF InceptionV1 graph expects.
    Same transform lucent's render_vis appends for InceptionV1."""
    return x * 255 - 117


class StopForward(Exception):
    pass


class LayerCapture:
    """Forward hooks that record the outputs of `layers` and abort the forward
    pass once the deepest one has run, so nothing past it is computed."""

    def __init__(self, model, layers, order=LAYERS, modules=None, use_input=False):
        # order / modules: the model's layer list and layer -> module path
        # (microscope.models.ModelSpec); defaults are InceptionV1's.
        # use_input: record each module's input instead of its output (pre-ReLU)
        self.layers = list(layers)
        self.use_input = use_input
        self.last = max(self.layers, key=list(order).index)
        self.out = {}
        get = (lambda n: getattr(model, n)) if modules is None else (lambda n: model.get_submodule(modules[n]))
        self.handles = [
            get(name).register_forward_hook(self._hook(name))
            for name in self.layers
        ]

    def _hook(self, name):
        def fn(module, inputs, output):
            self.out[name] = inputs[0] if self.use_input else output
            if name == self.last:
                raise StopForward
        return fn

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

