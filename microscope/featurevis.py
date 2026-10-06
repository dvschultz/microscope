"""Channel-objective feature visualization.

Mirrors lucent's `render_vis` defaults exactly (FFT + decorrelated-color
param.image, standard_transforms, InceptionV1 preprocessing, Adam lr=0.05,
512 steps) but replaces its hook_model, which registers a hook on every module
on every call and never removes them. Here only the target layer is hooked and
the forward pass stops there.
"""
import numpy as np
import torch

from .model import LayerCapture, preprocess


def render(model, layer, batch, loss_fn, size=224, steps=512, seed=0, spec=None, pre_relu=False):
    """Optimize `batch` images so loss_fn(acts) goes down; acts is the [B, C, H, W] output of `layer`.
    spec: a microscope.models.ModelSpec (default InceptionV1).
    pre_relu: capture the layer before its final ReLU (spec.pre_relu), for units that are zero
    on the random starting image and so get no gradient through the ReLU."""
    from lucent.optvis import param, transform

    torch.manual_seed(seed)
    np.random.seed(seed)
    params, image_f = param.image(size, batch=batch)
    optimizer = torch.optim.Adam(params, lr=5e-2)
    prep = preprocess if spec is None else spec.preprocess
    transform_f = transform.compose(list(transform.standard_transforms) + [prep])

    if spec is None:
        cap = LayerCapture(model, [layer])
    elif pre_relu:
        cap = LayerCapture(model, [layer], spec.layers, spec.pre_relu, use_input=True)
    else:
        cap = LayerCapture(model, [layer], spec.layers, spec.modules)
    try:
        for _ in range(steps):
            optimizer.zero_grad()
            loss = loss_fn(cap.run(model, transform_f(image_f()))[layer])
            loss.backward()
            optimizer.step()
    finally:
        cap.close()

    with torch.no_grad():
        img = image_f().detach().cpu().numpy()
    return (np.transpose(img, (0, 2, 3, 1)) * 255).round().astype(np.uint8)


def render_channels(model, layer, channels, size=224, steps=512, seed=0, spec=None, pre_relu=False):
    """Channel objective, lucent's objectives.channel: -mean over space of acts[i, c_i],
    summed over the batch (each image's loss only touches its own channel)."""
    idx, ch = torch.arange(len(channels)), torch.tensor(channels)
    return render(model, layer, len(channels), lambda acts: -acts[idx, ch].mean(dim=(1, 2)).sum(),
                  size, steps, seed, spec, pre_relu)
