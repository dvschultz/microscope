"""Pick a torch device and point lucent at it.

lucent hardcodes `cuda:0 if available else cpu` in a few places, so on Apple
Silicon it silently builds parameters on CPU. Patch those spots for MPS.
"""
import torch


def pick_device(name=None):
    if name:
        return _cap(torch.device(name))
    if torch.cuda.is_available():
        return _cap(torch.device("cuda:0"))
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _cap(device):
    """MICROSCOPE_GPU_MEM_GB=8 caps this process's CUDA allocations, so a job sharing a
    GPU with another one OOMs in this process instead of starving the other."""
    import os
    gb = os.environ.get("MICROSCOPE_GPU_MEM_GB")
    if gb and device.type == "cuda":
        idx = device.index or 0
        total = torch.cuda.get_device_properties(idx).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, float(gb) * 2 ** 30 / total), idx)
    return device


def patch_lucent(device):
    from lucent.optvis import transform
    from lucent.optvis.param import spatial, color

    transform.device = device
    spatial.device = device
    matrix = torch.tensor(color.color_correlation_normalized.T, dtype=torch.float32)

    on_device = {}

    def _linear_decorrelate_color(tensor):
        m = on_device.get(tensor.device)
        if m is None:
            m = on_device[tensor.device] = matrix.to(tensor.device)
        t = tensor.permute(0, 2, 3, 1)
        t = torch.matmul(t, m)
        return t.permute(0, 3, 1, 2)

    color._linear_decorrelate_color = _linear_decorrelate_color
