"""TopK sparse autoencoders on InceptionV1 layer activations.

Inputs are post-ReLU channel vectors at one spatial position, multiplied by a
fixed per-layer `scale` so the mean L2 norm over the training set is sqrt(d).
    z     = TopK(ReLU(W_enc^T (s*x - b_dec) + b_enc))
    x_hat = (W_dec^T z + b_dec) / s
Decoder rows are kept unit-norm, so a feature's code is its contribution's norm
(in scaled units).
"""
import numpy as np
import torch
import torch.nn as nn



class TopKSAE(nn.Module):
    def __init__(self, d, n, k):
        super().__init__()
        W = torch.randn(n, d)
        W /= W.norm(dim=1, keepdim=True)
        self.W_dec = nn.Parameter(W)                  # [n, d]
        self.W_enc = nn.Parameter(W.T.clone())        # [d, n]
        self.b_enc = nn.Parameter(torch.zeros(n))
        self.b_dec = nn.Parameter(torch.zeros(d))
        self.register_buffer("scale", torch.tensor(1.0))
        self.d, self.n, self.k = d, n, k

    def preact(self, x):
        """x already scaled. [N, d] -> [N, n]"""
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def topk(self, pre, k=None):
        v, i = pre.relu().topk(k or self.k, dim=-1)
        return torch.zeros_like(pre).scatter_(-1, i, v)

    def encode(self, x):
        return self.topk(self.preact(x))

    def decode(self, z):
        return z @ self.W_dec + self.b_dec

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec /= self.W_dec.norm(dim=1, keepdim=True)

    @torch.no_grad()
    def project_decoder_grad(self):
        """Remove the gradient component parallel to each (unit) decoder row."""
        g = self.W_dec.grad
        if g is not None:
            g -= (g * self.W_dec).sum(1, keepdim=True) * self.W_dec

    def save(self, path, **extra):
        torch.save(dict(state=self.state_dict(), d=self.d, n=self.n, k=self.k, **extra), path)

    @classmethod
    def load(cls, path, device="cpu"):
        ck = torch.load(path, map_location=device, weights_only=False)
        sae = cls(ck["d"], ck["n"], ck["k"]).to(device)
        sae.load_state_dict(ck["state"])
        return sae.eval(), ck


def render_features(model, layer, sae, feats, size=224, steps=512, seed=0):
    """Feature vis for SAE features, same recipe as featurevis.render_channels.

    Objective: spatial mean of the feature's encoder pre-activation at every
    position, w_f . (s*x - b_dec) + b_enc_f. The bias terms are constant in
    the image, so this is exactly the encoder direction; using the pre-ReLU
    value avoids a zero gradient when a feature is off everywhere at init.
    Each image's loss only touches its own feature."""
    from .featurevis import render
    W = (sae.W_enc[:, feats].T * sae.scale).detach()   # [B, C]
    return render(model, layer, len(feats),
                  lambda acts: -torch.einsum("bchw,bc->b", acts, W).sum() / (acts.shape[2] * acts.shape[3]),
                  size, steps, seed)
