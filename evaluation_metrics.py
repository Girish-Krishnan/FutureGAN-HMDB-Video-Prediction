import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3
from typing import Tuple
import numpy as np

__all__ = [
    "InceptionScore",
    "FrechetInceptionDistance",
]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _resize(images: torch.Tensor) -> torch.Tensor:
    """Force every batch to 299 × 299 before feeding Inception."""
    return F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)


def _to_double(x: torch.Tensor) -> torch.Tensor:
    """Float32 on CPU avoids rounding trouble during covariance and eigen operations."""
    return x.to(dtype=torch.float32, device="cpu")


def _cov(features: torch.Tensor) -> torch.Tensor:
    n = features.size(0)
    mean = features.mean(dim=0, keepdim=True)
    centered = features - mean
    return centered.t().mm(centered) / (n - 1)


def _sqrtm_psd(mat: torch.Tensor) -> torch.Tensor:
    """Square root for a symmetric positive matrix via eigen fallback."""
    eigvals, eigvecs = torch.linalg.eigh(mat)
    eigvals = torch.clamp(eigvals, min=0)
    return eigvecs @ torch.diag(eigvals.sqrt()) @ eigvecs.t()


def _sqrtm_product(sig1: torch.Tensor, sig2: torch.Tensor) -> torch.Tensor:
    """Symmetric square root of sig1 × sig2 using the approach in the original FID paper."""
    s1_sqrt = _sqrtm_psd(sig1)
    middle = s1_sqrt @ sig2 @ s1_sqrt
    return _sqrtm_psd(middle)

# -----------------------------------------------------------------------------
# Inception score
# -----------------------------------------------------------------------------

class InceptionScore:
    def __init__(self, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.model = inception_v3(pretrained=True).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, images: torch.Tensor, splits: int = 10) -> Tuple[float, float]:
        images = _resize(images.to(self.device))
        logits = self.model(images)
        probs = F.softmax(logits, dim=1)

        chunk = probs.size(0) // splits if probs.size(0) >= splits else probs.size(0)
        scores = []
        for i in range(0, probs.size(0), chunk):
            part = probs[i : i + chunk]
            py = part.mean(dim=0, keepdim=True)
            kl = (part * (torch.log(part + 1e-16) - torch.log(py + 1e-16))).sum(dim=1).mean()
            scores.append(torch.exp(kl).item())

        s = torch.tensor(scores)
        return float(s.mean()), float(s.std())

# -----------------------------------------------------------------------------
# Frechet Inception Distance
# -----------------------------------------------------------------------------

class FrechetInceptionDistance:
    def __init__(self, device: str | torch.device = "cpu", eps: float = 1e-6) -> None:
        self.device = torch.device(device)
        self.eps = eps
        self.feature_model = inception_v3(pretrained=True)
        self.feature_model.fc = nn.Identity()
        self.feature_model.to(self.device).eval()

    @torch.no_grad()
    def _features(self, images: torch.Tensor) -> torch.Tensor:
        images = _resize(images.to(self.device))
        feats = self.feature_model(images).view(images.size(0), -1)
        return _to_double(feats)

    @torch.no_grad()
    def __call__(self, real: torch.Tensor, fake: torch.Tensor) -> float:
        real_feat = self._features(real)
        fake_feat = self._features(fake)

        mu_r = real_feat.mean(dim=0)
        mu_f = fake_feat.mean(dim=0)

        sig_r = _cov(real_feat) + torch.eye(real_feat.size(1)) * self.eps
        sig_f = _cov(fake_feat) + torch.eye(fake_feat.size(1)) * self.eps

        cov_sqrt = _sqrtm_product(sig_r, sig_f)

        diff = mu_r - mu_f
        fid_val = diff.dot(diff) + torch.trace(sig_r + sig_f - 2 * cov_sqrt)
        return float(max(fid_val.item(), 0.0))
