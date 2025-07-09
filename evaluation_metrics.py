import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
import numpy as np

class EvaluationMetrics:
    def __init__(self, device):
        self.device = device
        self.inception_model = inception_v3(
            pretrained=True, aux_logits=False, transform_input=False
        ).to(device)
        self.inception_model.eval()

        self.fid_model = inception_v3(
            pretrained=True, aux_logits=False, transform_input=False
        ).to(device)
        self.fid_model.fc = nn.Identity()
        self.fid_model.eval()

    def calculate_inception_score(self, images, num_splits=10):
        images = F.interpolate(images, size=(299, 299))
        preds = self.inception_model(images)
        scores = []
        num_images = preds.shape[0]
        for i in range(num_splits):
            start = i * num_images // num_splits
            end = start + num_images // num_splits

            part = preds[start:end]
            py = F.softmax(part).mean(dim=0, keepdim=True)
            scores.append(F.kl_div(F.log_softmax(part), py, reduction='batchmean').exp().item())
        return np.mean(scores), np.std(scores)

    def calculate_frechet_inception_distance(self, real_images, generated_images):
        real_images = F.interpolate(real_images, size=(299, 299), mode="bilinear", align_corners=False)
        generated_images = F.interpolate(generated_images, size=(299, 299), mode="bilinear", align_corners=False)

        with torch.no_grad():
            real_features = self.fid_model(real_images).view(real_images.size(0), -1)
            gen_features = self.fid_model(generated_images).view(generated_images.size(0), -1)

        real_features = real_features.cpu().numpy()
        gen_features = gen_features.cpu().numpy()

        mu1 = np.mean(real_features, axis=0)
        mu2 = np.mean(gen_features, axis=0)
        sigma1 = np.cov(real_features, rowvar=False)
        sigma2 = np.cov(gen_features, rowvar=False)

        covmean = sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        diff = mu1 - mu2
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return float(fid)
