import torch
import torch.nn.functional as F
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
import numpy as np

class EvaluationMetrics:
    def __init__(self, device):
        self.device = device
        self.inception_model = inception_v3(pretrained=True).to(device)
        self.inception_model.eval()

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
        real_images = F.interpolate(real_images, size=(299, 299))
        generated_images = F.interpolate(generated_images, size=(299, 299))

        real_preds = self.inception_model(real_images).detach().cpu().numpy()
        gen_preds = self.inception_model(generated_images).detach().cpu().numpy()

        mu1, sigma1 = real_preds.mean(axis=0), np.cov(real_preds, rowvar=False)
        mu2, sigma2 = gen_preds.mean(axis=0), np.cov(gen_preds, rowvar=False)

        ssdiff = np.sum((mu1 - mu2)**2.0)
        covmean = sqrtm(sigma1.dot(sigma2))

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid
