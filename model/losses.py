import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectedODRLoss(nn.Module):
    def __init__(self, scales=(1, 2, 3), kernel_k=25):
        super().__init__()
        self.scales = list(scales)
        self.kernel_k = kernel_k
        self.epsilon = 1e-8

    def _compute_affinity_matrix(self, features: torch.Tensor) -> torch.Tensor:
        dist_sq = torch.cdist(features, features, p=2).pow(2)

        k = min(self.kernel_k, features.size(0))
        topk_dist, topk_idx = torch.topk(dist_sq, k=k, dim=1, largest=False)

        sigma = torch.mean(torch.sqrt(topk_dist + self.epsilon), dim=1, keepdim=True)
        sigma_mat = sigma @ sigma.t()

        knn_mask = torch.zeros_like(dist_sq, dtype=torch.bool)
        knn_mask.scatter_(1, topk_idx, True)
        mutual_mask = knn_mask & knn_mask.t()

        W = torch.exp(-dist_sq / (sigma_mat + self.epsilon))
        return W * mutual_mask.float()

    def forward(
        self, features: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        B = features.size(0)
        if B < 2:
            return features.new_tensor(0.0)

        W_feat = self._compute_affinity_matrix(features)

        labels = labels.view(-1, 1).float()
        direction_mask = (labels <= labels.t()).float()
        W = W_feat * direction_mask
        W = W + torch.eye(B, device=W.device, dtype=W.dtype)

        row_sum = W.sum(dim=1, keepdim=True)
        P = W / (row_sum + self.epsilon)

        s_diff = scores - scores.t()
        penalty_matrix = F.relu(s_diff)

        loss_odr: torch.Tensor = features.new_tensor(0.0)
        P_t = P
        max_scale = max(self.scales)

        for t in range(1, max_scale + 1):
            if t in self.scales:
                alpha_t = 1.0 / t
                loss_odr += alpha_t * ((P_t * penalty_matrix).sum(dim=1)).mean()
            if t < max_scale:
                P_t = P_t @ P

        return loss_odr
