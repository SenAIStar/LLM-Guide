import torch
import torch.nn as nn

class GroupNorm(nn.Module):
    def __init__(self, num_groups, num_features, eps=1e-5):
        super(GroupNorm, self).__init__()
        self.num_groups = num_groups
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x):
        # Reshape input into groups
        N, C, H, W = x.size()
        group_size = C // self.num_groups
        x = x.view(N, self.num_groups, group_size, H, W)

        # Compute mean and variance over the groups
        mean = x.mean([2, 3, 4], keepdim=True)
        var = x.var([2, 3, 4], unbiased=False, keepdim=True)

        # Normalize
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        x_hat = x_hat.view(N, C, H, W)  # Reshape back
        out = self.gamma.view(1, -1, 1, 1) * x_hat + self.beta.view(1, -1, 1, 1)
        return out