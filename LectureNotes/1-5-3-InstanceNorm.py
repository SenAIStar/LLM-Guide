import torch
import troch.nn as nn

class InstanceNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(InstanceNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x):
        # Compute mean and variance over the spatial dimensions (per instance)
        mean = x.mean([2, 3], keepdim=True)  # Assuming input is (batch, channels, height, width)
        var = x.var([2, 3], unbiased=False, keepdim=True)

        # Normalize
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma.view(1, -1, 1, 1) * x_hat + self.beta.view(1, -1, 1, 1)
        return out