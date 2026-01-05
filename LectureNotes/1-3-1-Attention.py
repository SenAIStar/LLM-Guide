import math
import torch

class BaseAttention(nn.Module):
    def __init__(self):
        super(BaseAttention, self).__init__()
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, dropout=None):
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])

        if mask is not None:
            attn = attn + mask
        
        attn = self.softmax(attn)
        if dropout is not None:
            attn = dropout(attn)
        output = torch.matmul(attn, v)
        return output