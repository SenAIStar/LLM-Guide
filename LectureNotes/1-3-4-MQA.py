import torch

class MultiQueryAttention(BaseAttention):
    def __init__(self, hidden_size, num_heads=8, dropout=None):
        super(MultiQueryAttention, self).__init__()
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0
        self.q_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size // num_heads)
        self.v_proj = torch.nn.Linear(hidden_size, hidden_size // num_heads)
        
        if dropout is not None:
            self.dropout = torch.nn.Dropout(p=dropout)
    
    def forward(self, x, mask=None):
        bs, seq_len, hidden_size = x.shape

        q = self.q_proj(x).view(bs, seq_len, self.num_heads, -1).transpose(1, 2)
        k = self.k_proj(x).view(bs, seq_len, -1, hidden_size // self.num_heads).transpose(1, 2)
        v = self.v_proj(x).view(bs, seq_len, -1, hidden_size // self.num_heads).transpose(1, 2)

        output = super().forward(q, k, v, mask, self.dropout)
        output = output.view(bs, seq_len, hidden_size)
        return output