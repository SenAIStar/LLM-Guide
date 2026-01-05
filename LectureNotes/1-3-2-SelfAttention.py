import torch

class SelfAttention(BaseAttention):
    def __init__(self, hidden_size, dropout=None):
        super(SelfAttention, self).__init__()
        self.q_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.v_proj = torch.nn.Linear(hidden_size, hidden_size)
        
        if dropout is not None:
            self.dropout = torch.nn.Dropout(p=dropout)
        else:
            self.dropout = None
    
    def forward(self, x, mask=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        output = super().forward(q, k, v, mask, self.dropout)
        return output


# 测试数据
input_data = torch.randn(1, 20, 4096)

model = SelfAttention(hidden_size=4096, dropout=0.1)
output = model(input_data)
print(output.shape)