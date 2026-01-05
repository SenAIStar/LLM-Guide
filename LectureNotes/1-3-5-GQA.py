class GroupQueryAttention(BaseAttention):
    def __init__(self, hidden_size, num_heads=8, num_kv_heads=4, dropout=None):
        super(GroupQueryAttention, self).__init__()
        assert hidden_size % num_heads == 0 and num_heads % num_kv_heads == 0

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_group = num_heads // num_kv_heads
        self.q_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size // num_heads * num_kv_heads)
        self.v_proj = torch.nn.Linear(hidden_size, hidden_size // num_heads * num_kv_heads)
        
        if dropout is not None:
            self.dropout = torch.nn.Dropout(p=dropout)

    def repeat_kv(self, feature, num_group):
        bs, num_kv_heads, seq_len, head_dims = feature.shape
        if num_group == 1:
            return feature
        feature = feature[:, :, None, :, :].expand(bs, num_kv_heads, num_group, seq_len, head_dims)
        return feature.reshape(bs, num_kv_heads * num_group, seq_len, head_dims)

    def forward(self, x, mask=None):
        bs, seq_len, hidden_size = x.shape

        q = self.q_proj(x).view(bs, seq_len, self.num_heads, -1).transpose(1, 2)
        k = self.k_proj(x).view(bs, seq_len, -1, hidden_size // self.num_heads).transpose(1, 2)
        v = self.v_proj(x).view(bs, seq_len, -1, hidden_size // self.num_heads).transpose(1, 2)
        k, v = self.repeat_kv(k, self.num_group), self.repeat_kv(v, self.num_group)

        output = super().forward(q, k, v, mask, self.dropout)
        output = output.view(bs, seq_len, hidden_size)
        return output