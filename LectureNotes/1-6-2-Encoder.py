class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head) # 《注意力》章节实现
        self.norm1 = LayerNorm(d_model=d_model) # 《归一化》章节实现
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = FFN(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob) # 《前馈网络》章节实现
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask):
        # 计算 self attention
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)
        
        # add & norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        
        # FFN
        _x = x
        x = self.ffn(x)
      
        # add & norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x
        

class Encoder(nn.Module):

    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        max_len=max_len,
                                        vocab_size=enc_voc_size,
                                        drop_prob=drop_prob,
                                        device=device)

        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x, src_mask):
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x