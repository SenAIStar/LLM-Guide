import torch.nn as nn

class TransformerEmbedding(nn.Module):
    """
    token embedding + positional encoding (sinusoid)
    """
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        """
        :param vocab_size: 词表大小
        :param d_model: 词嵌入维度（模型维度）
        """
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model) # 《嵌入》章节实现
        self.pos_emb = PositionalEncoding(d_model, max_len, device) # 《嵌入》章节实现
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)