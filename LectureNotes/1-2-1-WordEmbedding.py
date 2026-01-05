import torch.nn as nn

class TokenEmbedding(nn.Embedding):
    """
    Token Embedding 实现，继承 nn.Embedding
    """
    def __init__(self, vocab_size, d_model):
        """
        根据词表规模初始化
        :param vocab_size: 词表大小
        :param d_model: 词嵌入维度（模型维度）
        """
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)