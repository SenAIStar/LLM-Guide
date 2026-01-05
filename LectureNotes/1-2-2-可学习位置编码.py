import torch
import torch.nn as nn

class PositionEncoding(nn.Module):
    def __init__(self, max_len, embed_dim):
        """
        初始化可学习的位置编码层。
        :param max_len: 序列的最大长度
        :param embed_dim: 位置编码的维度
        """
        super(PositionEncoding, self).__init__()
        # 创建一个位置编码的嵌入层
        self.position_embedding = nn.Embedding(max_len, embed_dim)
    
    def forward(self, x):
        """
        前向传播，给定输入x，返回学习到的位置编码。    
        :param x: 输入的tensor，通常形状为 (batch_size, seq_len, embed_dim)
        :return: 输入的x加上位置编码后的tensor
        """
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)  # 形状 (1, seq_len)
        position_encodings = self.position_embedding(positions)  # 获取位置编码，形状 (1, seq_len, embed_dim)
        
        return x + position_encodings  # 将位置编码加到输入的x上