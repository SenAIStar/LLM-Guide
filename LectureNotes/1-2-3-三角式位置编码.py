import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        """
        正弦位置编码的构造函数
        :param d_model: 模型的维度
        :param max_len: 最大序列长度
        :param device: CPU or GPU
        """
        super(PositionalEncoding, self).__init__()

        # 创建与输入矩阵相同大小的零矩阵（用于与输入矩阵相加）
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # 不需要计算梯度

        # 生成位置索引（从0到max_len-1）
        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 将1D的pos转换为2D，表示词的位置

        # 生成偶数索引（即d_model的每隔一个位置）
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i'是d_model的索引（例如embedding大小为50时，'i' = [0, 2, 4, ..., 48]）
        # "step=2"表示'i'乘以2（即每两个位置）

        # 计算正弦和余弦位置编码
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # 计算位置编码，以便考虑词的位置信息

    def forward(self, x):
        # self.encoding形状为[max_len = 512, d_model = 512]

        batch_size, seq_len = x.size()
        # 输入的形状为[batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
        # 输出形状为[seq_len = 30, d_model = 512]
        # 将位置编码与词嵌入相加，得到[128, 30, 512]