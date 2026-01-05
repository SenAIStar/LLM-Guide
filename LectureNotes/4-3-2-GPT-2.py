# 位置编码
class PositionalEmbedding(nn.Embedding):
    def reset_parameters(self):
        # 使用正态分布初始化权重
        nn.init.normal_(self.weight, std=0.02)

    def _load_from_state_dict(self,
                              state_dict: Dict[str, torch.Tensor],
                              prefix: str,
                              *args,
                              **kwargs):
        weight = state_dict[f'{prefix}weight']

        # 调整位置嵌入矩阵的大小以适应序列长度的变化
        if weight.size(0) < self.num_embeddings:
            weight = torch.cat((weight, self.weight[weight.size(0):]), dim=0)
        elif weight.size(0) > self.num_embeddings:
            weight = weight[:self.num_embeddings]

        state_dict[f'{prefix}weight'] = weight
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        input: (..., seq_len)
        output: (..., seq_len, embedding_dim)
        """
        # 创建位置索引
        position = torch.arange(offset, offset + x.size(-1),
                                dtype=torch.long, device=x.device)
        position = position.view((1,) * (x.ndim - 1) + (-1,)).expand_as(x)

        return super().forward(position)
    
# 字符编码
class TokenEmbedding(nn.Embedding):
    def reset_parameters(self):
        # 使用正态分布初始化权重
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x: torch.Tensor, transposed: bool = False) -> torch.Tensor:
        """
        input: (..., seq_len) 或 (..., seq_len, embedding_dim)
        output: (..., seq_len, embedding_dim) 或 (..., seq_len, num_embeddings)
        """
        # 如果 transposed 为真，则进行矩阵乘法变换
        if transposed:
            return torch.matmul(x, self.weight.transpose(0, 1))
        else:
            return super().forward(x)
        
class BaseAttention(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        :param q: (..., query_len, dims)
        :param k: (..., kv_len, dims)
        :param v: (..., kv_len, dims)
        :param mask: (..., query_len, kv_len)
        :return: (..., query_len, dims)
        """
        # 计算注意力分数
        x = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))

        # 应用mask
        if mask is not None:
            x += mask.type_as(x) * x.new_tensor(-1e4)

        # 应用softmax并进行dropout
        x = self.dropout(x.softmax(-1))

        # 计算最终输出
        return torch.matmul(x, v)
    

class MultiHeadAttention(BaseAttention):
    def __init__(self, heads: int, dropout: float = 0.1):
        super().__init__(dropout)
        self.heads = heads

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        :param q: (..., query_len, dims)
        :param k: (..., kv_len, dims)
        :param v: (..., kv_len, dims)
        :param mask: (..., query_len, kv_len)
        :return: (..., query_len, dims)
        """
        # 将张量拆分为多个头
        q = q.view(q.size()[:-1] + (self.heads, q.size(-1) // self.heads))
        k = k.view(k.size()[:-1] + (self.heads, k.size(-1) // self.heads))
        v = v.view(v.size()[:-1] + (self.heads, v.size(-1) // self.heads))

        # 转置并调整维度
        q = q.transpose(-3, -2)
        k = k.transpose(-3, -2)
        v = v.transpose(-3, -2)

        if mask is not None:
            mask = mask.unsqueeze(-3)

        # 计算多头注意力并将其合并
        return (super().forward(q, k, v, mask)
                .transpose(-3, -2)
                .contiguous()
                .view(q.size()[:-3] + (q.size(-2), v.size(-1) * self.heads)))


class Attention(nn.Module):
    def __init__(self, heads: int, dims: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(heads, dropout)
        self.proj_q = nn.Linear(dims, dims)
        self.proj_k = nn.Linear(dims, dims)
        self.proj_v = nn.Linear(dims, dims)
        self.linear = nn.Linear(dims, dims)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                past: Optional[Past] = None, mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Past]:
        """
        :param q: float, 形状为 (..., query_len, dims)
        :param k: float, 形状为 (..., kv_len, dims)
        :param v: float, 形状为 (..., kv_len, dims)
        :param past (*): float, 形状为 (..., past_len, dims)
        :param mask: bool, 形状为 (..., query_len, past_len + kv_len)
        :return x: float, 形状为 (..., query_len, dims)
        :return (k, v) (*): float, 形状为 (..., past_len + kv_len, dims)
        """
        # 对q, k, v进行线性变换
        q, k, v = self.proj_q(q), self.proj_k(k), self.proj_v(v)

        # 如果有past，则拼接历史的k和v
        if past is not None:
            k = torch.cat((past[0], k), dim=-2)
            v = torch.cat((past[1], v), dim=-2)

        # 计算注意力并进行线性变换
        x = self.linear(self.attn(q, k, v, mask))
        return x, (k, v)
    
class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用Sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        返回Swish激活函数的输出
        :param x: float, 形状为 (..., dims)
        :return: float, 形状为 (..., dims)
        """
        return x * self.sigmoid(x)


class FFN(nn.Sequential):
    """
    input: float, 形状为 (..., dims)
    output: float, 形状为 (..., dims)
    """
    def __init__(self, dims: int, rate: int = 4, dropout: float = 0.1):
        super().__init__(
            # 线性变换，扩大维度
            nn.Linear(dims, dims * rate),
            # 使用Swish激活函数
            Swish(),
            # Dropout层
            nn.Dropout(dropout),
            # 线性变换，恢复原维度
            nn.Linear(dims * rate, dims))
        
# 用于生成填充掩码，标记输入张量中的填充部分
class PadMasking(nn.Module):
    def __init__(self, pad_idx: int):
        super().__init__()
        self.pad_idx = pad_idx  # 存储填充标记的索引

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        根据输入张量生成填充掩码。
        :param x (torch.Tensor): 形状为 (..., seq_len)，输入张量。
        :param offset (int, optional): 掩码的偏移量，默认值为0。
        :return: 填充掩码，形状为 (..., seq_len, seq_len + offset)。
        """
        # 创建一个标记填充位置的张量
        is_pad = (x == self.pad_idx).unsqueeze(-2)
        # 创建一个形状为 (seq_len, offset) 的全零张量
        shifted = torch.zeros(x.size()[:-1] + (1, offset,),
                              dtype=torch.bool, device=x.device)

        # 将填充位置与零张量拼接
        mask = torch.cat((shifted, is_pad), dim=-1)
        # 扩展掩码以适应输入张量的形状
        return mask.expand(x.shape + mask.shape[-1:])

# 用于生成未来掩码，标记输入张量中未来的部分，通常用于自回归任务。
class FutureMasking(nn.Module):
    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        根据输入张量生成未来掩码。
        :param x (torch.Tensor): 形状为 (..., seq_len)，输入张量。
        :param offset (int, optional): 掩码的偏移量，默认值为0。
        :return: 未来掩码，形状为 (..., seq_len, seq_len + offset)。
        """
        seq_len = x.size(-1)

        # 创建上三角矩阵来标记未来位置
        future = torch.ones((seq_len, seq_len + offset),
                            dtype=torch.bool, device=x.device)
        future = future.triu(offset + 1)  # 设置上三角部分为1，表示未来位置

        # 重新调整矩阵的形状，并扩展到输入张量的形状
        mask = future.view((1,) * (x.ndim - 1) + future.size())
        return mask.expand(x.shape + mask.shape[-1:])

class TransformerBlock(nn.Module):
    def __init__(self, heads: int, dims: int, rate: int, dropout: float = 0.1):
        super().__init__()
        # 初始化自注意力层、前馈网络层、层归一化层
        self.attn = Attention(heads, dims, dropout)
        self.ff = FFN(dims, rate, dropout)
        self.ln_attn = LayerNorm(dims)
        self.ln_ff = LayerNorm(dims)

    def forward(self, x: torch.Tensor, past: Optional[Past] = None,
                mask: Optional[torch.Tensor] = None,
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, Past]]:
        """
        前向传播，执行自注意力计算、前馈网络计算以及层归一化。
        :param x (torch.Tensor): 输入张量。
        :param past (Optional[Past], optional): 上一个状态。
        :param mask (Optional[torch.Tensor], optional): 掩码。
        :return: 当前层的输出，如果是训练模式则返回输出张量，否则返回输出和缓存状态。
        """
        # 在每个子层之前进行层归一化
        a = self.ln_attn(x)
        a, past = self.attn(a, a, a, past, mask)  # 执行自注意力

        x = x + a  # 残差连接
        x = x + self.ff(self.ln_ff(x))  # 前馈网络和残差连接

        return x if self.training else (x, past)
    

class GPT2(nn.Module):
    def __init__(self, layers: int, pad_idx: int, words: int, seq_len: int,
                 heads: int, dims: int, rate: int = 4, dropout: float = 0.1,
                 bidirectional: bool = True):
        super().__init__()
        self.bidirectional = bidirectional  # 是否使用双向掩码
        self.pad_masking = PadMasking(pad_idx)  # 初始化填充掩码
        self.future_masking = FutureMasking()  # 初始化未来掩码

        # 初始化位置嵌入、词嵌入、Dropout嵌入
        self.positional_embedding = PositionalEmbedding(seq_len, dims)
        self.token_embedding = TokenEmbedding(words, dims)
        self.dropout_embedding = nn.Dropout(dropout)

        # 初始化Transformer层的堆叠
        self.transformers = nn.ModuleList([
            TransformerBlock(heads, dims, rate, dropout)
            for _ in range(layers)])
        self.ln_head = LayerNorm(dims)  # 最后的层归一化

    def forward(self, x: torch.Tensor, past: Optional[List[Past]] = None,
                use_grad_ckpt: bool = False
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Past]]]:
        """
        前向传播，执行输入的嵌入、掩码、Transformer层计算。
        :param x (torch.Tensor): 输入张量。
        :param past (Optional[List[Past]], optional): 先前的状态，用于缓存。
        :param use_grad_ckpt (bool, optional): 是否使用梯度检查点，默认为False。
        :return: 输出张量（训练时）或输出和更新后的缓存状态（评估时）。
        """
        offset = past[0][0].size(-2) if past is not None else 0

        # 创建掩码张量
        mask = self.pad_masking(x, offset)
        if not self.bidirectional:
            mask = mask + self.future_masking(x, offset)

        # 使用词嵌入和位置嵌入层
        x = self.token_embedding(x) + self.positional_embedding(x, offset)
        x = self.dropout_embedding(x)

        # 顺序应用多个Transformer层
        present = []
        for i, transformer in enumerate(self.transformers):
            if self.training and use_grad_ckpt:
                transformer = partial(torch.utils.checkpoint.checkpoint, transformer)
            x = transformer(x, past[i] if past is not None else None, mask)

            if not self.training:
                present.append(x[1])
                x = x[0]

        x = self.ln_head(x)  # 最后一层归一化
        x = self.token_embedding(x, transposed=True)  # 转置后返回嵌入

        return x if self.training else (x, present)