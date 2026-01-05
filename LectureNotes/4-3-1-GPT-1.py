class SelfAttention(Module):
    def __init__(self, d_in: int, d_out: int, device: str):
        """ 初始化自注意力层

        参数：
            d_in: 输入维度
            d_out: 输出维度
            device: 设备类型（CPU/GPU）
        """
        super().__init__()
        # 定义查询、键、值的线性变换
        self.wq = Linear(d_in, d_out).to(device)
        self.wk = Linear(d_in, d_out).to(device)
        self.wv = Linear(d_in, d_out).to(device)
        self.scale = sqrt(Tensor([d_out])).to(device)  # 缩放因子
        self.softmax = Softmax(dim=2).to(device)

        # 初始化权重
        self.init_weights()

    def init_weights(self):
        """ 初始化各个线性层的权重 """
        normal_(self.wq.weight, mean=0.0, std=0.02)
        normal_(self.wk.weight, mean=0.0, std=0.02)
        normal_(self.wv.weight, mean=0.0, std=0.02)
        zeros_(self.wq.bias)
        zeros_(self.wk.bias)
        zeros_(self.wv.bias)

    def forward(self, x, ignore):
        """ 前向传播 """
        # 计算查询、键、值
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # 计算注意力分数
        att = einsum('bqd,bkd->bqk', q, k)
        att /= self.scale

        # 创建掩码，防止未来位置的影响
        causal_mask = tril(ones(att.shape)).to(self.device)
        ignore_mask = self.get_ignore_mask(att, ignore)
        mask = causal_mask * ignore_mask

        # 计算加权平均
        att = att.masked_fill(mask==0, float('-inf'))
        att = self.softmax(att)
        att = einsum('bqk,bkd->bqd', att, v)

        return att

    def get_ignore_mask(self, att: Tensor, ignore: Tensor) -> Tensor:
        """ 获取忽略的掩码

        参数：
            att: 注意力矩阵
            ignore: 要忽略的输入位置
        返回：
            (Tensor): 忽略位置的掩码
        """
        ignore_mask = ones(att.shape).to(self.device)
        ignore_mask = einsum(
            'bqk,bk->bqk', 
            ignore_mask, 
            (ignore==0).float()
        )
        ignore_mask += triu(tril(ones(att.shape))).to(self.device)
        ignore_mask[(ignore_mask==2)] = 1

        return ignore_mask
    
class MultiHeadAttention(Module):
    def __init__(self, n_heads: int, dim: int, device: str):
        """ 初始化多头自注意力层

        参数：
            n_heads: 注意力头数
            dim: 输入/输出维度
            device: 设备类型（CPU/GPU）
        """
        super().__init__()
        # 每个头使用独立的自注意力层
        self.heads = ModuleList([
            SelfAttention(dim, dim // n_heads, device) 
            for i in range(n_heads)
        ])
        self.wo = Linear(dim, dim).to(device)  # 最终输出线性变换

        # 初始化权重
        self.init_weights()

    def init_weights(self):
        """ 初始化输出权重 """
        normal_(self.wo.weight, mean=0.0, std=0.02)
        zeros_(self.wo.bias)

    def forward(self, x, ignore):
        """ 前向传播 """
        # 将每个头的输出进行拼接并通过线性变换
        out = cat([head(x, ignore) for head in self.heads], dim=2)
        return self.wo(out)
    
class FFN(Module):
    def __init__(self, d_in: int, d_h: int, device: str):
        """ 初始化前馈层

        参数：
            d_in: 输入单元数
            d_h: 隐藏单元数
            device: 设备类型（CPU/GPU）
        """
        super().__init__()
        # 定义两个线性变换
        self.l1 = Linear(d_in, d_h).to(device)
        self.l2 = Linear(d_h, d_in).to(device)
        self.gelu = GELU().to(device)  # 激活函数

        # 初始化权重
        self.init_weights()

    def init_weights(self) -> None:
        """ 初始化前馈层权重 """
        normal_(self.l1.weight, mean=0.0, std=0.02)
        normal_(self.l2.weight, mean=0.0, std=0.02)
        zeros_(self.l1.bias)
        zeros_(self.l2.bias)

    def forward(self, x: Tensor) -> Tensor:
        """ 前向传播 """
        out = self.l1(x)
        out = self.l2(out)
        return self.gelu(out)
    
class TransformerBlock(Module):
    def __init__(self, n_heads: int, dim: int, hidden: int, dropout: float,
                 device: str):
        """ 初始化 Transformer 块

        参数：
            n_heads: 注意力头数
            dim: 输入/输出维度
            hidden: 前馈层的隐藏单元数
            dropout: dropout 比例
            device: 设备类型（CPU/GPU）
        """
        super().__init__()
        # 初始化各层
        self.att = MultiHeadAttention(n_heads, dim, device)
        self.ffl = FFN(dim, hidden, device)
        self.norm1 = LayerNorm(dim).to(device)
        self.norm2 = LayerNorm(dim).to(device)
        self.drop1 = Dropout(dropout).to(device)
        self.drop2 = Dropout(dropout).to(device)

        # 初始化权重
        self.init_weights()

    def init_weights(self):
        """ 初始化层的权重 """
        ones_(self.norm1.weight)
        ones_(self.norm2.weight)
        zeros_(self.norm1.bias)
        zeros_(self.norm2.bias)

    def forward(self, x, ignore):
        """ 前向传播 """
        # 先通过注意力层再进行 LayerNorm 和 Dropout
        out = self.norm1(x + self.drop1(self.att(x, ignore)))
        # 然后通过前馈层并进行第二次 LayerNorm 和 Dropout
        return self.norm2(out + self.drop2(self.ffl(out)))
    
class GPT1(Module):
    def __init__(self, vocab: int, seq: int, n_layers: int, n_heads: int, 
                 dim: int, hidden: int, dropout: float, device: str):
        """ 初始化 GPT-1 模型

        参数：
            vocab: 词汇表大小
            seq: 序列长度（窗口大小）
            n_layers: Transformer 层数
            n_heads: 每层的多头注意力头数
            dim: 嵌入维度
            hidden: 前馈层的隐藏单元数
            dropout: dropout 比例
            device: 设备类型（CPU/GPU）
        """
        super().__init__()
        
        # 初始化 BPE 和位置嵌入
        self.bpe_embed = Embedding(vocab, dim).to(device)
        self.pos_embed = Embedding(seq, dim).to(device)
        self.pos = LongTensor([i for i in range(128)]).to(device)  # 固定位置编码

        # 初始化 Transformer 块（多个堆叠的层）
        self.blocks = ModuleList([
            TransformerBlock(n_heads, dim, hidden, dropout, device) 
            for i in range(n_layers)
        ])

        # 输出层
        self.output = Linear(dim, vocab).to(device)
        self.drop = Dropout(dropout).to(device)  # Dropout 层

        # 初始化权重
        self.init_weights()

    def init_weights(self):
        """ 初始化各层权重 """
        normal_(self.bpe_embed.weight, mean=0.0, std=0.02)
        normal_(self.pos_embed.weight, mean=0.0, std=0.02)
        normal_(self.output.weight, mean=0.0, std=0.02)
        zeros_(self.output.bias)

    def forward(self, x, ignore):
        """ 前向传播 """
        # 获取嵌入表示
        be = self.bpe_embed(x)
        pe = self.pos_embed(self.pos)

        # 加和 BPE 嵌入和位置编码，经过 Dropout
        out = self.drop(be + pe)
        
        # 通过每一层 Transformer
        for block in self.blocks:
            out = block(out, ignore)

        return self.output(out)

    def get_parameters(self) -> List[Dict]:
        """ 获取模型参数及其权重衰减参数 """
        params = [
            { 'params': [], 'weight_decay': 1e-2 },
            { 'params': [], 'weight_decay': 0.00 },
        ]

        # 将不同的参数分配到不同的权重衰减组
        for name, parameter in self.named_parameters():
            if ('att' in name or 'ffl' in name or 'output' in name) and \
               name.endswith('weight'):
                params[0]['params'].append(parameter)
            else:
                params[1]['params'].append(parameter)
            
        return params