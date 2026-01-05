import torch
import torch.nn as nn


class DualChunkAttention(nn.Module):
    def __init__(self, embed_size, num_heads, chunk_size, adjacent_bias=0.5):
        super(DualChunkAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        # 相邻块注意力偏置
        self.adjacent_bias = adjacent_bias
        
        # 定义线性层
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        # 输出线性层
        self.out = nn.Linear(embed_size, embed_size)

    def split_into_chunks(self, x):
        # 切分输入x为多个块（chunk），每个块大小为chunk_size
        batch_size, seq_len, embed_size = x.shape
        num_chunks = seq_len // self.chunk_size
        chunks = x.view(batch_size, num_chunks, self.chunk_size, embed_size)
        return chunks

    def cross_block_attention(self, Q_chunks, K_chunks, V_chunks):
        # 跨块注意力计算（包含相邻注意力增强）
        batch_size, num_chunks, chunk_size, embed_size = Q_chunks.shape
        cross_attn_out = []
        
        for i in range(num_chunks):
            # 取出查询块
            q_chunk = Q_chunks[:, i, :, :]  # (batch_size, chunk_size, embed_size)
            
            # 计算该查询块与所有键块之间的注意力分数
            attn_scores = torch.matmul(q_chunk, K_chunks.transpose(2, 3)) / (self.embed_size ** 0.5)
            
            # 相邻注意力增强：对前一个和后一个块添加偏置
            if num_chunks > 1:
                if i > 0:  # 前相邻块
                    attn_scores[:, :, i-1] += self.adjacent_bias
                if i < num_chunks - 1:  # 后相邻块
                    attn_scores[:, :, i+1] += self.adjacent_bias
            
            # 注意力分数归一化
            # (batch_size, chunk_size, num_chunks)
            attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)  
            
            # 将注意力加权到值块上
            # (batch_size, chunk_size, embed_size)
            cross_attn_out.append(torch.matmul(attn_probs, V_chunks[:, i, :, :]))

        # 拼接所有块之间的跨块注意力输出
        # (batch_size, num_chunks * chunk_size, embed_size)
        cross_attn_out = torch.cat(cross_attn_out, dim=1)
        return cross_attn_out

    def forward(self, x):
        batch_size, seq_len, embed_size = x.shape

        # 获取查询、键和值的表示
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # 将Q, K, V分块
        Q_chunks = self.split_into_chunks(Q)
        K_chunks = self.split_into_chunks(K)
        V_chunks = self.split_into_chunks(V)

        # 1. 块内自注意力计算：捕捉块内元素关系
        attn_out = []
        for q_chunk, k_chunk, v_chunk in zip(Q_chunks, K_chunks, V_chunks):
            # 计算每个块内的注意力分数
            attn_scores = torch.matmul(q_chunk, k_chunk.transpose(-1, -2)) / (self.embed_size ** 0.5)
            attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)
            attn_out.append(torch.matmul(attn_probs, v_chunk))

        # 拼接块内注意力结果
        attn_out = torch.cat(attn_out, dim=2)  # (batch_size, seq_len, embed_size)

        # 2. 跨块注意力计算（含相邻注意力）：捕捉块间关系，重点增强相邻块关联
        cross_attn_out = self.cross_block_attention(Q_chunks, K_chunks, V_chunks)

        # 3. 融合注意力计算：结合块内和跨块（含相邻）注意力结果
        combined_out = attn_out + cross_attn_out  # 可根据需求改为加权融合，这里为相加
        # 通过输出层得到最终结果
        out = self.out(combined_out)
        return out
