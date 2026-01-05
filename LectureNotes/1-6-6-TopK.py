import torch

def top_k_sampling(logits, k=50):
    # 过滤top-k
    topk_logits, topk_idx = logits.topk(k, dim=-1)
    
    # 采样
    probs = F.softmax(topk_logits, dim=-1)
    next_token_idx = torch.multinomial(probs, 1)
    return topk_idx.gather(-1, next_token_idx)