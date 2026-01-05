import math
from collections import defaultdict, Counter

def initialize_vocab(text, initial_size=1000):
    """初始化词汇表"""
    vocab = Counter()
    for word in text:
        for i in range(len(word)):
            for j in range(i+1, len(word)+1):
                sub = word[i:j]
                vocab[sub] += 1
    return dict(vocab.most_common(initial_size))

def compute_likelihood(text, vocab_probs):
    """计算数据似然值"""
    total_log_likelihood = 0
    for word in text:
        probs = [vocab_probs.get(word[i:j], 1e-10) 
                for i in range(len(word)) for j in range(i+1, len(word)+1)]
        if probs:
            total_log_likelihood += math.log(sum(probs))
    return total_log_likelihood

def train_ulm(text, vocab_size, num_iterations=10):
    """ULM训练主函数"""
    # 1. 初始化词汇表
    vocab = initialize_vocab(text, vocab_size)
    
    for _ in range(num_iterations):
        # 2. 计算当前子词概率
        total_count = sum(vocab.values())
        vocab_probs = {sub: count/total_count for sub, count in vocab.items()}
        
        # 3. 计算每个子词的损失变化
        losses = defaultdict(float)
        for word in text:
            # 寻找最优分割（Viterbi算法简化版）
            best_score = [0] * (len(word)+1)
            best_path = [""] * (len(word)+1)
            
            for end in range(1, len(word)+1):
                best_score[end] = float('-inf')
                for start in range(end):
                    sub = word[start:end]
                    if sub in vocab_probs:
                        score = best_score[start] + math.log(vocab_probs[sub])
                        if score > best_score[end]:
                            best_score[end] = score
                            best_path[end] = sub
            
            # 回溯记录最优分割
            pos = len(word)
            while pos > 0:
                sub = best_path[pos]
                losses[sub] += 1
                pos -= len(sub)
        
        # 4. 更新词汇表（保留损失最小的子词）
        vocab = dict(Counter(losses).most_common(vocab_size))
    
    return vocab

# 示例用法
text = ["low", "lower", "newest", "widest"]
vocab_size = 50
ulm_vocab = train_ulm(text, vocab_size)
print("ULM Vocabulary:", list(ulm_vocab.keys())[:10])