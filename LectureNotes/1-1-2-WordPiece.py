import re
from collections import Counter

def compute_pair_scores(vocab):
    """计算所有可能符号对的得分"""
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[(symbols[i], symbols[i+1])] += freq
    
    scores = {}
    for pair, count in pairs.items():
        score = count / (vocab.get(pair[0], 1e-6) * vocab.get(pair[1], 1e-6))
        scores[pair] = score
    return scores

def merge_wordpiece(pair, vocab):
    """合并得分最高的符号对"""
    new_vocab = {}
    pattern = re.compile(r'(?<!\S)' + re.escape(' '.join(pair)) + r'(?!\S)')
    replacement = ''.join(pair)
    
    for word, freq in vocab.items():
        new_word = pattern.sub(replacement, word)
        new_vocab[new_word] = freq
    
    return new_vocab

def train_wordpiece(vocab, num_merges):
    """WordPiece训练主函数"""
    for _ in range(num_merges):
        scores = compute_pair_scores(vocab)
        if not scores:
            break
        best_pair = max(scores, key=scores.get)
        vocab = merge_wordpiece(best_pair, vocab)
    return vocab

# 示例用法
vocab = {
    'l o w </w>': 5,
    'l o w e r </w>': 2,
    'n e w e s t </w>': 6,
    'w i d e s t </w>': 3
}

final_vocab = train_wordpiece(vocab, 10)
print("WordPiece Vocabulary:", final_vocab)