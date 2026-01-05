import re
from collections import defaultdict, Counter

def bytes_to_unicode():
    """
    将字节(0-255)映射到可打印Unicode字符
    处理不可打印字符，类似GPT-2的方法
    """
    # 可打印ASCII字符
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]  # 复制一份
    
    # 处理不可打印字节
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    
    # 创建映射表
    byte_to_char = {b: chr(c) for b, c in zip(bs, cs)}
    return byte_to_char

class BBPE:
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.byte_to_char = bytes_to_unicode()
        self.char_to_byte = {v: k for k, v in self.byte_to_char.items()}
        self.vocab = None
        self.merges = None
    
    def _preprocess(self, text):
        """将文本转换为字节序列并映射到字符"""
        # 将文本编码为UTF-8字节
        byte_sequence = text.encode('utf-8')
        # 将每个字节映射到可打印字符
        return ''.join(self.byte_to_char[b] for b in byte_sequence)
    
    def _deprocess(self, char_sequence):
        """将字符序列转换回字节序列"""
        # 将字符映射回字节
        byte_sequence = bytes(self.char_to_byte[c] for c in char_sequence)
        # 解码为UTF-8字符串
        return byte_sequence.decode('utf-8', errors='replace')
    
    def train(self, corpus):
        """训练BBPE模型"""
        # 预处理文本
        processed_corpus = [self._preprocess(text) for text in corpus]
        
        # 初始词汇表: 所有单个字符
        vocab = set()
        for text in processed_corpus:
            vocab.update(text)
        vocab = list(vocab)
        
        # 初始词表: 每个字符作为一个token
        word_freqs = defaultdict(int)
        for text in processed_corpus:
            tokens = list(text)  # 初始为单个字符
            word_freqs[' '.join(tokens)] += 1
        
        merges = {}
        num_merges = self.vocab_size - len(vocab)
        
        # BPE合并
        for i in range(num_merges):
            # 统计符号对频率
            pair_freqs = defaultdict(int)
            for word, freq in word_freqs.items():
                symbols = word.split()
                for j in range(len(symbols)-1):
                    pair = (symbols[j], symbols[j+1])
                    pair_freqs[pair] += freq
            
            if not pair_freqs:
                break
            
            # 选择最常见的一对
            best_pair = max(pair_freqs, key=pair_freqs.get)
            new_token = ''.join(best_pair)
            
            # 记录合并操作
            merges[best_pair] = new_token
            vocab.append(new_token)
            
            # 合并词表中的符号对
            new_word_freqs = {}
            for word, freq in word_freqs.items():
                new_word = word
                # 替换所有出现的符号对
                new_word = new_word.replace(' '.join(best_pair), new_token)
                new_word_freqs[new_word] = freq
            
            word_freqs = new_word_freqs
        
        self.vocab = vocab
        self.merges = merges
        return vocab, merges
    
    def encode(self, text):
        """编码文本为BBPE tokens"""
        if not self.vocab:
            raise ValueError("Model not trained yet")
        
        # 预处理文本
        processed_text = self._preprocess(text)
        tokens = list(processed_text)  # 初始化为单个字符
        
        # 应用所有合并操作
        for pair, new_token in self.merges.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                # 检查是否可以合并
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        return tokens
    
    def decode(self, tokens):
        """解码BBPE tokens为文本"""
        # 连接所有token
        char_sequence = ''.join(tokens)
        # 转换回原始文本
        return self._deprocess(char_sequence)

# 示例用法
if __name__ == "__main__":
    # 训练数据
    corpus = [
        "Hello world!",
        "BBPE handles all languages.",
        "多语言支持: 中文, 日本語, 한국어, русский",
        "Emoji: 😊👍🌟",
        "Special chars: \t\n\\"
    ]
    
    # 训练BBPE
    bbpe = BBPE(vocab_size=500)
    vocab, merges = bbpe.train(corpus)
    print(f"Vocab size: {len(vocab)}")
    print(f"Merges: {list(merges.items())[:5]}")
    
    # 编码和解码测试
    test_text = "BBPE可以处理任何语言和符号: 日本語ができます！😊"
    print("\nOriginal text:", test_text)
    
    encoded = bbpe.encode(test_text)
    print("\nEncoded tokens:", encoded)
    
    decoded = bbpe.decode(encoded)
    print("\nDecoded text:", decoded)
    
    # 检查是否完美重建
    assert decoded == test_text, "Decoding failed!"
    print("\nPerfect reconstruction!")