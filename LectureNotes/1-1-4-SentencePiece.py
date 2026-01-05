import re
import math
from collections import defaultdict, Counter
import sys


class SentencePieceTrainer:
    def __init__(self, vocab_size=8000):
        self.vocab_size = vocab_size
        self.vocab = None
        self.subword_probs = None
        self.base_chars = set()  # 存储所有基础字符
        self.min_prob = 1e-10  # 最小概率值，防止对数计算错误

    def _initialize_vocab(self, text):
        """初始化词汇表为所有字符和常见子串"""
        char_vocab = set()
        for s in text:
            char_vocab.update(s)
            self.base_chars.update(s)  # 保存所有基础字符

        # 添加常见双字符组合
        bigrams = Counter()
        for s in text:
            for i in range(len(s) - 1):
                bigrams[s[i:i + 2]] += 1

        # 合并初始词汇表
        vocab = set(char_vocab)
        vocab.update([bg for bg, cnt in bigrams.most_common(self.vocab_size // 2)])
        return list(vocab)

    def _safe_log(self, prob):
        """安全的对数计算，防止数学域错误"""
        return math.log(max(prob, self.min_prob))

    def _compute_subword_probs(self, text, vocab):
        """EM算法估计子词概率 - 修复版本"""
        subword_counts = defaultdict(float)
        total_count = 0

        for sentence in text:
            n = len(sentence)
            # 使用动态规划寻找最佳分割
            best_score = [-10 ** 15] * (n + 1)  # 初始化为非常小的数
            best_path = [None] * (n + 1)  # 存储回溯路径
            best_score[0] = 0  # 起始位置分数为0

            # 前向传递：计算最佳分数
            for end in range(1, n + 1):
                # 确保至少处理单个字符
                found = False
                # 限制子词最大长度为5，同时从尽可能远的地方开始（但不超过5）
                for start in range(max(0, end - 5), end):
                    subword = sentence[start:end]
                    if subword in vocab:
                        # 使用安全的对数概率防止下溢
                        prob = self.subword_probs.get(subword, self.min_prob)
                        current_score = best_score[start] + self._safe_log(prob)
                        if current_score > best_score[end]:
                            best_score[end] = current_score
                            best_path[end] = (start, subword)
                            found = True

                # 如果没有找到子词，使用单个字符作为回退
                if not found and end > 0:
                    start = end - 1
                    subword = sentence[start:end]
                    # 使用最小概率的对数
                    current_score = best_score[start] + self._safe_log(self.min_prob)
                    best_score[end] = current_score
                    best_path[end] = (start, subword)

            # 回溯收集子词计数
            pos = n
            segments = []
            while pos > 0:
                if best_path[pos] is None:
                    # 处理无法分割的情况：回退到单个字符
                    start = pos - 1
                    subword = sentence[start:pos]
                    pos = start
                else:
                    start, subword = best_path[pos]
                    pos = start

                segments.append(subword)
                subword_counts[subword] += 1
                total_count += 1

        # 更新概率估计
        new_probs = {}
        for subword in vocab:
            count = subword_counts.get(subword, 0)
            if total_count > 0:
                prob = count / total_count
            else:
                prob = self.min_prob
            # 确保概率不为零
            new_probs[subword] = max(prob, self.min_prob)

        return new_probs, subword_counts

    def train(self, text, num_iterations=10):
        """训练SentencePiece模型 - 修复版本"""
        # 过滤空字符串
        text = [s for s in text if s.strip()]
        if not text:
            raise ValueError("训练文本为空")

        # 初始化词汇表和概率
        self.vocab = self._initialize_vocab(text)
        self.subword_probs = {subword: 1.0 / len(self.vocab) for subword in self.vocab}

        # EM训练
        for iter in range(num_iterations):
            print(f"Iteration {iter + 1}/{num_iterations} - 计算子词概率...")
            self.subword_probs, subword_counts = self._compute_subword_probs(text, set(self.vocab))

            # 保留概率最高的子词，但确保基础字符被包含
            sorted_subwords = sorted(
                subword_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )

            # 确保基础字符始终在词汇表中
            new_vocab = set(self.base_chars)
            for subword, cnt in sorted_subwords:
                if len(new_vocab) >= self.vocab_size:
                    break
                new_vocab.add(subword)

            self.vocab = list(new_vocab)
            print(
                f"  Vocab size: {len(self.vocab)}, Top subword: {sorted_subwords[0][0] if sorted_subwords else 'None'}")

        # 添加特殊标记
        special_tokens = ['<unk>', '<s>', '</s>', '<pad>']
        self.vocab = special_tokens + [v for v in self.vocab if v not in special_tokens][
                                      :self.vocab_size - len(special_tokens)]

        # 更新概率字典，确保所有词汇表项都有概率值
        for token in self.vocab:
            if token not in self.subword_probs:
                self.subword_probs[token] = self.min_prob

        return self.vocab

    def encode(self, text):
        """使用训练好的模型编码文本 - 修复版本"""
        if not self.vocab:
            raise ValueError("模型尚未训练")

        encoded = []
        for sentence in text:
            n = len(sentence)
            best_score = [-10 ** 15] * (n + 1)
            best_path = [None] * (n + 1)
            best_score[0] = 0

            # 处理空字符串
            if n == 0:
                encoded.append([])
                continue

            # 前向传递：计算最佳分割
            for end in range(1, n + 1):
                found = False
                # 限制子词最大长度以提高效率
                max_subword_len = min(5, end)  # 最大5个字符
                for start in range(max(0, end - max_subword_len), end):
                    subword = sentence[start:end]
                    if subword in self.vocab:
                        prob = self.subword_probs.get(subword, self.min_prob)
                        current_score = best_score[start] + self._safe_log(prob)
                        if current_score > best_score[end]:
                            best_score[end] = current_score
                            best_path[end] = (start, subword)
                            found = True

                # 回退机制：使用单个字符
                if not found:
                    start = end - 1
                    subword = sentence[start:end]
                    best_score[end] = best_score[start] + self._safe_log(self.min_prob)
                    best_path[end] = (start, subword)

            # 回溯获取最佳分割
            tokens = []
            pos = n
            while pos > 0:
                if best_path[pos] is None:
                    # 回退到单个字符
                    start = pos - 1
                    token = sentence[start:pos]
                    pos = start
                else:
                    start, token = best_path[pos]
                    pos = start
                tokens.append(token)

            encoded.append(list(reversed(tokens)))

        return encoded


# 测试函数
def test_sentencepiece():
    # 更健壮的测试数据
    text = [
        "This is a test sentence.",  # 英文
        "SentencePiece 是一个文本分词器。",  # 中文
        "It supports multiple languages.",  # 英文
        "こんにちは、世界！",  # 日语
        "안녕하세요 세상!",  # 韩语
        "12345 67890",  # 数字
        "!@#$%^&*()",  # 特殊符号
        "a",  # 单个字符
        "   ",  # 空白字符
        ""  # 空字符串
    ]

    print("开始SentencePiece训练...")
    sp = SentencePieceTrainer(vocab_size=100)
    vocab = sp.train(text, num_iterations=5)
    print("\n训练完成! 词汇表大小:", len(vocab))
    print("前20个子词:", vocab[:20])

    # 编码新文本
    test_text = [
        "This is a new sentence to tokenize.",
        "新句子测试",
        "Short",
        "Emoji test: 😊👍🌟",
        "A very long sentence that needs to be tokenized properly with SentencePiece.",
        ""  # 空字符串
    ]

    print("\n编码测试:")
    for i, sentence in enumerate(test_text):
        print(f"\n输入 {i + 1}: {repr(sentence)}")
        encoded = sp.encode([sentence])[0]
        print(f"分词结果: {encoded}")
        print(f"分词数量: {len(encoded)}")

    # 测试未知字符
    print("\n测试未知字符处理:")
    test_text = "ΔΣ unknown characters ΩΨ"
    print(f"输入: {test_text}")
    encoded = sp.encode([test_text])[0]
    print(f"分词结果: {encoded}")


if __name__ == "__main__":
    test_sentencepiece()