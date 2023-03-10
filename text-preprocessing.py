import collections
import re

from d2l import torch as d2l

d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt', '090b5e7e70c295757f55df93cb0a180b9691891a')


def read_time_machine():  #@save
    """將時間機器資料集載入到文字行的列表中"""
    with open(d2l.download('time_machine', cache_dir="download-data"), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


lines = read_time_machine()
print(f'# 文字總行數: {len(lines)}')
print(f'  Lines[0] {lines[0]}')
print(f'  Lines[10] {lines[10]}')


def tokenize(lines, token='word'):
    """將文字行拆分為單詞或字元詞元"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        raise Exception('錯誤：未知詞元類型：' + token)


tokens = tokenize(lines)

print(f'# tokens: {len(tokens)}')
print(f'  tokens[0] {tokens[0]}')
print(f'  tokens[10] {tokens[10]}')


class Vocab:
    """文字詞表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出現頻率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # 未知詞元的索引為0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知詞元的索引為0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs


def count_corpus(tokens):
    """統計詞元的頻率"""
    # 這裡的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 將詞元列表展平成一個列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


vocab = Vocab(tokens)
print('vocab tokens len', len(vocab.token_to_idx.items()))
print('top 10 vocab tokens', list(vocab.token_to_idx.items())[:10])
# print('vocab tokens', vocab.token_to_idx.items())


def load_corpus_time_machine(max_tokens=-1):  #@save
    """返回時光機器資料集的詞元索引列表和詞表"""
    tokens = tokenize(read_time_machine(), 'char')
    vocab = Vocab(tokens)
    # 因為時光機器資料集中的每個文字行不一定是一個句子或一個段落，
    # 所以將所有文字行展平到一個列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


# corpus here is the document's token indexs in vocab
corpus, vocab = load_corpus_time_machine()

print('vocab(char) tokens len', len(vocab.token_to_idx.items()))
print('top 10 vocab(char) tokens', list(vocab.token_to_idx.items())[:10])
print('corpus len', len(corpus))
print('corpus ', corpus[:10])