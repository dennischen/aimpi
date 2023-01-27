import random
import re

import numpy as np
import torch
from d2l import torch as d2l

from utils import (basename_noext, get_fashion_mnist_labels, kmp_duplicate_lib_ok, load_data_fashion_mnist, predict,
                   savefig, train_ani)

kmp_duplicate_lib_ok()
torch.set_printoptions(linewidth=120)

d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt', '090b5e7e70c295757f55df93cb0a180b9691891a')


def read_time_machine():  #@save
    """將時間機器資料集載入到文字行的列表中"""
    with open(d2l.download('time_machine', cache_dir="download-data"), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


tokens = d2l.tokenize(read_time_machine())
# 因為每個文字行不一定是一個句子或一個段落，因此我們把所有文字行拼接到一起
corpus = [token for line in tokens for token in line]
print('corpus', corpus[:10])
vocab = d2l.Vocab(corpus)
print('vocab.token_freqs', vocab.token_freqs[:10])

freqs = [freq for _, freq in vocab.token_freqs]
d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)', xscale='log', yscale='log')
savefig(f'out/{basename_noext(__file__)}.png')

bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
bigram_vocab = d2l.Vocab(bigram_tokens)
print('BiGram', bigram_vocab.token_freqs[:10])

trigram_tokens = [triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])]
trigram_vocab = d2l.Vocab(trigram_tokens)
print('TriGram', trigram_vocab.token_freqs[:10])

bigram_freqs = [freq for _, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for _, freq in trigram_vocab.token_freqs]
d2l.plot([freqs, bigram_freqs, trigram_freqs],
         xlabel='token: x',
         ylabel='frequency: n(x)',
         xscale='log',
         yscale='log',
         legend=['unigram', 'bigram', 'trigram'])
savefig(f'out/{basename_noext(__file__)}_3.png')


def seq_data_iter_random(corpus, batch_size, num_steps):
    """使用隨機抽樣生成一個小批次子序列"""
    # 從隨機偏移量開始對序列進行分區，隨機範圍包括num_steps-1
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 減去1，是因為我們需要考慮標籤
    num_subseqs = (len(corpus) - 1) // num_steps
    # 長度為num_steps的子序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 在隨機抽樣的迭代過程中，
    # 來自兩個相鄰的、隨機的、小批次中的子序列不一定在原始序列上相鄰
    random.shuffle(initial_indices)

    def data(pos):
        # 返回從pos位置開始的長度為num_steps的序列
        return corpus[pos:pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 在這裡，initial_indices包含子序列的隨機起始索引
        initial_indices_per_batch = initial_indices[i:i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield np.array(X), np.array(Y)


my_seq = list(range(35))
i = 0
for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    print(i, 'X:', X)
    print(i, 'Y:', Y)
    i += 1


def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    """使用順序分區生成一個小批次子序列"""
    # 從隨機偏移量開始劃分序列
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = np.array(corpus[offset:offset + num_tokens])
    Ys = np.array(corpus[offset + 1:offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i:i + num_steps]
        Y = Ys[:, i:i + num_steps]
        yield X, Y


my_seq = list(range(35))
i = 0
for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
    print(i, 'X:', X)
    print(i, 'Y:', Y)
    i += 1

