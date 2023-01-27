import math
from typing import Callable, Union

import torch
from d2l import torch as d2l
from torch import nn
from torch.nn import functional as F

batch_size, num_steps = 32, 35
train_iter: d2l.SeqDataLoader
vocab: d2l.Vocab

train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

# char mode, 26 + <unk> + ' ' = 28
print("len(vocab)", len(vocab))
print("vocab.token_freqs", vocab.token_freqs[:5])

print("one hot", F.one_hot(torch.tensor([0, 2]), len(vocab)))

X = torch.arange(10).reshape((2, 5))
print("X.shape", X.shape)
print("X.T.shape", X.T.shape)
print("F.one_hot(X.T, 28).shape", F.one_hot(X.T, 28).shape)
# print(F.one_hot(X.T, 28))


def get_params(vocab_size: int, num_hiddens: int, device: torch.device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 隱藏層參數
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    # 輸出層參數
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def init_rnn_state(batch_size: int, num_hiddens: int, device: torch.device):
    return (torch.zeros((batch_size, num_hiddens), device=device),)


def rnn(
    # (steps, batch_size, vocab_size)
    inputs: torch.Tensor,
    state: tuple[torch.Tensor],
    # (W_xh, W_hh, b_h, W_hq, b_q)
    params: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
):
    # inputs的形狀：(時間步數量，批次大小，詞表大小)
    W_xh, W_hh, b_h, W_hq, b_q = params
    (H,) = state
    outputs: list[torch.Tensor] = []
    # X的形狀：(批次大小，詞表大小)
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)


class RNNModelScratch:  # @save
    """從零開始實現的循環神經網路模型"""

    def __init__(
        self,
        vocab_size: int,
        num_hiddens: int,
        device: torch.device,
        get_params: Callable[
            [int, int, torch.device], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ],
        init_state: Callable[[int, int, torch.device], tuple[torch.Tensor]],
        forward_fn: Callable[
            [
                list[torch.Tensor],
                tuple[torch.Tensor],
                tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            ],
            tuple[torch.Tensor, tuple[torch.Tensor]],
        ],
    ):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        Y, HS = self.forward_fn(X, state, self.params)
        return (Y, HS)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params, init_rnn_state, rnn)
state = net.begin_state(X.shape[0], d2l.try_gpu())

X = torch.arange(10).reshape((2, 5)).to(d2l.try_gpu())
print("init state ", state[0].shape, state[0])
print("X ", X.shape, X)

Y, new_state = net(X, state)
print("new state len", len(new_state))
print("new state ", new_state[0].shape, new_state[0])
print("Y", Y.shape, Y)


def predict_ch8(prefix: str, num_preds: int, net: RNNModelScratch, vocab, device: torch.device):
    """在prefix後面生成新字元"""
    print('>>>>>predict_ch8 ', prefix)
    state = net.begin_state(batch_size=1, device=device)
    
    outputs = [vocab[prefix[0]]]
    print('outputs', outputs)
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:  # 預熱期
        inputs = get_input()
        print('warn  inputs', outputs)
        _, state = net(inputs, state)
        outputs.append(vocab[y])
        print('warn outputs', outputs)
    for _ in range(num_preds):  # 預測num_preds步
        y, state = net(get_input(), state)
        print('warn y', y)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return "".join([vocab.idx_to_token[i] for i in outputs])


r = predict_ch8("time traveller ", 10, net, vocab, d2l.try_gpu())
print(">>>>", r)
