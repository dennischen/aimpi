from typing import Callable, Union
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

train_iter: d2l.SeqDataLoader
vocab: d2l.Vocab

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

num_hiddens = 256
rnn_layer = nn.RNN(len(vocab), num_hiddens)

state = torch.zeros((1, batch_size, num_hiddens))
print(state.shape)

X = torch.rand(size=(num_steps, batch_size, len(vocab)))
Y, state_new = rnn_layer(X, state)
print(Y.shape, state_new.shape)


class RNNModel(nn.Module):
    """循環神經網路模型"""

    def __init__(self, rnn_layer: nn.RNN, vocab_size: int, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # 如果RNN是雙向的（之後將介紹），num_directions應該是2，否則應該是1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs: torch.Tensor, state: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # 全連接層首先將Y的形狀改為(時間步數*批次大小,隱藏單元數)
        # 它的輸出形狀是(時間步數*批次大小,詞表大小)。
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以張量作為隱狀態
            return torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device)
        else:
            # nn.LSTM以元組作為隱狀態
            return (
                torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device),
                torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device),
            )
