import torch
from torch import nn
from utils import (basename_noext, get_fashion_mnist_labels,
                   kmp_duplicate_lib_ok, load_data_fashion_mnist, predict,
                   savefig, train_ani)

kmp_duplicate_lib_ok()
torch.set_printoptions(linewidth=200, sci_mode=False, precision=3, threshold=100)

batch_size = 256
lr = 0.1
num_epochs = 10

train_dataloader, test_dataloader = load_data_fashion_mnist(batch_size)


for X, y in train_dataloader:
    print('train_dataloader X', X.shape)
    print('train_dataloader y', y.shape)
    break

# Flatten makes 28x28 to 784
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))


def init_linear(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.01)
        nn.init.zeros_(m.bias)


net.apply(init_linear)

loss = nn.CrossEntropyLoss(reduction='none')

trainer = torch.optim.SGD(net.parameters(), lr=lr)

train_ani(net, train_dataloader, test_dataloader, loss, num_epochs, trainer)

savefig(f'out/{basename_noext(__file__)}_train.png')

predict(net, test_dataloader)

savefig(f'out/{basename_noext(__file__)}_pred.png')