import torch
from torch import nn

from utils import (basename_noext, get_fashion_mnist_labels,
                   kmp_duplicate_lib_ok, load_data_fashion_minist, predict,
                   savefig, train_ani)

kmp_duplicate_lib_ok()
torch.set_printoptions(linewidth=120)

def main():
    batch_size = 256
    num_epochs = 10
    lr = 0.1

    train_dataloader, test_dataloader = load_data_fashion_minist(batch_size)

    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(784, 256, device="cuda"),
                        nn.ReLU(),
                        nn.Linear(256, 10, device="cuda"))

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.01)
            nn.init.zeros_(m.bias)

    net.apply(init_weights)

    loss = nn.CrossEntropyLoss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(), lr=lr)

    train_ani(net, train_dataloader, test_dataloader, loss, num_epochs, trainer, "cuda")

    savefig(f'out/{basename_noext(__file__)}_train.png')

    predict(net, test_dataloader, 18, "cuda")

    savefig(f'out/{basename_noext(__file__)}_pred.png')


if __name__ == '__main__':
    main()
