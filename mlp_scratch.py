import torch
from torch import nn

from utils import (basename_noext, get_fashion_mnist_labels,
                   kmp_duplicate_lib_ok, load_data_fashion_mnist, predict,
                   savefig, train_ani)

kmp_duplicate_lib_ok()
torch.set_printoptions(linewidth=120)
# torch.set_default_tensor_type('torch.cuda.FloatTensor')

def main():
    batch_size = 256
    num_epochs = 10
    lr = 0.1
    train_dataloader, test_dataloader = load_data_fashion_mnist(batch_size)

    num_inputs = 784  # 28 x 28
    num_hiddens = 256  # use value of 2**n
    num_outputs = 10  # 10 category

    print(f'Default tensor {torch.arange(4, device="cuda")}')

    W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True, device="cuda") * 0.01)
    b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True, device="cuda"))
    W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True, device="cuda") * 0.01)
    b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True, device="cuda"))

    params = [W1, b1, W2, b2]

    ######################################
    def relu(X: torch.Tensor):
        a = torch.zeros_like(X, device="cuda")
        return torch.max(X, a)

    def net(X):
        X = X.reshape((-1, num_inputs))
        H = relu(X @ W1 + b1)  # 這裡“@”代表矩陣乘法
        # H = torch.matmul(X, W1) + b1
        return (H @ W2 + b2)

    loss = nn.CrossEntropyLoss(reduction='none')
    trainer = torch.optim.SGD(params, lr=lr)

    train_ani(net, train_dataloader, test_dataloader, loss, num_epochs, trainer, "cuda")

    savefig(f'out/{basename_noext(__file__)}_train.png')

    predict(net, test_dataloader, 18, "cuda")

    savefig(f'out/{basename_noext(__file__)}_pred.png')


if __name__ == '__main__':
    main()
