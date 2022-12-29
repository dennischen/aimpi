import os

import torchvision
from d2l import torch as d2l
from torch.utils import data
from torchvision import transforms


def kmp_duplicate_lib_ok(on: bool = True):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" if on else "FALSE"


def basename_noext(p):
    return os.path.splitext(os.path.basename(p))[0]


def load_data_fashion_minist(batch_size: int):
    trans = transforms.Compose([transforms.ToTensor()])
    mnist_train = torchvision.datasets.FashionMNIST(root="download-data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="download-data", train=False, transform=trans, download=True)

    train_dataloader = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_dataloader = data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader


def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST資料集的文字標籤"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

