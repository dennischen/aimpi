import torch
import torchvision
import numpy
import math
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
from matplotlib.axes import Axes, Subplot
from matplotlib.figure import Figure, FigureBase, figaspect
import multiprocessing

from utils import basename_noext, kmp_duplicate_lib_ok

kmp_duplicate_lib_ok()
# d2l.use_svg_display()

# fix "An attempt has been made to start a new process before the current process has finished its bootstrapping phase."
# wrap process to a method and call it in main
def run_in_main():
    print('Start to run in mp mode')

    img_resize = 64

    trans = transforms.Compose([transforms.Resize(img_resize), transforms.ToTensor()])
    mnist_train = torchvision.datasets.FashionMNIST(root="download-data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="download-data", train=False, transform=trans, download=True)

    print(f'train data size: {len(mnist_train)}, test data size: {len(mnist_test)}')

    # <class 'torchvision.datasets.mnist.FashionMNIST'>, 60000
    print(f'{type(mnist_train)}, {len(mnist_train)}')

    # <class 'tuple'>, 2
    # (feature, label)
    print(f'{type(mnist_train[0])}, {len(mnist_train[0])}')

    # <class 'torch.Tensor'>, torch.Size([1, 28, 28])
    # [ 1 , height, width]
    print(f'{type(mnist_train[0][0])}, {mnist_train[0][0].shape}')
    # print(f'{mnist_train[0][0]}')

    # <class 'int'>, 9
    print(f'{type(mnist_train[0][1])}, {mnist_train[0][1]}')


    def get_fashion_mnist_labels(labels):
        """返回Fashion-MNIST資料集的文字標籤"""
        text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        return [text_labels[int(i)] for i in labels]


    batch_size = 16

    dataloader = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=False)

    dataloader_iter = iter(dataloader)

    X: torch.Tensor
    y: torch.Tensor
    X, y = next(dataloader_iter)

    print(f'{X.shape}, {y.shape}')

    print(f'{get_fashion_mnist_labels(y)}')

    img_cols = 8


    def show_images(imgs: torch.Tensor, num_rows: int, num_cols: int, titles: tuple[str] = None, scale=1.5):
        """繪製圖像列表"""
        figsize = (num_cols * scale, num_rows * scale)
        axes_arr: numpy.ndarray[Axes]
        _, axes_arr = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
        axes_arr = axes_arr.flatten()
        axes: Axes
        for i, (axes, img) in enumerate(zip(axes_arr, imgs)):
            if torch.is_tensor(img):
                pixels = img.numpy()  # (28, 28)
                axes.imshow(pixels)
            else:
                # PIL圖片
                axes.imshow(img)
            axes.axes.get_xaxis().set_visible(False)
            axes.axes.get_yaxis().set_visible(False)
            if titles:
                axes.set_title(titles[i])
        return axes_arr


    img_rows = batch_size / img_cols
    img_rows = math.ceil(img_rows)
    show_images(X.reshape(batch_size, img_resize, img_resize), img_rows, img_cols, titles=get_fashion_mnist_labels(y))

    d2l.plt.savefig(f'out/{basename_noext(__file__)}.png')

    batch_size = 256

    # for i in ([0, 1, 4]):
    #     print(f'Number of worker {i}')
    #     train_dataloader = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=i)
    #     timer = d2l.Timer()
    #     j = 0
    #     for X, y in train_dataloader:
    #         j += 1
    #         continue
    #     print(f'Number of worker {i}, read loop {j}, time cost {timer.stop():.2f} sec')


    train_dataloader = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_dataloader = data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True)




if __name__ == '__main__':
    run_in_main()
