import torch
from torch import nn

# 請注意，這裡每邊都填充了1行或1列，因此總共新增了2行或2列
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)


# 為了方便起見，我們定義了一個計算摺積層的函數。
# 此函數初始化摺積層權重，並對輸入和輸出提高和縮減相應的維數
def comp_conv2d(conv2d, X):
    # wrap
    # 這裡的(1,1)表示批次大小和通道數都是1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)

    # unwrap
    # 省略前兩個維度：批次大小和通道
    return Y.reshape(Y.shape[2:])


X = torch.rand(size=(8, 8))
print(f'{comp_conv2d(conv2d, X).shape}')

conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
print(f'{comp_conv2d(conv2d, X).shape}')