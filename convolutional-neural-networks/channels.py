import torch


def corr2d(X, K):
    """計算二維互相關運算"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


def corr2d_multi_in(X, K):
    # 先遍歷“X”和“K”的第0個維度（通道維度），再把它們加在一起
    return sum(corr2d(x, k) for x, k in zip(X, K))


# in-c,w,h
# 2,3,3
X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
# in-c,w,h
# 2,2,2
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

Y = torch.arange(6).reshape(2, 3)

print(f'X.shape {X.shape}')
print(f'K.shape {K.shape}')
print(f'Y.shape {Y.shape}')

for x, k, y in zip(X, K, Y):
    print('============')
    print(f'x {x.shape} {x}')
    print(f'k {k.shape} {k}')
    print(f'y {y.shape} {y}')
    print('============')

print(f'{corr2d_multi_in(X, K)}')


def corr2d_multi_in_out(X, K):
    # 迭代“K”的第0個維度，每次都對輸入“X”執行互相關運算。
    # 最後將所有結果都疊加在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)


print(f'K {K.shape} {K}')
# create 1 dim to K's beginning
K = torch.stack((K, K + 1, K + 2), 0)
print(f'K {K.shape} {K}')

print(f'{corr2d_multi_in_out(X, K)}')


def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    print(f'X shape {X.shape}')
    K = K.reshape((c_o, c_i))
    print(f'K shape {K.shape}')
    # 全連接層中的矩陣乘法
    Y = torch.matmul(K, X)
    print(f'Y shape {Y.shape}')
    return Y.reshape((c_o, h, w))


# in-c w h
X = torch.normal(0, 1, (3, 3, 3))
# out-c in-cc w h
K = torch.normal(0, 1, (2, 3, 1, 1))

print(f'X {X.shape} {X}')
print(f'K {K.shape} {K}')

Y = corr2d_multi_in_out(X, K)
print(f'Y {Y.shape} {Y}')
Y1 = corr2d_multi_in_out_1x1(X, K)
print(f'Y1 {Y1.shape} {Y1}')

print(f'{torch.abs(Y1 - Y)}')
assert float(torch.abs(Y1 - Y).sum()) < 1e-6