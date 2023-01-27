import torch
from d2l import torch as d2l

X, W_xh = torch.normal(0, 1, (3, 1)), torch.normal(0, 1, (1, 4))
H, W_hh = torch.normal(0, 1, (3, 4)), torch.normal(0, 1, (4, 4))
print('X:', X.shape, X)
print('W_xh:', W_xh.shape, W_xh)
print('torch.matmul(X, W_xh):', torch.matmul(X, W_xh))
print('H:', H.shape, H)
print('W_hh:', W_hh.shape, W_hh)
print('torch.matmul(H, W_hh):', torch.matmul(H, W_hh))
Y1 = torch.matmul(X, W_xh) + torch.matmul(H, W_hh)
print('Y1:', Y1.shape, Y1)

print('torch.cat((X, H):', torch.cat((X, H), 1))
print('torch.cat((W_xh, W_hh):', torch.cat((W_xh, W_hh), 0))

Y2 = torch.matmul(torch.cat((X, H), 1), torch.cat((W_xh, W_hh), 0))
print('Y2:', Y2.shape, Y2)