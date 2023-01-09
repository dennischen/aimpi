import time
import torch
from torch import nn

print(f'''{torch.device('cpu')}''')
print(f'''{torch.device('cuda')}''')
print(f'''{torch.device('cuda:0')}''')
print(f'''{torch.device('cuda:1')}''')

print([torch.device('cpu'), torch.device('cuda'), torch.device('cuda:0'), torch.device('cuda:1')])
print(torch.cuda.device_count())


def try_gpu(i=0):  #@save
    """如果存在,則返回gpu(i),否則返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():  #@save
    """返回所有可用的GPU,如果沒有GPU,則返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


print([try_gpu(), try_gpu(10), try_all_gpus()])

X = torch.ones(2, 3)
print(X)

Y = torch.rand(2, 3, device=try_gpu())
print(Y)

print(f'''A {X.cuda(0) + Y}''')
print(f'''B {X + Y.to('cpu')}''')

net = nn.Sequential(nn.Linear(3, 1))
print(f'C {net} {net[0].weight}')
net.to(device=try_gpu())
print(f'D {net} {net[0].weight}')

st1 = time.time()
for i in range(100):
    A = torch.ones(5000, 500)
    B = torch.ones(500, 5000)
    C = torch.matmul(A, B)
et1 = time.time()

print('cpu计算总时长:', round((et1 - st1) * 1000, 2), 'ms')

st2 = time.time()
for i in range(100):
    A = torch.ones(5000, 500, device=try_gpu())
    B = torch.ones(500, 5000, device=try_gpu())
    C = torch.matmul(A, B)
et2 = time.time()
print('gpu计算总时长:', round((et2 - st2) * 1000, 2), 'ms')
