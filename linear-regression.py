import math
import time
import numpy as np
import torch
from d2l import torch as d2l

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
def stem(p):
    return os.path.splitext(os.path.basename(p))[0]

n = 10000
a = torch.ones(n)
b = torch.ones(n)

c = torch.zeros(n)
t = time.time()
for i in range(n):
    c[i] = a[i] + b[i]

print(f'a + b = {c}')
print(f'Loop spent \t{time.time() - t:.5f} sec')
t = time.time()
c = a + b
print(f'a + b = {c}')
print(f'Tensor spent \t{time.time() - t:.5f} sec')


def normal_distribution(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 * (x - mu)**2 / sigma**2)


# Use NumPy again for visualization
x = np.arange(-7, 7, 0.01) # -7 > 7, step +0.01

print(f'x = {x}, size = {x.size}')

# Mean and standard deviation pairs
params = [(0, 1), (0, 2), (3, 1)]

norms = [normal_distribution(x, mu, sigma) for mu, sigma in params]

print(f'y = {norms}')

d2l.plot(x, norms, xlabel='x',
         ylabel='p(x)', figsize=(4.5, 2.5),
         legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])


d2l.plt.savefig(f'out/{stem(__file__)}.png')
