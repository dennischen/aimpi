import math
from typing import Callable, Union

import numpy as np
import torch
from d2l import torch as d2l
from torch import nn

from utils import (basename_noext, kmp_duplicate_lib_ok, savefig, synthetic_data, evaluate_loss, linreg, squared_loss,
                   sgd, load_data, load_data_fashion_minist, train_ani, plot)

import hashlib
import os
import tarfile
import zipfile
import requests

import pandas as pd
import pandas.core.frame as frame
import pandas.core.series as series

kmp_duplicate_lib_ok()
np.set_printoptions(linewidth=200, precision=3)
torch.set_printoptions(linewidth=200, sci_mode=False, precision=3)

DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'


def download(name, cache_dir=os.path.join('download-data', 'kaggle')):
    """下載一個DATA_HUB中的檔案，返回本地檔案名稱"""
    assert name in DATA_HUB, f"{name} 不存在於 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # 命中快取
    print(f'正在從{url}下載{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname


def download_extract(name, folder=None):  #@save
    """下載並解壓zip/tar檔案"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar檔案可以被解壓縮'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir


def download_all():  #@save
    """下載DATA_HUB中的所有檔案"""
    for name in DATA_HUB:
        download(name)


DATA_HUB['kaggle_house_train'] = (DATA_URL + 'kaggle_house_pred_train.csv', '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (DATA_URL + 'kaggle_house_pred_test.csv', 'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

train_data: frame.DataFrame = pd.read_csv(download('kaggle_house_train'))
test_data: frame.DataFrame = pd.read_csv(download('kaggle_house_test'))

print('===============data=============')
print(train_data.iloc[0:4, :])
print(test_data.iloc[-4:, :])

# print(train_data.shape)
# print(test_data.shape)
# print(train_data.iloc[0:4, :])
# print(test_data.iloc[0:4, :])

# concat data and remove id (column 0) and price column (last column in training data)
all_features: frame.DataFrame = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
print('===============concat=============')
print(all_features.iloc[0:4, :])
print(all_features.iloc[-4:, :])

# 若無法獲得測試資料，則可根據訓練資料計算均值和標準差
numeric_features_idx = all_features.dtypes[all_features.dtypes != 'object'].index


# standalize the data
def fn1(x: series.Series):
    return (x - x.mean()) / (x.std())


print('===============standalize=============')
all_features[numeric_features_idx] = all_features[numeric_features_idx].apply(fn1)
print(all_features.iloc[0:4, :])
print(all_features.iloc[-4:, :])

print('===============fill na=============')
# 在標準化資料之後，所有均值消失，因此我們可以將缺失值設定為0
all_features[numeric_features_idx] = all_features[numeric_features_idx].fillna(0)
print(all_features.iloc[0:4, :])
print(all_features.iloc[-4:, :])

print('===============add fields for object cloumn=============')
# “Dummy_na=True”將“na”（缺失值）視為有效的特徵值，並為其建立指示符特徵
all_features = pd.get_dummies(all_features, dummy_na=True)
print(all_features.iloc[0:4, :])
print(all_features.iloc[-4:, :])

n_train = train_data.shape[0]  # row number -> train number
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(train_data['SalePrice'].values.reshape(-1, 1), dtype=torch.float32)

print(f'train_features {train_features}')
print(f'train_labels {train_labels}')

loss = nn.MSELoss()
num_features = train_features.shape[1]


def get_net():
    net = nn.Sequential(nn.Linear(num_features, 1))
    return net


def log_rmse(net: nn.Module, X: torch.Tensor, y: torch.Tensor):
    # 為了在取對數時進一步穩定該值，將小於1的值設定為1
    clipped_preds = torch.clamp(net(X), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(y)))
    return rmse.item()


def train(net: nn.Module, train_features: torch.Tensor, train_labels: torch.Tensor, test_features: torch.Tensor,
          test_labels: torch.Tensor, num_epochs: int, learning_rate: float, weight_decay: float, batch_size: int):
    train_ls, test_ls = [], []
    train_dataloader = load_data((train_features, train_labels), batch_size)
    # 這裡使用的是Adam最佳化演算法
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch in range(num_epochs):
        X: torch.Tensor
        y: torch.Tensor
        for X, y in train_dataloader:
            optimizer.zero_grad()
            lo = loss(net(X), y)
            lo.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls


def get_k_fold_data(k: int, i: int, X: torch.Tensor, y: torch.Tensor):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_test, y_test = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_test, y_test


def k_fold_train(k: int, X_train: torch.Tensor, y_train: torch.Tensor, num_epochs: int, learning_rate: float,
                 weight_decay: float, batch_size: int):
    train_l_sum, test_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, test_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        test_l_sum += test_ls[-1]

        plot(list(range(1, num_epochs + 1)), [train_ls, test_ls],
             xlabel='epoch',
             ylabel='rmse',
             xlim=[1, num_epochs],
             legend=['train', 'test'],
             yscale='log',
             figsize=(5, 5))
        savefig(f'out/{basename_noext(__file__)}_{i}.png')
        print(f'fold {i + 1}, train log rmse {float(train_ls[-1]):f}, test log rmse {float(test_ls[-1]):f}')
    return train_l_sum / k, test_l_sum / k


k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, test_l = k_fold_train(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
print(f'{k}-fold: avg train log rmse: {float(train_l):f}, ' f'avg test log rmse: {float(test_l):f}')


def train_and_pred(train_features: torch.Tensor, test_features: torch.Tensor, train_labels: torch.Tensor,
                   test_data: torch.Tensor, num_epochs: int, lr: float, weight_decay: float, batch_size: int):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None, num_epochs, lr, weight_decay, batch_size)
    plot(np.arange(1, num_epochs + 1), [train_ls],
         xlabel='epoch',
         ylabel='log rmse',
         xlim=[1, num_epochs],
         yscale='log')
    savefig(f'out/{basename_noext(__file__)}_pred.png')
    print(f'train log rmse: {float(train_ls[-1]):f}')
    # 將網路應用於測試集。
    preds = net(test_features).detach().numpy()
    # 將其重新格式化以匯出到Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv(f'out/{basename_noext(__file__)}.csv', index=False)


train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)    