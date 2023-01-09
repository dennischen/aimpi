import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    # 用模型參數聲明層。這裡，我們聲明兩個全連接的層
    def __init__(self):
        # 呼叫MLP的父類Module的建構函式來執行必要的初始化。
        # 這樣，在類實例化時也可以指定其他函數參數，例如模型參數params（稍後將介紹）
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隱藏層
        self.out = nn.Linear(256, 10)  # 輸出層

    # 定義模型的前向傳播，即如何根據輸入X返回所需的模型輸出
    def forward(self, X):
        # 注意，這裡我們使用ReLU的函數版本，其在nn.functional模組中定義。
        return self.out(F.relu(self.hidden(X)))


net = MLP()
X = torch.rand(2, 20)
y = net(X)
print(f'{y}')


class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 這裡，module是Module子類的一個實例。我們把它保存在'Module'類的成員
            # 變數_modules中。_module的類型是OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict保證了按照成員新增的順序遍歷它們
        for block in self._modules.values():
            X = block(X)
        return X


net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

X = torch.rand(2, 20)
y = net(X)
print(f'{y}')


class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不計算梯度的隨機權重參數。因此其在訓練期間保持不變
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # 使用建立的常數參數以及relu和mm函數
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 復用全連接層。這相當於兩個全連接層共享參數
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()


net = FixedHiddenMLP()
X = torch.rand(2, 20)
y = net(X)
print(f'{y}')


class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))


net = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
X = torch.rand(2, 20)
y = net(X)
print(f'{y}')
