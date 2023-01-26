import torch
import torch.nn as nn
from utils import (kmp_duplicate_lib_ok, basename_noext, savefig, plot, load_data, evaluate_loss)

kmp_duplicate_lib_ok()
torch.set_printoptions(linewidth=200, sci_mode=False, precision=3, threshold=100)

T = 1000  # 總共產生1000個點
# [1000]
time = torch.arange(1, T + 1, dtype=torch.float32)
print('time', time.shape, time)
# [1000]
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T, ))
print('x', x.shape, x)
plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
savefig(f'out/{basename_noext(__file__)}.png')

tau = 4
# [996, 4] , 996 examples, 4 input in each example, value in each example is from x[t:t+tau]
features = torch.zeros((T - tau, tau))
for i in range(tau):
    features[:, i] = x[i:T - tau + i]
print('features', features.shape, features)

# [996,1], 996 results, 1 ouput in each result, value in each result is from [t+tau]
labels = x[tau:].reshape((-1, 1))
print('labels', labels.shape, labels)

batch_size, n_train = 16, 600
# 只取前n_train個樣本用於訓練 (small than T)
train_dataloader = load_data((features[:n_train], labels[:n_train]), batch_size, is_train=True)


# 一個簡單的多層感知機, tau input, 1 output
def build_net():
    # 初始化網路權重的函數
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

    net = nn.Sequential(nn.Linear(tau, 10), nn.ReLU(), nn.Linear(10, 1))
    net.apply(init_weights)
    return net


# 平方損失。注意：MSELoss計算平方誤差時不帶係數1/2
loss = nn.MSELoss(reduction='none')


def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:

            trainer.zero_grad()
            lo = loss(net(X), y)
            lo.sum().backward()
            trainer.step()
        print(f'epoch {epoch + 1}, ' f'loss: {evaluate_loss(net, train_iter, loss):f}')


net = build_net()
train(net, train_dataloader, loss, 5, 0.01)

# 由T個example,
# 每個example為x中的依序四筆資料(x[t:t+tau]),
# 其result為x中依序的第五筆資料x[t+tau]
# 訓練出網路

# prediction
# 使用訓練時用的資料 [996, 4]來進行預測, 擬合度高
print('onestep features ', features.shape)
onestep_preds = net(features)
print('onestep_preds ', onestep_preds.shape)
plot([time, time[tau:]], [x.detach().numpy(), onestep_preds.detach().numpy()],
     'time',
     'x',
     legend=['data', '1-step preds'],
     xlim=[1, 1000],
     figsize=(6, 3))

savefig(f'out/{basename_noext(__file__)}_onestep.png')

multistep_preds = torch.zeros(T)
# init the true value in x to multistep_preds in range 0 ~ 600+4
multistep_preds[:n_train + tau] = x[:n_train + tau]

print('multistep_preds init', multistep_preds.shape)

# predict 從n_train+tau 到 T, 一次只predict一筆, 並且predict結果成為接下來predict的輸入(非本來的x[t])
for i in range(n_train + tau, T):
    X = multistep_preds[i - tau:i].reshape((1, -1))
    y = net(X)
    print('multistep features ', i, X.shape, y.shape)
    multistep_preds[i] = y

print('multistep_preds final', multistep_preds.shape)

plot([time, time[tau:], time[n_train + tau:]],
     [x.detach().numpy(),
      onestep_preds.detach().numpy(), multistep_preds[n_train + tau:].detach().numpy()],
     'time',
     'x',
     legend=['data', '1-step preds', 'multistep preds'],
     xlim=[1, 1000],
     figsize=(6, 3))
savefig(f'out/{basename_noext(__file__)}_multisteps.png')

max_steps = 64

# tau + max_steps = 64+4 = 68 cases
# 933, 68
features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))
# init, frist i column （i<tau）是來自x的觀測, 其時間步從（i）到（i+T-tau-max_steps+1）
for i in range(tau):
    features[:, i] = x[i:i + T - tau - max_steps + 1]

print('all_step_case_preds init', features.shape)
# 以下eval部份看不懂

# eval , column i（i>=tau）是來自（i-tau+1）步的預測，其時間步從（i）到（i+T-tau-max_steps+1）
for i in range(tau, tau + max_steps):
    X = features[:, i - tau:i]
    y = net(X)
    print('all step cases features ', i, X.shape, y.shape)
    features[:, i] = y.reshape(-1)

print('all step cases preds final', features.shape)

show_steps = (1, 4, 16, 64)
plot([time[tau + i - 1:T - max_steps + i] for i in show_steps],
     [features[:, tau + i - 1].detach().numpy() for i in show_steps],
     'time',
     'x',
     legend=[f'{i}-step preds' for i in show_steps],
     xlim=[5, 1000],
     figsize=(6, 3))

savefig(f'out/{basename_noext(__file__)}_all_steps_cases.png')
