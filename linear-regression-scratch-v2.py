import torch
from d2l import torch as d2l

from utils import kmp_duplicate_lib_ok

kmp_duplicate_lib_ok()

DEBUG = False


class LinearRegressionScratch(d2l.Module):
    def __init__(self, num_inputs: int, lr: float, sigma=0.01):
        super().__init__()
        # self.save_hyperparameters()
        self.num_imputs = num_inputs
        self.lr = lr
        self.sigma = sigma
      
        self.w = torch.normal(0, sigma, (num_inputs, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)

    def forward(self, X: torch.Tensor):
        """The linear regression model."""
        return torch.matmul(X, self.w) + self.b

    def loss(self, y_hat: torch.Tensor, y: torch.Tensor):
        lo = (y_hat - y)**2 / 2
        return lo.mean()


class SGD(d2l.HyperParameters):
    def __init__(self, params: tuple[torch.Tensor], lr: float):
        """Minibatch stochastic gradient descent."""
        # self.save_hyperparameters()
        self.params = params
        self.lr = lr

    def step(self):
        for param in self.params:
            param -= self.lr * param.grad

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

@d2l.add_to_class(LinearRegressionScratch)
def configure_optimizers(self):
    return SGD([self.w, self.b], self.lr)

@d2l.add_to_class(d2l.Trainer)
def prepare_batch(self, batch):
    return batch


@d2l.add_to_class(d2l.Trainer)
def fit_epoch(self):
    self.model.train()
    for batch in self.train_dataloader:
        loss = self.model.training_step(self.prepare_batch(batch))
        self.optim.zero_grad()
        with torch.no_grad():
            loss.backward()
            if self.gradient_clip_val > 0:  # To be discussed later
                self.clip_gradients(self.gradient_clip_val, self.model)
            self.optim.step()
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    self.model.eval()
    for batch in self.val_dataloader:
        with torch.no_grad():
            self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1


model = LinearRegressionScratch(2, lr=0.03)
data = d2l.SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
trainer = d2l.Trainer(max_epochs=3)
trainer.fit(model, data)


print(f'different in estimating w: {data.w - model.w.reshape(data.w.shape)}')
print(f'different in estimating b: {data.b - model.b}')
