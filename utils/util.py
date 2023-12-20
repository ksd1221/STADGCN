import numpy as np
import torch
import math
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.metrics import r2_score


def data_transform_multivariate(data, n_his, n_pred, device, target=3):
    """
    Produce train/test set (multivariate ver.)
    The task to perform is single-step ahead forecasting
    :param data: nparray
        Input data (Train, val, test dataset)
    :param n_his: Int
        Window size of the historical observation
    :param n_pred: Int
        Horizon to be predicted
    :param device: Torch.device
        Place the data to which device
    :return: Torch.Tensor
        x with the shape [
        len(data)-n_his-n_pred (생성 가능한 데이터 슬라이스 개수),
        n_c (# of channels on time domain),
        n_his,
        num_nodes (주유소의 개수)
        ]
        y with the shape [len(data)-n_his-n_pred, num_nodes]
    """

    # number of recordings, stations, features
    l, n_station, n_c = data.shape
    # number of instances
    num = l-n_his-n_pred

    x = np.zeros([num, n_c, n_his, n_station])
    x2 = np.zeros([num, n_station])
    y = np.zeros([num, n_station])

    idx = 0
    for i in range(l - n_his - n_pred):
        head = i
        tail = i + n_his
        # x is the historical observations
        x[idx, :, :, :] = data[head: tail].transpose(2, 0, 1)   # [n_his, n_station, n_c] -> [n_c, n_his, n_station]
        # x2 is the information whether to operate on target day
        x2[idx] = np.where(data[tail+n_pred-1, :, target] == 0, 0, 1)
        # y is 'n_pred'
        y[idx] = data[tail + n_pred - 1, :, target]
        # idx from 0 to num-1
        idx += 1

    return torch.Tensor(x).to(device), torch.LongTensor(x2).to(device), torch.Tensor(y).to(device)


def sequential_data_split(data, num_train, num_val, num_test, device, dtype=float):
    total_samples = len(data)
    idx = 0
    train_data, val_data, test_data = [], [], []

    while idx + num_train + num_val + num_test <= total_samples:
        train_ = data[idx:idx+num_train]
        val_ = data[idx+num_train:idx+num_train+num_val]
        test_ = data[idx+num_train+num_val:idx+num_train+num_val+num_test]

        if idx == 0:
            train_data, val_data, test_data = train_, val_, test_
        else:
            train_data = np.concatenate((train_data, train_), axis=0)
            val_data = np.concatenate((val_data, val_), axis=0)
            test_data = np.concatenate((test_data, test_), axis=0)
        idx += num_train+num_val+num_test

    if idx < total_samples:
        remaining_data = data[idx:]
        train_data = np.concatenate((train_data, remaining_data), axis=0)

    if data.ndim == 4: # x
        train_data = np.array(train_data).reshape(-1, data.shape[1], data.shape[2], data.shape[3])
        val_data = np.array(val_data).reshape(-1, data.shape[1], data.shape[2], data.shape[3])
        if num_test != 0:
            test_data = np.array(test_data).reshape(-1, data.shape[1], data.shape[2], data.shape[3])
    else:
        train_data = np.array(train_data).reshape(-1, data.shape[1])
        val_data = np.array(val_data).reshape(-1, data.shape[1])
        if num_test != 0:
            test_data = np.array(test_data).reshape(-1, data.shape[1])

    print('train: {}, val: {}, test: {}'.format(len(train_data), len(val_data), len(test_data)))

    if dtype == float:
        return torch.tensor(train_data).to(device), torch.tensor(val_data).to(device), torch.tensor(test_data).to(device)
    else:
        return (torch.LongTensor(train_data).to(device), torch.LongTensor(val_data).to(device),
                torch.LongTensor(test_data).to(device))


class Custom_CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(Custom_CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (
                        1 + math.cos(math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.eta_max = self.base_eta_max * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class EarlyStopping:
    def __init__(self, patience=5, delta=0.0001, min_loss=float('inf')):
        self.patience = patience
        self.delta = delta
        self.min_loss = min_loss
        self.counter = 0
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model):
        if val_loss < self.min_loss - self.delta:
            self.min_loss = val_loss
            self.counter = 0
            self.best_model = model.state_dict()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop, self.best_model


def evaluate_model_multi(model, loss, data_iter, num_features, module=None):
    model.eval()    # set evaluation mode, specially dropout, batch normalization etc.
    l_sum, n = 0.0, 0   # l_sum: sum of loss, n: # of data
    with torch.no_grad():   # inactivate gradient calculate in evaluation stage to make faster
        for x, y in data_iter:  # get batch size data with data_iter
            y_pred = model(x, num_features).view(len(x), -1) # predict then change predict size to (batch size, -1) to keep batch size
            if module==None:
                l = loss(y_pred, y, x[:, :, -1, -1])
            else:
                l = loss(y_pred, y)
            l_sum += l.item()*y.shape[0]    # weighted loss sum
            n += y.shape[0]
        return l_sum / n    # average of loss


def evaluate_metric_multi(model, data_iter, scaler, num_features):
    """
    calculate MAE, MAPE, RMSE and R2 score
    :param model: trained model
    :param data_iter:
    :param scaler:
    :return: MAE, MAPE, RMSE, R2
    """
    model.eval()
    with torch.no_grad():
        mae, mape, mse = [], [], []
        y_true_all, y_pred_all = [], []
        for x, y in data_iter:
            y = scaler.inverse_transform(y.cpu().numpy()).reshape(-1)
            y_pred = scaler.inverse_transform(model(x, num_features).view(len(x), -1).cpu().numpy()).reshape(-1)
            d = np.abs(y-y_pred)

            y_true_all.extend(y.tolist())
            y_pred_all.extend(y_pred.tolist())

            mae += d.tolist()
            mape += (d/y).tolist()
            mse += (d ** 2).tolist()
        MAE = np.array(mae).mean()
        MAPE = np.array(mape).mean()
        RMSE = np.sqrt(np.array(mse).mean())
        R2 = r2_score(y_true_all, y_pred_all)
        return MAE, MAPE, RMSE, R2