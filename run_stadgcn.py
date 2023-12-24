import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import tqdm
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import model.model as model
import utils.util as util
import utils.astgcn_util as util2
import gc

gc.collect()
torch.cuda.empty_cache()
warnings.filterwarnings('ignore')

class RUN_stadgcn:
    def __init__(
        self,
        learning_rate,
        batch_size,
        epochs,
        window,
        horizon,
        df_price,
        df_dist,
        df_macroeconomics,
        col_list,
        cat_list,
        lstm_hidden_dim,
        lstm_num_layers,
        nb_block,
        K,
        nb_chev_filter,
        nb_time_filter,
        time_strides,
        num_for_predict,
        save_path,
        device,
    ):

        self.lr = learning_rate
        self.bach_size = batch_size
        self.epochs = epochs
        self.window = window
        self.horizon = horizon
        self.df_price = df_price
        self.df_dist = df_dist
        self.df_macroeconomics = df_macroeconomics
        self.col_list = col_list
        self.cat_list = cat_list
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.device = device
        self.nb_block = nb_block
        self.K = K
        self.nb_chev_filter = nb_chev_filter
        self.nb_time_filter = nb_time_filter
        self.time_strides = time_strides
        self.num_for_predict = num_for_predict
        self.save_path = save_path

        # 0. hyperparameters
        min_weeks = 4
        in_channels = len(self.col_list) - len(cat_list) - 1
        len_input = self.window


        # 1. Preprocessing
        # 1.1 filter target
        df_price = self.df_price[~self.df_price['상표'].isin(['알뜰주유소', '알뜰(ex)'])]
        df_pivot = df_price.pivot_table(index='기간', columns='번호', values='휘발유', aggfunc='mean', fill_value=0)
        self.station_ids = list(df_pivot.columns)
        valid_counts = df_pivot[df_pivot != 0].count(axis=0)
        idx_list = np.where(valid_counts.values>min_weeks)[0].tolist()
        self.station_ids = [self.station_ids[i] for i in idx_list]
        df_price = df_price[df_price['번호'].isin(self.station_ids)]
        num_of_vertices = len(self.station_ids)

        # 1.2 merge datasets
        self.df_merge = pd.merge(df_price, self.df_macroeconomics, on=['year', 'month', 'week'], how='left')
        self.df_merge['supply'] = self.df_merge[['SK에너지', 'GS칼텍스', '현대오일뱅크', 'S-OIL']].mean(axis=1)

        # 1.3 make pivot
        self.num_periods = df_price['기간'].nunique()
        self.num_stations = len(self.station_ids)
        self.num_values = len(self.col_list)
        self.pivot_array, self.cardinalities, self.scaler_list = self.make_pivot_array(self.horizon)

        # 1.4 del outlier edge based on distance between stations
        self.df_dist = self.df_dist[
            self.df_dist['start'].isin(self.station_ids) & self.df_dist['end'].isin(self.station_ids)]

        # 25%와 75% 분위수 계산
        Q1 = self.df_dist['distance'].quantile(0.25)
        Q3 = self.df_dist['distance'].quantile(0.75)
        IQR = Q3 - Q1   # IQR (Inter-Quartile Range) 계산

        # 이상치 범위 계산
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        print(Q1, Q3, lower_bound, upper_bound)
        criteria_d = upper_bound

        # 2. make adjacency matrix
        adj_mx, distance_mx = util2.get_adjacency_matrix(self.df_dist, self.station_ids, criteria_d)
        L_tilde = util2.scaled_Laplacian(adj_mx)
        cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(device) for i in
                            util2.cheb_polynomial(L_tilde, K)]

        # 3. data split
        x, _, y = util.data_transform_multivariate(self.pivot_array, self.window, self.horizon, device, target=0)
        print(x.shape, y.shape)
        x = x.permute(0, 3, 1, 2)
        print(x.shape, y.shape)
        x_, y_ = np.array(x.cpu()), np.array(y.cpu())
        x_train, x_val, x_test = util.sequential_data_split(x_, 6, 2, 2, device=device)
        y_train, y_val, y_test = util.sequential_data_split(y_, 6, 2, 2, device=device)

        train_data = TensorDataset(x_train, y_train)
        train_iter = DataLoader(train_data, batch_size, shuffle=True)
        val_data = TensorDataset(x_val, y_val)
        val_iter = DataLoader(val_data, batch_size)
        test_data = TensorDataset(x_test, y_test)
        test_iter = DataLoader(test_data, batch_size)

        print(x_train.shape, y_train.shape)
        print(x_val.shape, y_val.shape)
        print(x_test.shape, y_test.shape)

        # 4. Declare the model object with default parameters.
        self.model = model.model_STADGCN(
            device,
            self.nb_block,
            in_channels,
            self.K,
            self.nb_chev_filter,
            self.nb_time_filter,
            self.time_strides,
            cheb_polynomials,
            self.num_for_predict,
            len_input,
            num_of_vertices,
            self.lstm_hidden_dim,
            self.lstm_num_layers,
            self.cardinalities
        )

        # 5. train model
        self.train_model(train_iter, val_iter)

        # 6. test model
        test_MAE, test_MAPE, test_RMSE, test_R2 = util.evaluate_metric_multi(
            self.model, test_iter, self.scaler_list[0], 5)
        print('MAE {:.5f} | MAPE {:.5f} | RMSE {:.5f} | R2 {:.5f}'.format(test_MAE, test_MAPE, test_RMSE, test_R2))


    def make_pivot_array(self, horizon):
        cardinalities = []  # make cardinality list
        scaler_list = []
        pivot_array = np.empty((self.num_periods, self.num_stations, self.num_values))

        for i, value in enumerate(self.col_list):
            # future operation features
            if value == 'operation_yn':
                df_pivot_ = self.df_merge.pivot_table(index='기간', columns='번호', values='휘발유', aggfunc='mean', fill_value=0)
                gasoline_lagged = np.roll(df_pivot_.values, shift=-horizon, axis=0)
                # 'horizon'개의 값들이 이전값으로 ffill
                for col_idx in range(gasoline_lagged.shape[1]):
                    gasoline_lagged[-horizon:, col_idx] = gasoline_lagged[-horizon - 1, col_idx]
                # 만약 휘발유 가격이 0(=nan)이면 0, 아니면 1로 운영여부 확인
                scaled_values = np.where(gasoline_lagged == 0, 0, 1)
                cardinalities.append(2)
                pivot_array[:, :, i] = scaled_values
            # numeric features
            elif value not in self.cat_list:
                df_pivot_ = self.df_merge.pivot_table(index='기간', columns='번호', values=value, aggfunc='mean', fill_value=0)
                scaler = StandardScaler()
                scaled_values = scaler.fit_transform(df_pivot_.values)
                pivot_array[:, :, i] = scaled_values
                scaler_list.append(scaler)
            # categorical features
            elif value in self.cat_list:
                self.df_merge[value] = self.df_merge[value].astype('category')
                self.df_merge[value + '_code'] = self.df_merge[value].cat.codes
                value_nunique = self.df_merge[value + '_code'].nunique()
                cardinalities.append(value_nunique + 1)
                df_pivot_ = self.df_merge.pivot_table(index='기간', columns='번호', values=value + '_code',
                                                 aggfunc=lambda x: x.mode().iloc[0], fill_value=value_nunique)
                pivot_array[:, :, i] = df_pivot_.values

        print(pivot_array.shape)

        return pivot_array, cardinalities, scaler_list

    def train_model(self, train_iter, val_iter):
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

        loss = util.CustomLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = util.Custom_CosineAnnealingWarmUpRestarts(optimizer, T_0=20, T_mult=1, eta_max=0.02,  T_up=10, gamma=0.5)
        early_stopping = util.EarlyStopping(patience=10, delta=0.0005)

        train_losses = []
        val_losses = []
        gc.collect()
        torch.cuda.empty_cache()

        for epoch in range(1, self.epochs + 1):
            l_sum, n = 0.0, 0  # 'l_sum' is epoch sum loss, 'n' is epoch instance number
            self.model.train()  # set 'train' mode
            for x, y in tqdm.tqdm(train_iter):
                gc.collect()
                torch.cuda.empty_cache()
                y_pred = self.model(x, 5).view(len(x), -1)
                l = loss(y_pred, y, x[:, :, -1, -1])  # 예측시점의 oper_yn
                optimizer.zero_grad(set_to_none=True)  # gradient initialize
                l.backward()  # calculate gradient of loss
                optimizer.step()  # update parameter
                l_sum += l.item() * y.shape[0]
                n += y.shape[0]

            scheduler.step()  # update learning rate
            val_loss = util.evaluate_model_multi(self.model, loss, val_iter, 5)  # epoch validation
            # GPU mem usage
            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
            # save model every epoch
            torch.save(self.model.state_dict(), self.save_path)  # when load model, use 'load_state_dict()

            print('Epoch {:03d} | lr {:.6f} |Train Loss {:.5f} | Val Loss {:.5f} | GPU {:.1f} MiB'.format(
              epoch, optimizer.param_groups[0]['lr'], l_sum / n, val_loss, gpu_mem_alloc))

            train_losses.append(l_sum / n)
            val_losses.append(val_loss)

            # early stop
            should_stop, best_model = early_stopping(val_loss, self.model)
            if should_stop:
                print('Early stopping')
                break

        if best_model is not None:
            self.model.load_state_dict(best_model)


# set directories
cur_dir = os.getcwd()
data_dir = os.path.join(cur_dir, 'data')
df_price = pd.read_csv(os.path.join(data_dir, 'price_seoul.csv'), encoding='utf-8-sig')
df_distance = pd.read_csv(os.path.join(data_dir, 'df_dist.csv'), encoding='utf-8-sig')
df_international = pd.read_csv(os.path.join(data_dir, 'macroeconomics.csv'), encoding='utf-8-sig')
save_model_dir = os.path.join(cur_dir, 'checkpoints/stadgcn.pt')
os.makedirs(os.path.join(cur_dir, 'checkpoints'), exist_ok=True)

# check device
DisableGPU = False
device = torch.device('cuda') if torch.cuda.is_available() and not DisableGPU else torch.device('cpu')

model = RUN_stadgcn(
        learning_rate=0.001,
        batch_size=16,
        epochs=300,
        window=24,
        horizon=1,
        df_price=df_price,
        df_dist=df_distance,
        df_macroeconomics=df_international,
        col_list=['휘발유', '경유', 'Dubai', 'WTI', 'supply', '상표', '셀프여부', 'operation_yn'],
        cat_list=['상표', '셀프여부'],
        lstm_hidden_dim=64,
        lstm_num_layers=2,
        nb_block=2,
        K=3,
        nb_chev_filter=64,
        nb_time_filter=64,
        time_strides=1,
        num_for_predict=1,
        save_path=save_model_dir,
        device=device,
    )
