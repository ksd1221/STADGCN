import torch
import torch.nn as nn
import torch.nn.functional as F
from MAGNET.model.layers import Temporal_Attention_layer, Spatial_Attention_layer, cheb_conv_withSAt, MultipleEmbeddings

class ASTGCN_block(nn.Module):
    def __init__(self, DEVICE, in_channels, K, nb_chev_filter, nb_time_filter,time_strides, cheb_polynomials, num_of_vertices, num_of_timesteps):
        """
        Make ASTGCN Block. This block integrates attention mechanisms across time and space with graph convolution
        :param DEVICE: CPU or GPU
        :param in_channels: # of input features
        :param K: degree of the Chebyshev polynomial
        :param nb_chev_filter: # of output filters of the Chebyshev convolution
        :param nb_time_filter: # of output filters of time convolution
        :param time_strides: stride of time convolution
        :param cheb_polynomials: coefficients of the Chebyshev polynomial
        :param num_of_vertices: # of Graph nodes
        :param num_of_timesteps: length of time dimension
        """
        super(ASTGCN_block, self).__init__()
        # Initialize Temporal Attention Layer
        self.TAt = Temporal_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        # Initialize Spatial Attention Layer
        self.SAt = Spatial_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        # Initialize Chebyshev Graph Convolution with Spatial Attention
        self.cheb_conv_SAt = cheb_conv_withSAt(K, cheb_polynomials, in_channels, nb_chev_filter)
        # Initialize Time Convolution Layer
        self.time_conv = nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(1, 3), stride=(1, time_strides), padding=(0, 1))
        # Initialize Residual Convolution Layer for shortcut connection
        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self.ln = nn.LayerNorm(nb_time_filter)

    def forward(self, x):
        """
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, nb_time_filter, T)
        """
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        # Temporal Attention Layer
        temporal_At = self.TAt(x)   # (b, T, T)
        # Multiply the temporal attetion weights with the input
        x_TAt = torch.matmul(x.reshape(batch_size, -1, num_of_timesteps), temporal_At).reshape(
            batch_size, num_of_vertices, num_of_features, num_of_timesteps)

        # Spatial Attention Layer
        spatial_At = self.SAt(x_TAt)    # (b, N, N)

        # chebyshev Graph Convolution
        spatial_gcn = self.cheb_conv_SAt(x, spatial_At) # (b, N, F, T)

        # Time Convolution Layer: convolution along the time axis
        time_conv_output = self.time_conv(spatial_gcn.permute(0, 2, 1, 3))  # (b, N, F, T) -> (b, F, N, T)

        # residual shortcut Connection
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))  # (b, N, F, T) -> (b, F, N, T)
        x_residual = self.ln(F.relu(x_residual+time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)

        # (b, F, N, T) -> (b, T, N, F) -> (b, N, F, T)

        return x_residual


class ASTGCN_submodule(nn.Module):
    def __init__(self, DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides,
                 cheb_polynomials, num_for_predict, len_input, num_of_vertices):
        """
        Initialize the ASTGCN submodule
        :param DEVICE: CPU/GPU
        :param nb_block: # of ASTCN blocks in the submodule
        :param in_channels: # of input features (F_in)
        :param K: Degree of Chebyshev polynomial (a hyperparamter for the spatial convolution)
        :param nb_chev_filter: # of output filters in Chebyshev convolution
        :param nb_time_filter: # of output filters in temporal convolution
        :param time_strides: stride of the temporal convolution
        :param cheb_polynomials: pre-calculated Chebyshev polynomials
        :param num_for_predict: Number of steps to predict (used in the last layer)
        :param len_input: Length of the input sequence (T)
        :param num_of_vertices: Number of nodes (N)
        """

        super(ASTGCN_submodule, self).__init__()
        self.BlockList = nn.ModuleList()
        # Initialize the first ASTGCN block with the given parameters
        self.BlockList.append(ASTGCN_block(
            DEVICE, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials, num_of_vertices, len_input))

        # Initialize the remaining ASTGCN blocks (nb_block - 1 of them) with updated paramters
        self.BlockList.extend([ASTGCN_block(DEVICE, nb_time_filter, K, nb_chev_filter, nb_time_filter,
            1, cheb_polynomials, num_of_vertices, len_input//time_strides) for _ in range(nb_block-1)])

        # Final convolution layer to map to the output sequence
        self.final_conv = nn.Conv2d(int(len_input/time_strides), num_for_predict, kernel_size=(1, nb_time_filter))

        self.DEVICE = DEVICE
        self.to(DEVICE)

    def forward(self, x):
        """

        :param x: (B, N_nodes, F_in, T_in) F_in: # of input features, T_in: length of input sequence
        :return: (B, N_nodes, T_out)
        """

        # Pass the input through each ASTGCN block in the submodule
        for block in self.BlockList:
            x = block(x)

        # Final convolution step
        output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)
        # (B, N, F, T) -> (B, T, N, F) -> convolution -> (B, c_out*T, N, 1) -> (B, c_out*T, N) -> (B, N, T)

        return output


class LSTMModel(nn.Module):
  def __init__(self, input_dim, hidden_dim, num_layers, output_dim, DEVICE):
    super(LSTMModel, self).__init__()

    self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_dim, output_dim)
    self.layer_norm = nn.LayerNorm(hidden_dim)
    self.relu = nn.ReLU()
    self.DEVICE = DEVICE

  def forward(self, x):
    # x (batch, feature, window, nodes)
    batch_size, feature, window, nodes = x.size()

    # 각 노드별로 LSTM 계산을 수행하고 결과를 저장할 배열
    outputs = []

    for i in range(nodes):
        x_node = x[:, :, :, i]  # 형태: (batch, feature, window)
        x_node = x_node.permute(0, 2, 1)  # LSTM에 적합한 형태로 변경: (batch, window, feature)

        h_0 = torch.zeros(self.lstm.num_layers, x_node.size(0), self.lstm.hidden_size).requires_grad_().to(self.DEVICE)
        c_0 = torch.zeros(self.lstm.num_layers, x_node.size(0), self.lstm.hidden_size).requires_grad_().to(self.DEVICE)

        out, (hn, cn) = self.lstm(x_node, (h_0.detach(), c_0.detach()))
        out = self.layer_norm(self.relu(out))
        out = self.fc(out[:, -1, :])  # 마지막 시점의 hidden state만 사용

        outputs.append(out)

    outputs = torch.stack(outputs, dim=1).squeeze()  # 형태: (batch, nodes)

    return outputs


class model_MAGNET(nn.Module):
    def __init__(self, DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides,
                 cheb_polynomials, num_for_predict, len_input, num_of_vertices,
                 lstm_hidden_dim, lstm_num_layers, cardinalities):
        """
        Initialize the ASTGCN submodule
        :param DEVICE: CPU/GPU
        :param nb_block: # of ASTCN blocks in the submodule
        :param in_channels: # of input features (F_in)
        :param K: Degree of Chebyshev polynomial (a hyperparamter for the spatial convolution)
        :param nb_chev_filter: # of output filters in Chebyshev convolution
        :param nb_time_filter: # of output filters in temporal convolution
        :param time_strides: stride of the temporal convolution
        :param cheb_polynomials: pre-calculated Chebyshev polynomials
        :param num_for_predict: Number of steps to predict (used in the last layer)
        :param len_input: Length of the input sequence (T)
        :param num_of_vertices: Number of nodes (N)
        :param lstm_hidden_dim:
        :param lstm_out_dim:
        :param lstm_num_layers:
        :param cardinalities: cardinality list of categorical features
        """

        super(model_MAGNET, self).__init__()
        # embedding layer
        self.multiple_embeddings = MultipleEmbeddings(cardinalities)

        # Initialize the first ASTGCN block with the given parameters
        self.BlockList_astgcn = nn.ModuleList()

        self.BlockList_astgcn.append(ASTGCN_block(
            DEVICE, in_channels + sum(self.multiple_embeddings.embed_size_list), K, nb_chev_filter, nb_time_filter,
            time_strides, cheb_polynomials, num_of_vertices, len_input))

        # Initialize the remaining ASTGCN blocks (nb_block - 1 of them) with updated paramters
        for _ in range(nb_block - 1):
            self.BlockList_astgcn.append(ASTGCN_block(DEVICE, nb_time_filter, K, nb_chev_filter, nb_time_filter,
                                                      1, cheb_polynomials, num_of_vertices, len_input // time_strides))

        # Final convolution layer to map to the output sequence
        self.final_conv = nn.Conv2d(int(len_input / time_strides), num_for_predict, kernel_size=(1, nb_time_filter))
        self.lstm = LSTMModel(in_channels + sum(self.multiple_embeddings.embed_size_list), lstm_hidden_dim,
                              lstm_num_layers, 1, DEVICE).to(DEVICE)

        # self.DotProductAttention = DotProductAttention()
        # self.ln = nn.LayerNorm(nb_block+1)
        self.final_fc = nn.Linear(nb_block + 1, 1)  # nb_block+1, 1) # model 개수 (lstm, astgcn), 최종 예측값 개수

        self.DEVICE = DEVICE
        self.to(DEVICE)

    def forward(self, x, num_x_size):
        """

        :param x: (B, N_nodes, F_in, T_in) F_in: # of input features, T_in: length of input sequence
        :return: (B, N_nodes, T_out)
        """
        # seperate x_num, x_cat
        x_num = x[:, :, :num_x_size, :]
        x_cat = x[:, :, num_x_size:, :]

        # categorical feature embedding
        embedded_cat_features = self.multiple_embeddings(x_cat)

        # concat numeric features and embedded categorical features
        x = torch.cat((x_num, embedded_cat_features), dim=2).to(self.DEVICE)

        # lstm output
        lstm_out = self.lstm(x.permute(0, 2, 3, 1)).unsqueeze(-1)  # (B, N) -> (B, N, 1)

        # Pass the input through each ASTGCN block in the submodule
        astgcn_outputs = []  # 각 block의 출력을 저장할 list
        for i, block in enumerate(self.BlockList_astgcn):
            x = block(x)
            # (B, N, F, T) -> (B, T, N, F) -> convolution -> (B, c_out*T, N, 1) -> (B, c_out*T, N) -> (B, N, T)
            astgcn_output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)
            astgcn_outputs.append(astgcn_output)
        # astgcn_output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)

        # astgcn_outputs를 하나의 tensor로 concatenate
        astgcn_combined = torch.cat(astgcn_outputs, dim=2)

        # concat outputs of astgcn and lstm and operation yn for predict time
        combined = torch.cat((astgcn_combined, lstm_out), dim=2)

        # Apply final fully connected layer
        output = self.final_fc(F.relu(combined))  # (B, N, 1)

        return output

