import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Spatial_Attention_layer(nn.Module):
    """
    compute spatial attention scores
    """
    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Spatial_Attention_layer, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(num_of_timesteps).to(DEVICE)) # parameter matrix of dimension [T]
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_timesteps).to(DEVICE)) # parameter matrix of dimension [F, T]
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE)) # parameter matrix of dimension [F]
        self.bs = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices).to(DEVICE)) # Bias matrix
        self.Vs = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices).to(DEVICE)) # Vertex matrix of dimension [N, N]

    def forward(self, x):
        """
        Calculate spatial attention scores
            1. Spatial attention left hand side
            2. Spatial attention right hand side
            3. Final product for spatial attention
        :param x: (batch_size, N, F_in, T)
        :return: (B, N, N)
        """

        # Implementing the Spatial attention mechanism
        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)   # (b, N, F, T)(T) -> (b, N, F)(F, T) -> (b, N, T)
        rhs = torch.matmul(self.W3, x).transpose(-1, -2)    # (F)(b, N, F, T) -> (b, N, T) -> (b, T, N)
        product = torch.matmul(lhs, rhs)    # (b, N, T)(b, T, N) -> (B, N, N)

        # Apply sigmoid activation and matrix multiplication
        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs)) # (N, N)(B, N, N) -> (B, N, N)
        # Normalize the spatial attention scores using softmax
        S_normalized = F.softmax(S, dim=1)

        return S_normalized


class cheb_conv_withSAt(nn.Module):
    """
    K-order chebyshev graph convolution with Spatial Attention
    - Graph Laplacian의 전체 고유값 분해를 계산하지 않고도 푸리에 방법을 그래프 구조 데이터에 활용할 수 있게 하는 효율적인 방법 제공
    - Locality: 고차 Chebyshev polinomial을 사용하면 더 넓은 이웃을 고려할 수 있음. 이는 더 정확한 예측이나 표현을 만들기 위해 중요할 수 있음
        (K: degree of Chebyshev polinomial)
        1) 연산량과 메모리 사용량: 'K'가 크면 계산 복잡도와 메모리 사용량이 늘어남
        2) 노드 간의 상호 작용: 'K'가 크면 더 많은 '이웃' 노드를 고려함. 이는 노드 간의 상호 작용이 더 넓은 범위에서 발생할 수 있지만, Overfitting을 유발할 수도 있음
        3) 모델 성능: 'K' 값에 따라 모델의 성능이 달라질 수 있음
        4) 문제의 복잡성: 복잡한 문제일수록 더 높은 차수의 다항식을 필요로 할 수 있음
    - 계산 효율성, 유연성 등의 목적
    """

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        """
        :param K: int
            Order of Chebyshev polynomial
        :param cheb_polynomials: np.ndarray
            List of Chebyshev polynomials
        :param in_channels: int
            num of channels in the input sequence
        :param out_channels: int
            num of channels in the output sequence
        """
        super(cheb_conv_withSAt, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device

        # Theta parameters for each Chebyshev polynomial
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])

    def forward(self, x, spatial_attention):
        """
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :param spatial_attention: Spatial attention scores.
        :return: (batch_size, N, F_out, T)
        """
        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape
        outputs = []

        for time_step in range(num_of_timesteps):
            graph_signal = x[:, :, :, time_step]    # (b, N, F_in)
            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)    # (b, N, F_out)
            for k in range(self.K):
                T_k = self.cheb_polynomials[k]  # (N, N)
                T_k_with_at = T_k.mul(spatial_attention)    # Incorporate spatial attention, (N, N)*(N, N) = (N, N)
                theta_k = self.Theta[k]     # (in_channel, out_channel)
                rhs = T_k_with_at.permute(0, 2, 1).matmul(graph_signal) # (N, N)(b, N, F_in) = (b, N, F_in)
                output = output + rhs.matmul(theta_k)   # (b, N, F_in)(F_in, F_out) = (b, N, F_out)
            outputs.append(output.unsqueeze(-1)) # (b, N, F_out, 1)
        return F.relu(torch.cat(outputs, dim=-1))   # (b, N, F_out, T)


class Temporal_Attention_layer(nn.Module):
    """
    Compute temporal attention scores
    """
    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Temporal_Attention_layer, self).__init__()
        # Define paramters of the temporal attention mechanism
        self.U1 = nn.Parameter(torch.FloatTensor(num_of_vertices).to(DEVICE)) # parameter matrix of dimension [N]
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_vertices).to(DEVICE)) # parameter matrix of dimension [F, N]
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE)) # parameter matrix of dimension [F]
        self.be = nn.Parameter(torch.FloatTensor(1, num_of_timesteps, num_of_timesteps).to(DEVICE)) # Bias
        # Ve: 시간 차원에서의 attention score에 추가 가중치 부여하는 역할
        self.Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps).to(DEVICE)) # Vertex matrix of dimension [T, T]

    def forward(self, x):
        """
        :param x: (batch_size, N, F_in, T)
        :return: (B, T, T)
        """
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)
        # x: (B, N, F_in, T) => (B, T, F_in, N)
        # (B, T, F_in, N)(N) -> (B, T, F_in)
        # (B, T, F_in)(F_in, N) -> (B, T, N)

        rhs = torch.matmul(self.U3, x)  # (F)(B, N, F, T) -> (B, N, T)
        product = torch.matmul(lhs, rhs)    # (B, T, N)(B, N, T) -> (B, T, T)
        # Calculate the attention scores before normalization
        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be)) # (B, T, T)
        # Normalize the attention scores using softmax
        E_normalized = F.softmax(E, dim=1)
        return E_normalized


class DotProductAttention(nn.Module):
    def __init__(self):
        super(DotProductAttention, self).__init__()

    def forward(self, x, mask=None):
        batch_size, node_size, features = x.shape  # 입력 형태를 가져옴

        queries, keys, values = x, x.permute(0, 2, 1), x

        attention_scores = torch.matmul(queries, keys)/np.sqrt(node_size)
        attention_scores = F.softmax(attention_scores, dim=-1)

        attention_weighted_values = torch.matmul(attention_scores, values) # .transpose(1, 2)  # 결과를 원래 형태로 변환
        attention_weighted_values = attention_weighted_values.view(batch_size, node_size, features)

        return attention_weighted_values


class MultipleEmbeddings(nn.Module):
    def __init__(self, cardinalities):
        super(MultipleEmbeddings, self).__init__()
        self.embedding_layers = nn.ModuleList()
        self.embed_size_list = []

        for cardinality in cardinalities:
            # Rule of thumb for setting embedding size
            embed_size = min(50, (cardinality + 1) // 2)
            self.embed_size_list.append(embed_size)
            self.embedding_layers.append(nn.Embedding(cardinality, embed_size))

    def forward(self, x):
        embeddings = []
        for i, embedding_layer in enumerate(self.embedding_layers):
            # Apply embedding only to the i-th feature across all batches, nodes, and time
            feature_slice = x[:, :, i, :].long()
            embedding = embedding_layer(feature_slice)
            embeddings.append(embedding)
        # Concatenate along the last dimension
        return torch.cat(embeddings, dim=3).permute(0, 1, 3, 2)