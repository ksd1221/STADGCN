# MAGNET
Market Price Prediction with Attention-based Graph Network for Spatial-Temporal Dynamics, pytorch version

# Reference

# Configuration
Due to a known bug causing conflicts between DGL and CUDA version 12.1, it is recommended to set the CUDA version to 11.8.
You can download a CUDA-compatible version of DGL using the following code:
```
!pip install  dgl -f https://data.dgl.ai/wheels/cu118/repo.html
!pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html
```

# Datasets
We sourced our data from 'Opinet'(https://www.opinet.co.kr/user/main/mainView.do), the official website of the South Korea National Oil Corporation.

# Train and Test
## Model Parameters
- device: CPU/GPU
- nb_block: number of ASTGCN blocks in the submodule
- in_channels: number of input features
- K: Degree of Chebyshev polynomial
- nb_chev_filter: number of output filters in Chebyshev convolution
- nb_time_filter: number of output filters in temporal convolution
- time_strides: stride of the temporal convolution
- cheb_polynomials: pre-calculated Chebyshev polynomials
- num_for_predict: number of steps to predict
- len_input: length of the input sequence (T)
- num_of_vertices: number of nodes (N)
- lstm_hidden_dim: number of hidden dimension of LSTM module
- lstm_num_layers: number of LSTM layers
- cardinalities: cardinality list of categorical variables
