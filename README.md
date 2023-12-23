# STAD-GCN: Spatial-Temporal Attention-based Dynamic Graph Convolutional Network for Market Price Prediction
we introduce STAD-GCN (Spatial-Temporal Attention-based Dynamic Graph Convolutional Network), a pioneering model that integrates temporal and spatial elements within a graph-based framework. This fusion amplifies our understanding of market dynamics and the factors influencing price determination. 

The below figure is the Architectural Overview of STAD-GCN: Multimodal Fusion with ASTGCN and LSTM for Retail Gasoline Price Prediction is outlined as follows: (A) Preprocessing: Input data is preprocessed based on its characteristics. It is formatted as a graph-based time series and routed to (B), while also treated as a standard time series sent to (C). (B) Spatial-Temporal (ST) Blocks: Within (B), the data undergoes processing via two ST blocks, correlating to the number of nodes. These blocks handle the data, ultimately stacking the resultant vectors. (C) Final Prediction: The vectors from the previous step are directed to (D) for final prediction. (D) Prediction Generation: In this phase, the received vectors are concatenated and processed through a fully-connected layer, generating the ultimate predicted values.

# Requirements
```
python==
cuda==
```
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
