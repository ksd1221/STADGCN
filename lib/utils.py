import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from .metrics import masked_mape_np
from scipy.sparse.linalg import eigs
from scipy.linalg import eigvalsh
from scipy.linalg import fractional_matrix_power

# keshihua
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker


def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}

    '''

    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials


def calculate_stad_matrix(traffic_flow_vectors):
    """
    Calculate the STAD matrix based on traffic flow vectors using cosine distance.

    Parameters:
    traffic_flow_vectors (np.array): A matrix where each row represents a traffic flow vector at a node.

    Returns:
    np.array: A matrix representing the spatial-temporal aware distances (STAD) between nodes.
    """
    # Compute the cosine similarity between traffic flow vectors
    # Note: cosine similarity is the dot product of the l2-normalized traffic flow vectors
    normed_vectors = traffic_flow_vectors / np.linalg.norm(traffic_flow_vectors, axis=1, keepdims=True)
    cosine_similarity = np.dot(normed_vectors, normed_vectors.T)

    # Convert the cosine similarity to cosine distance
    cosine_distance = 1 - cosine_similarity

    # Since the diagonal represents self-similarity, we set it to zero
    np.fill_diagonal(cosine_distance, 0)

    return cosine_distance


def create_strg_from_stad(stad_matrix, sparsity_level):
    """
    Create a Spatial-Temporal Relevance Graph (STRG) from the STAD matrix.

    Parameters:
    stad_matrix (np.array): A matrix representing the spatial-temporal aware distances (STAD) between nodes.
    sparsity_level (float): The percentage of the top distances to be considered relevant.

    Returns:
    np.array: A binary adjacency matrix representing the STRG.
    """
    # Determine the number of relevant entries based on the sparsity level
    n_relevant = int(np.ceil(sparsity_level * stad_matrix.size))

    # Flatten the STAD matrix to sort the distances
    flat_stad = stad_matrix.flatten()

    # Find the threshold distance that separates the top sparsity_level distances
    sorted_indices = np.argsort(flat_stad)
    threshold_index = sorted_indices[n_relevant]
    threshold_value = flat_stad[threshold_index]

    # Create the STRG by setting distances below the threshold to 1, others to 0
    strg_matrix = (stad_matrix <= threshold_value).astype(int)

    # Ensure the diagonal is zero as no node should be connected to itself in STRG
    np.fill_diagonal(strg_matrix, 0)

    return strg_matrix