import numpy as np
from scipy.sparse.linalg import eigs


def get_adjacency_matrix(distance_df, station_ids, criteria_d=0):
    """
    Motified STGCN ver.

    :param distance_df: DataFrame
        Station distances, data frame with three columns: [from, to, distance]
    :param station_ids: List
        List of sensor ids.
    :return: Array
        Adjacency matrix, distance matrix
    """

    # init dist_mx
    num_stations = len(station_ids)
    A = np.zeros((num_stations, num_stations), dtype=np.float32)
    dist_mx = np.zeros((num_stations, num_stations), dtype=np.float32)
    dist_mx[:] = np.inf

    # builds station id to index map
    station_id_to_ind = {}
    for i, station_id in enumerate(station_ids):
        station_id_to_ind[station_id] = i

    # fills cells in the dist_mx with distances
    for row in distance_df.values:
        dist_mx[station_id_to_ind[row[0]], station_id_to_ind[row[1]]] = row[2]  # distance
        if criteria_d == 0:
            A[station_id_to_ind[row[0]], station_id_to_ind[row[1]]] = 1  # connectivity
        elif row[2] <= criteria_d:    # if distance is shorter than criteria, then connectivity is true
            A[station_id_to_ind[row[0]], station_id_to_ind[row[1]]] = 1     # connectivity
        else:
            continue

    return A, dist_mx


def scaled_Laplacian(W):
    """
    Scaled Laplacian 값을 사용함으로써 수치 안정성 [-1, 1], 학습 효율성, 이론적 분석 용이성을 얻을 수 있음
    :param W: np.ndarray
        shape is (N, N), N is the num of vertices
    :return: np.ndarray
        shape is (N, N), scaled_Laplacian
    """

    assert W.shape[0] == W.shape[1] # check shape
    D = np.diag(np.sum(W, axis=1))
    L = D-W
    lambda_max = eigs(L, k=1, which='LR')[0].real   # maximum eigenvalue of Laplacian matrix L
    return (2*L)/lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    """
    compute a list of chebyshev polynomials from T_0 to T_{K-1}
    :param L_tilde: np.ndarray
        scaled Laplacian, shape(N, N)
    :param K: the maximum order of chebyshev polynomials
    :return: list(np.ndarray) A list containing the Chebyshev polynomials
        length: K, from T_0 to T_{K-1}
    """

    # Get the number of vertices (nodes) in the graphs
    N = L_tilde.shape[0]

    # Initialize the list of Chebyshev polynomials with T_0 and T_1
    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    # Compute the Chebyshev polynomials up to the (K-1)-th order
    for i in range(2, K):
        # Chebyshev polynomial recurrence relation: T_k(x) = 2*x*T_{k-1}(x)-T_{k-2}(x)
        cheb_polynomials.append(2*L_tilde*cheb_polynomials[i-1]-cheb_polynomials[i-2])

    return cheb_polynomials