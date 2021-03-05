"""
Useful functions based on numpy
"""
import numpy as np
from scipy.sparse import csr_matrix


def clamp_mat(mat: np.ndarray, percent: float, up: bool = True) -> np.ndarray:
    """
    Clamp a matrix over its rows and columns to a percentile.

    Args:
        mat: matrix
        percent: [0, 100]
        up: if True, clamp the matrix downstairs, else clamp it upstairs.

    Return:
        a clampped matrix
    """
    rows = np.percentile(mat, percent, axis=1, keepdims=True)
    cols = np.percentile(mat, percent, axis=0, keepdims=True)
    if up:
        return mat * (mat >= rows) * (mat >= cols)
    else:
        return mat * (mat <= rows) * (mat <= cols)


def pair2sparse(pairs: list, shape: tuple) -> csr_matrix:
    """
    Convert a edge list to a sparse adjacency matrix.

    Args:
        pairs: edge list
        shape: [N, M]

    Returns:
        a sparse adjacency matrix
    """
    s, t = list(zip(*pairs))
    mat = csr_matrix((np.ones(len(s)), (s, t)), shape=shape)
    return mat


def np_relu(x: np.ndarray) -> np.ndarray:
    """
    A numpy implementation of ReLU.
    """
    return x * (x > 0)
