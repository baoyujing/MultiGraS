import numpy as np
import scipy.sparse as sp


def pad_adj(adj, n, max_len):
    diag = np.concatenate([np.ones([1, n]), np.zeros([1, max_len - n])], axis=1)
    adj = diag * adj * diag.T
    return adj


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).toarray()
