import argparse
import multiprocessing as mp
import torch
from instructors.NE import NEIns
from utils.general import read_pickle, write_pickle
import config as cfg
import numpy as np
import time
import os

from utils.torch_extension import cosine

# os.chdir('../')
# sims: user similarity matrix
# gs: indices of neighbors in the source and target networks, [[[i, ...], ...], [[j, ...], ...]]
# sims and gs are set as global variables to allow multi-processing pre-processing in the function test_row_nei_index_mul.
sims, gs = None, None


def row_k_max(mat: np.ndarray, k: int, sort: bool = False) -> tuple:
    """
    Find the rowwise top-k elements of a given matrix.
    Transpose mat to get the column version.

    Args:
        mat: similarity matrix
        k: number of candidates, default: 250
        sort: return descending rows or not, default: False

    Return:
        matrix: a filtered similarity matrix with rowwise top-k elements
        col: column index of rowwise top-k elements
    """
    # find indices of rowwise top-k elements
    col = np.argpartition(mat, -k, -1)[:, -k:]
    m, n = mat.shape
    row = np.arange(m).reshape(-1, 1).repeat(k, 1)
    if sort:
        # sort the elements and indices
        tmp = - mat[row, col]
        idx = np.argsort(tmp, 1)
        col = col[row, idx]
    matrix = np.zeros((m, n))
    matrix[row, col] = mat[row, col]
    return matrix, col


def first_match(mat: np.ndarray, threshold: float, strict: bool = True) -> tuple:
    """
    Given a similarity matrix, find matched users under the one-to-one_{<=} constraint, i.e., each user find at most one matched user.

    Args:
        mat: similarity matrix
        threshold: a threshold for the minimum similairty of matched user pairs
        strict: assert each user find at most one matched user, default: True

    Return:
        mat: a similarity matrix masking the first round of matched user pairs, possibly userful in multiple rounds of matching
        match: a similarity matrix where the similarity of matched users are nonzero
    """
    from utils.np_extensions import np_relu
    # similarities of matched user pairs should be above a given threshold
    rowmax = np.maximum(np.max(mat, axis=0, keepdims=True), threshold)
    colmax = np.maximum(np.max(mat, axis=1, keepdims=True), threshold)
    rowmatch = np.sign(mat - rowmax) + 1
    colmatch = np.sign(mat - colmax) + 1
    # (i, j) is a matched user pair if j is the most similar user to i within the target network and vice versa
    # if sim(i, j) = sim(i, k) = max_i' sim(i, i'), both j and k are matched users for i, which is not expected under the one-to-one constraint
    match = rowmatch * colmatch * mat
    if strict:
        # filter rows with more than two matched user pairs
        sign = (match > 0).sum(axis=0, keepdims=True)
        match = match * (0 < sign) * (sign < 2)
        sign = (match > 0).sum(axis=1, keepdims=True)
        match = match * (0 < sign) * (sign < 2)
    if match.any():
        # mask matched user pairs
        mat = mat - np.sum(match, axis=0, keepdims=True)
        mat = mat - np.sum(match, axis=1, keepdims=True)
        mat = np_relu(mat)
    return mat, match


def sim_nei_index(pair: tuple) -> tuple:
    """
    Split the neighborhoods of a given user pairs into two disjoint subsets, similar neighbors and dissimilar neighbors.

    Args:
        pair: a user pair, (i, j)

    Return:
        i_s, j_s: indices of similar users from the source/target networks
        i_t, j_t: indices of dissimilar users from the source/target networks
    """
    i, j = pair
    # neighbors of i and j
    nei_i, nei_j = gs[0][i], gs[1][j]
    # sub similiarity matrix w.r.t. the neighbors of i and j
    sub_sims = sims[nei_i][:, nei_j]
    # find similar neighbor pairs
    _, match = first_match(sub_sims, 0)
    i_s, j_s = [i.tolist() for i in match.nonzero()]
    # indices of similar neighbors
    i_s = [nei_i[k] for k in i_s]
    j_s = [nei_j[k] for k in j_s]
    # indices of dissimilar neighbors
    i_t = list(set(nei_i) - set(i_s))
    j_t = list(set(nei_j) - set(j_s))
    return i_s, i_t, j_s, j_t


def test_row_nei_index_mul():
    """
    Find potentially matched user pairs and split their neighborhood into two disjoint subsets, containing similar neighbors and dissimilar neighbors.
    """
    global sims, gs
    path = 'data/sims_{:.1f}.pkl'.format(cfg.ratio)
    # compute the user similarity matrix if not existed.
    if not os.path.exists(path):
        # load the pre-trained user embeddings of users in the source/target network.
        left, right = read_pickle('data/emb_{:.1f}.pkl'.format(cfg.ratio))
        # compute the pairwise cosine similarities among users from the source and target networks.
        sims = cosine(left, right)
        # save the similarity matrix as a dense numpy array.
        sims = sims.detach().numpy()
        write_pickle(sims, path)
    else:
        sims = read_pickle(path)
    train, test = read_pickle('data/train_test_{:.1f}.pkl'.format(cfg.ratio))
    row_train, col_train = list(zip(*train))
    # elements of the similarity matrix are in [-1, 1]
    # assign the largest value 1 to matched user pairs in the training set.
    sims[row_train, col_train] = 1.
    # candidate users for neighborhood enhancement.
    # candidate[:len(train), 0] are ground truth on training set.
    k = 250
    sims, candidate = row_k_max(sims, k)

    # potentially matched user pairs
    left, right = [i.tolist() for i in sims.nonzero()]
    # adjacency list
    gs = read_pickle('results/adj_list.pkl')
    # use multiple processes to find similar and dissimilar neighbors
    pool = mp.Pool(8)
    t = time.time()
    pairs = list(zip(left, right))
    idx = list(pool.map(sim_nei_index, pairs))
    pool.close()
    pool.join()
    print('time: {:.2f}'.format(time.time() - t))
    # idx: indices of matched and unmatched neighbors
    # pairs: potentially matched user pairs, [(i, j), ...]
    # candidate: candidate users, same thing as pairs, [[i, ...], ...]
    write_pickle(idx, 'results/col_nei_idx_{:.1f}.pkl'.format(cfg.ratio))
    write_pickle(pairs, 'results/pairs_{:.1f}.pkl'.format(cfg.ratio))
    write_pickle(candidate, 'results/candidate_{:.1f}.pkl'.format(cfg.ratio))


def load_data():
    """
    Load data for neighborhood enhancement.

    Return:
        emb: user embeddings
        nei: potentially matched user pairs and their splitted neighborhoods
        candidate: candidate users
        train_test: ground truth user pairs for training and testing
    """
    path = 'results/nei_{:.1f}.pkl'.format(cfg.ratio)
    if not os.path.exists(path):
        pairs = read_pickle('results/pairs_{:.1f}.pkl'.format(cfg.ratio))
        nei_idx = read_pickle('results/col_nei_idx_{:.1f}.pkl'.format(cfg.ratio))
        # key: pairs, value: idx
        # building the dict is time-consuming
        nei = dict(zip(pairs, nei_idx))
        write_pickle(nei, path)
    else:
        nei = read_pickle(path)
    return \
        read_pickle('results/emb_{:.1f}.pkl'.format(cfg.ratio)), \
        nei, \
        read_pickle('results/candidate_{:.1f}.pkl'.format(cfg.ratio)), \
        read_pickle('data/train_test_{:.1f}.pkl'.format(cfg.ratio))


def match(emb: list, nei: dict, candidate: np.ndarray, idx: list, k: int) -> None:
    """
    Main function for neighborhood enhancement.

    Args:
        emb: user embeddings
        nei: a dict containing potentially matched user pairs and their splitted neighborhoods
        candidate: candidate users
        idx: ground truth user pairs for training and testing
        k: number of matched candidates, default: 30
    """

    # set cuda device, multiple gpus are not supported yet.
    torch.cuda.set_device(cfg.cuda)
    # initialize an instructor that makes necessary reports.
    ins = NEIns(emb, nei, candidate, idx, k)
    ins.train()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', help='cuda id.', default=0, type=int)
    parser.add_argument('--epochs', help='number of training epochs.', default=40, type=int)
    parser.add_argument('--model', help='name of the model, used in logging.', default='NE', type=str)
    parser.add_argument('--ratio', help='ratio of training set, e.g. 0.1-0.9.', default=cfg.ratio, type=float)
    return parser.parse_args()


if __name__ == '__main__':
    # pre-processing, time-cosuming, suggested to run and save first, and comment the following line when things are done.
    # test_row_nei_index_mul()
    # neighborhood enhancement
    args = get_args()
    print(args)
    cfg.init_args(args)
    match(*load_data(), k=cfg.k)
