import torch
from torch import Tensor


def to_device(x, DEVICE='cuda'):
    if DEVICE == 'cuda' and torch.cuda.is_available():
        return x.cuda()
    else:
        return x.cpu()


def cosine(xs: Tensor, ys: Tensor, epsilon: float = 1e-8) -> Tensor:
    """
    Efficiently calculate the pairwise cosine similairties between two set of vectors.

    Args:
        xs: feature matrix, [N, dim]
        ys: feature matrix, [M, dim]
        epsilon: a small number to avoid dividing by zero

    Retrun:
        a [N, M] matrix of pairwise cosine similairties
    """
    mat = xs @ ys.t()
    x_norm = xs.norm(2, dim=1) + epsilon
    y_norm = ys.norm(2, dim=1) + epsilon
    x_diag = (1 / x_norm).diag()
    y_diag = (1 / y_norm).diag()
    return x_diag @ mat @ y_diag


def bi_hit_x(sims: Tensor, k: int, test: tuple) -> tuple:
    """
    Calculate the average precision@k and hit_precision@k from two sides, i.e., source-to-target and target-to-source.

    Args:
        sims: similarity matrix
        k: number of candidates
        test: index pairs of matched users, i.e., the ground truth

    Return:
        coverage: precision@k
        hit: hit_precison@k
    """
    row, col = [list(i) for i in zip(*test)]
    target = sims[row, col].reshape((-1, 1))
    left = sims[row]
    right = sims.t()[col]
    # match users from source to target
    c_l, h_l = score(left, target, k)
    # match users from target to source
    c_r, h_r = score(right, target, k)
    # averaging the scores from both sides
    return (c_l + c_r) / 2, (h_l + h_r) / 2


def score(mat: Tensor, target: Tensor, k: int) -> tuple:
    """
    Calculate the average precision@k and hit_precision@k from while matching users from the source network to the target network.

    Args:
        sims: similarity matrix
        k: number of candidates
        test: index pairs of matched users, i.e., the ground truth

    Return:
        coverage: precision@k
        hit: hit_precison@k
    """
    # number of users with similarities larger than the matched users
    rank = (mat >= target).sum(1)
    # rank = min(rank, k + 1)
    rank = rank.min(torch.tensor(k + 1).cuda())
    tmp = (k + 1 - rank).float()
    # hit_precision@k
    hit_score = (tmp / k).mean()
    # precision@k
    coverage = (tmp > 0).float().mean()
    return coverage, hit_score
