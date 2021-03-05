import os

from scipy.sparse import csr_matrix
from torch import Tensor
from torch.optim.optimizer import Optimizer

import config as cfg
from utils.general import write_pickle
import logging
from utils.log import create_logger
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from utils.np_extensions import clamp_mat
from utils.torch_extension import cosine, bi_hit_x


class DataWrapper(Dataset):
    """
    A wrapper for torch.utils.data.Dataset.
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


class Instructor:
    """
    A base class for training.
    """
    def __init__(self):
        self.log = create_logger(
            __name__, silent=False,
            to_disk=True, log_file=cfg.log)

    def rename_log(self, filename: str) -> None:
        """
        Rename the log file with a new filename.
        """
        logging.shutdown()
        os.rename(cfg.log, filename)

    @staticmethod
    def optimize(opt: Optimizer, loss: Tensor) -> None:
        """
        Optimize the parameters based on the loss and the optimizer.

        Args:
            opt: optimizer
            loss: loss, a scalar
        """
        opt.zero_grad()
        loss.backward()
        opt.step()

    @staticmethod
    def load_data(input, batch_size: int) -> DataLoader:
        """
        Return a dataloader given the input and the batch size.
        """
        data = DataWrapper(input)
        batches = DataLoader(
            data,
            batch_size=batch_size,
            shuffle=True)
        return batches

    @staticmethod
    def early_stop(current: Tensor, results: Tensor,
                   size: int = 3, epsilon: float = 5e-5) -> float:
        """
        Early stop the training when a given indicator tends to converge.

        Args:
            current: current value of the indicator
            results: historical values of the indicator
            size: windows size of indicator values
            epsilon: an absolute tolerance number

        Return:
            True if the absolute difference surpsses the tolerance number
        """

        results[:-1] = results[1:]
        results[-1] = current
        assert len(results) == 2 * size
        pre = results[:size].mean()
        post = results[size:].mean()
        return abs(pre - post) > epsilon


class UILIns(Instructor):
    """
    A base class for training a UIL model.
    """
    def __init__(self, idx: list, k: int) -> None:
        """
        Args:
            idx: ground truth user pairs for training and testing
            k: number of candidates
        """
        super(UILIns, self).__init__()
        self.idx = idx
        self.k = k

    def get_emb(self):
        """
        Get user embeddings.
        """
        raise NotImplementedError

    @staticmethod
    def sim_pairwise(xs: Tensor, ys: Tensor) -> Tensor:
        """
        Default similarity function: cosine.
        """
        return cosine(xs, ys)

    def save_emb(self, path: str) -> None:
        """
        Save user embeddings.

        Args:
            path: path to save user embeddings
        """
        embs = [i.cpu() for i in self.get_emb()]
        write_pickle(embs, path)

    @staticmethod
    def add_assist(mat: csr_matrix, exponent: float = 3 / 4, percent: float = cfg.percent) -> None:
        """
        Given a similarity matrix, create weights for negative sampling and indices of similar users.

        Args:
            mat: similarity matrix
            exponent: a coefficient to downplay the similarities to create negative sampling weights, default: 3/4 (as suggested by word2vec)
            percent: percent of users to filter, range in [0, 100]

        Return:
            pairs: user pairs with high similairties
            weights: negative sampling weights
            mat: similarity matrix
        """
        if not isinstance(mat, np.ndarray):
            mat = mat.toarray()
        weights = np.abs(mat).sum(axis=0) ** exponent
        # total = weights.sum()
        # weights = np.round(weights / total * 1e6)
        clamp = clamp_mat(mat, percent)
        pairs = [i.tolist() for i in clamp.nonzero()]
        pairs = list(zip(*pairs))
        return pairs, weights, csr_matrix(mat)

    def eval_bi(self, mask: Tensor = None, default: float = 0.) -> float:
        """
        Evaluation precision@k and hit-precision@k in the training set and testing set.

        Args:
            mask: a matrix masking known matched user pairs
            default: default similarity for matched user pairs in the training set

        Return:
            hit-precision@k
        """
        with torch.no_grad():
            f_s, f_t = self.get_emb()
            sims = self.sim_pairwise(f_s, f_t)
            if mask is not None:
                sims = sims * mask
            train, test = self.idx
            coverage, hit_p = bi_hit_x(sims, self.k, train)
            self.log.info('Train Coverage {:.4f} | Hit {:.4f}'.format(
                coverage, hit_p
            ))
            row, col = [list(i) for i in zip(*train)]
            # mask similarities of matched user pairs in the training set
            sims[row] = default
            sims[:, col] = default
            coverage, hit_p = bi_hit_x(sims, self.k, test)
            self.log.info('Test Coverage {:.4f} | Hit {:.4f}'.format(
                coverage, hit_p
            ))
        return hit_p

    def eval_within(self):
        """
        An unstandard function evaluating precision@k and hit-precision@k in the training set and testing set separately.
        Not recommended!!!
        """
        with torch.no_grad():
            f_s, f_t = self.get_emb()
            sims = self.sim_pairwise(f_s, f_t)
            train, test = self.idx
            row, col = [list(i) for i in zip(*train)]
            sims_train = sims[row][:, col]
            idx_train = [(i, i) for i in range(len(train))]
            coverage, hit_p = bi_hit_x(sims_train, self.k, idx_train)
            self.log.info('Train Coverage {:.4f} | Hit {:.4f}'.format(
                coverage, hit_p
            ))
            row, col = [list(i) for i in zip(*test)]
            sims_test = sims[row][:, col]
            idx_test = [(i, i) for i in range(len(test))]
            coverage, hit_p = bi_hit_x(sims_test, self.k, idx_test)
            self.log.info('Test Coverage {:.4f} | Hit {:.4f}'.format(
                coverage, hit_p
            ))
        return hit_p

    def infer(self, targets: list, k: int = None, default: float = 0.) -> dict:
        """
        Given target users, find their top-k candidates using the pre-trained user embeddings.

        Args:
            targets: list of target users
            k: number of candidates
            default: default  similarity of known matched user pairs, default: 0

        Return:
            top-k candidats of each target user
        """
        with torch.no_grad():
            if k is None:
                k = self.k
            f_s, f_t = self.get_emb()
            x_s = f_s[targets]
            sims = self.sim_pairwise(x_s, f_t)
            labels = self.idx[0] + self.idx[1]
            row, col = [list(i) for i in zip(*labels)]
            # mask known matched users
            sims[:, col] = default
            _, rank = torch.topk(sims, k)
            rank = rank.tolist()
            candidates = {targets[i]: rank[i] for i in range(len(targets))}
        return candidates
