import torch
from torch import nn
from torch import Tensor
from scipy.sparse import csr_matrix


class NSLoss(nn.Module):
    """
    Negative sampling loss
    """
    def __init__(self, sim=None, mono=None, loss=None):
        """
        Args:
            sim: a similarity function
            mono: a monotonic function that map the similarity to a valid domain
            loss: a criterion measuring the predicted similarities and the ground truth labels
        """
        super(NSLoss, self).__init__()
        if sim is not None:
            self.sim = sim
        else:
            self.sim = self.inner_product
        if mono is None:
            self.mono = self.identity
        else:
            self.mono = mono
        if loss is not None:
            self.loss = loss
        else:
            self.loss = torch.nn.MSELoss()
            # self.loss = torch.nn.BCEWithLogitsLoss()

    @staticmethod
    def sample(neg: int, probs: Tensor, batch_size: int, scale: int = 16) -> Tensor:
        """
        Get indices of negative samples w.r.t. given probability distribution.

        Args:
            neg: number of negative samples
            probs: a probability vector for a multinomial distribution
            batch_size: batch size
            scale: maximum index, valid only when probs is None, leading to a uniform distribution over [0, scale - 1]

        Return:
            a LongTensor with shape [neg, batch_size]
        """
        assert neg > 0
        if probs is None:
            idx = torch.Tensor(batch_size * neg).uniform_(0, scale).long()
        else:
            if not isinstance(probs, torch.Tensor):
                probs = torch.tensor(probs)
            idx = torch.multinomial(probs, batch_size * neg, True)
        return idx.view(neg, batch_size)

    @staticmethod
    def inner_product(x: Tensor, y: Tensor) -> Tensor:
        """
        Calculate the pairwise inner product between two sets of vectors.
        """
        return x.mul(y).sum(dim=1, keepdim=True)

    @staticmethod
    def identity(x: Tensor) -> Tensor:
        return x

    @staticmethod
    def get_weights(i_s: Tensor, i_t: Tensor, weights: csr_matrix = None) -> Tensor:
        """
        Given indices, get corresponding weights from a weighted adjacency matrix.

        Args:
            i_s: row indices
            i_t: column indices
            weights: a weighted adjacency matrix, a sparse matrix is preferred as it saves memory

        Return:
            a weight vector of length len(i_s)
        """
        if weights is None:
            return torch.ones(len(i_s))
        else:
            i_s = i_s.tolist()
            i_t = i_t.tolist()
            weights = weights[i_s, i_t]
            return torch.FloatTensor(weights).squeeze()

    def get_xy(self, *inputs):
        # s for source and t for target
        """
        Calculate the pairwise similarities between two set of vectors in the common space.
        inputs contains a pack of arguements.

        Args:
            f_s/f_t: Tensor, [N, dim], user embedding matrix from the source/target network
            i_s/i_t: LongTensor, [batch], user indices from the source/target network
            map_s/map_t: MLP, mappings that map users from source/target network to the common space
            num: number of negative samples, int
            probs: a probability vector for the negative sampling distribution
            weights: a sparse weighted adjacency matrix

        Return:
            a similarity vectors and its corresponding ground truth labels.
        """
        f_s, f_t, i_s, i_t, map_s, map_t, \
            num, probs, weights = inputs
        x_s, x_t = f_s[i_s], f_t[i_t]
        if len(x_t.shape) > 2:
            x_t = x_t.mean(dim=1)
        # map x_s, x_t to the common space
        x_s, x_t = map_s(x_s), map_t(x_t)
        # calculate the pairwise similarities, stand for positive samples
        pos = self.sim(x_s, x_t).view(-1)
        y_pos = self.get_weights(i_s, i_t, weights)
        if num > 0:
            # get negative samples
            i_n = self.sample(num, probs, len(i_s), len(f_t))
            # calculate the pairwise similarities, stand for negative samples
            neg = torch.stack([self.sim(x_s, map_t(f_t[i])) for i in i_n])
            neg = neg.view(-1)
            x = torch.cat([pos, neg])
            if weights is None:
                y_neg = torch.zeros(len(neg))
            else:
                y_neg = torch.stack([self.get_weights(i_s, i, weights) for i in i_n]).view(-1)
            # ground truth labels
            y = torch.cat([y_pos, y_neg])
        else:
            x = pos
            y = y_pos
        return x, y

    def forward(self, *inputs):
        # x: similarities, [batch]
        # y: labels, [batch]
        x, y = self.get_xy(*inputs)
        # map x to a valid domain, e.g. a positive number
        x = self.mono(x)
        if x.is_cuda:
            y = y.cuda()
        return self.loss(x, y)
