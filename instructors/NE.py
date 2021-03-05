from itertools import chain
import torch
import config as cfg
from torch.nn.functional import relu, cosine_similarity
from torch import nn
import torch.optim as optimizer
from instructors.base import UILIns
from models.gnn import TriGNN
import numpy as np
from utils.general import split_list
from utils.torch_extension import cosine, bi_hit_x


class NEIns(UILIns):
    """
    Neighborhood Enhancement Component.
    """
    def __init__(self, emb: list, nei: dict, candidate: np.ndarray, idx: list, k: int) -> None:
        """
        Args:
            emb: user embeddings
            nei: a dict containing potentially matched user pairs and their splitted neighborhoods
            candidate: candidate users
            idx: ground truth user pairs for training and testing
            k: number of matched candidates, default: 30
        """
        super(NEIns, self).__init__(idx, k)
        self.nei = nei
        self.candidate = candidate
        self.emb = [e.cuda() for e in emb]
        self.sims = cosine(self.emb[0], self.emb[1])
        self.gnn = TriGNN(cfg.dim).cuda()
        self.opt = optimizer.Adam(
            self.gnn.parameters(), lr=cfg.lr
        )
        self.sim = cosine_similarity
        self.mse = nn.MSELoss()

    def get_emb(self):
        return self.emb

    @staticmethod
    def row_sample(mat: np.ndarray, k: int) -> np.ndarray:
        """
        Get k random samples for each row of a candidate matrix.

        Args:
            mat: (m, n), each row contains n candidate users
            k: number of samples

        Return:
            a matrix containing sampled candidates
        """
        m, n = mat.shape
        col = np.random.randint(n, size=(m, k))
        row = np.arange(m).reshape(-1, 1).repeat(k, 1)
        return mat[row, col]

    @staticmethod
    def ns2es(key: tuple, value: list) -> list:
        """
        Args:
            key: user pair, (i, j)
            value: splitted neighborhoods

        Return:
            list of edges
        """
        return [[(u, v) for v in vs] for u, vs in zip(
            (key[0], key[0], key[1], key[1]), value
        )]

    def get_edges(self, i: list, j: list, table: dict) -> list:
        """
        Get neighbors of given users.

        Argss:
            i/j: users from the source/target network
            table: a dict containing splitted neighborhoods

        Return:
            list of edges for graph neural networks to operate on
        """
        keys = list(zip(i, j))
        values = map(table.get, keys)
        edges = list(zip(*map(self.ns2es, keys, values)))
        edges = [list(chain(*e)) for e in edges]
        return [torch.tensor(e).t().cuda() for e in edges]

    def train_gnn(self):
        """
        Train the neighborhood enhancement component within a single epoch.
        """
        train, _ = self.idx
        left, right = list(zip(*train))
        # get negative samples
        negs = self.row_sample(self.candidate[left, 1:], cfg.neg)
        # concatenate positive pairs and negative pairs to form a dataset
        data = np.hstack((np.array(train), negs))
        # construct batches
        batches = split_list(data, cfg.batch_size)
        N = len(batches)
        loss_c = torch.tensor(0.).cuda()
        for batch in batches:
            # positve labels
            ones = torch.ones(len(batch)).cuda()
            # negative labels
            zeros = torch.zeros(len(batch)).cuda()
            x_s, x_t = self.get_emb()
            loss = torch.tensor(0.).cuda()

            # evalute loss on positive samples
            i_pos = batch[:, 0].tolist()
            j_pos = batch[:, 1].tolist()
            edges = self.get_edges(i_pos, j_pos, self.nei)
            y_s, y_t = self.gnn(x_s, x_t, i_pos, j_pos, edges)
            loss = loss + self.mse(relu(self.sim(y_s, y_t)), ones)
            # evaluate loss on negative samples, one at a time
            for k in range(2, cfg.neg + 2):
                j_neg = batch[:, k].tolist()
                edges = self.get_edges(i_pos, j_neg, self.nei)
                y_s, y_t = self.gnn(x_s, x_t, i_pos, j_neg, edges)
                loss = loss + self.mse(relu(self.sim(y_s, y_t)), zeros)
            self.optimize(self.opt, loss)
            loss_c += loss / (1 + cfg.neg)
        return loss_c / N

    def eval_k(self) -> list:
        """
        Evaluate the precision@k and hit-precision@k w.r.t. different mixture coefficients weighting the importance of neighborhood similarites.

        Return:
            hits: list of hit-precision@k w.r.t. different mixture coefficients
        """
        x_s, x_t = self.get_emb()
        m, n = self.sims.shape
        sims_nei = torch.zeros(m, n).cuda()
        i = [k for k in range(m)]
        size = self.candidate.shape[1]
        candidate = self.candidate.T.tolist()
        with torch.no_grad():
            # WARNING: this part is hard to compute in parallel and it's time-consuming
            for k in range(size):
                # indices of candidates
                j = candidate[k]
                # indices of neighbors of i and j
                edges = self.get_edges(i, j, self.nei)
                # neighborhood embeddings
                y_s, y_t = self.gnn(x_s, x_t, i, j, edges)
                # assign neighborhood similarities
                sims_nei[i, j] = self.sim(y_s, y_t)
        k = torch.tensor(30).cuda()
        hits = []

        for alpha in range(10):
            # mixture coefficient alpha ranges in {0., 0.1, ..., 0.9}
            # make a copy of sims to avoid modifying the values of sims
            sims_cp = self.sims.clone()
            alpha *= 0.1
            print('sims + sims_nei * {:.2f}'.format(alpha))
            sims_both = sims_cp + sims_nei * alpha
            sims_both /= (1 + alpha)

            train, test = self.idx
            coverage, hit_p = bi_hit_x(sims_both, k, train)
            self.log.info('Train Coverage {:.4f} | Hit {:.4f}'.format(
                coverage, hit_p
            ))
            row, col = [list(i) for i in zip(*train)]
            sims_both[row] = 0.
            sims_both[:, col] = 0.
            coverage, hit_p = bi_hit_x(sims_both, k, test)
            self.log.info('Test Coverage {:.4f} | Hit {:.4f}'.format(
                coverage, hit_p
            ))
            hits.append(hit_p)
        return hits

    def train(self):
        for epoch in range(1, cfg.epochs + 1):
            loss = self.train_gnn()
            self.log.info('epoch {:03d} loss {:.4f}'.format(epoch, loss))
            _ = self.eval_k()
