from itertools import chain
import config as cfg
from instructors.base import UILIns
from utils.general import write_pickle
from utils.np_extensions import pair2sparse
from utils.torch_extension import to_device
import torch
from torch import nn
import torch.optim as opt
from models.base import MLP, Emb
from models.loss import NSLoss


class IFIns(UILIns):
    """
    Information Fusion Component.
    """
    def __init__(self, sims: dict, idx: list, k: int) -> None:
        """
        Args:
            sims: a dict containing intra-/inter-network similarity matrices
            idx: ground truth user pairs for training and testing
            k: number of candidates
        """
        super(IFIns, self).__init__(idx, k)
        # assist: {key: [pairs, weights, sim], ...}
        self.assist = self.sims_assist(sims)
        shape = self.get_shape(sims)
        if idx is not None:
            # construct a matrix indicating if two users are matched from ground truth user pairs in the training set
            mat = pair2sparse(idx[0], shape)
            self.assist['labels'] = self.add_assist(mat)

        self.model = nn.ModuleDict({
            'embs': Emb(shape, cfg.dim),
            'common': nn.ModuleList([
            MLP(cfg.dim, cfg.dim),
            MLP(cfg.dim, cfg.dim)
        ]),
            'intra': nn.ModuleList([
            MLP(cfg.dim, cfg.dim)
            for i in range(len(sims['intra']))
        ]),
            'inter': nn.ModuleList([
            MLP(cfg.dim, cfg.dim)
            for i in range(len(sims['inter']))
        ])
        })
        self.model = to_device(self.model)

        # the unified user embeddings embs and the mappings for intra-/inter-network similarity matrices reconstruction are jointly learnt
        self.opt_emb = opt.Adam(
            chain(self.model['embs'].parameters(),
                  self.model['intra'].parameters(),
                  self.model['inter'].parameters()
                  ),
            lr=cfg.lr
        )
        # the mappings that map user to a common space are trained separately to ensure stable learning
        self.opt_labels = opt.Adam(
            self.model['common'].parameters(),
            lr=cfg.lr
        )

        self.loss = NSLoss(
            sim=nn.CosineSimilarity(),
            mono=nn.ReLU(),
            loss=nn.MSELoss()
        )

    @staticmethod
    def get_shape(sims: dict) -> tuple:
        """
        Get number of users in the source/target network.

        Args:
            sims: a dict containing intra-/inter-network similarity matrices

        Return；
            shape: (m, n), where m and n are the number of users in the source network and target network, respectively
        """
        inter = sims['inter']
        if len(inter) == 0:
            intra = sims['intra']
            if len(intra) == 0:
                raise ValueError('sims is empty!')
            else:
                shape = (intra[0][0].shape[0], intra[0][1].shape[0])
        else:
            shape = inter[0].shape
        return shape

    def sims_assist(self, sims: dict) -> dict:
        """
        Create assisted variables for training.

        Args:
            sims: a dict containing intra-/inter-network similarity matrices

        Return；
            assist: a dict containing assisted variables, i.e., positive samples, negative sampling weights and similarity matrices
        """
        assist = {'intra': [], 'inter': []}
        if len(sims['intra']) > 0:
            # intra: [[a, b]]
            assist['intra'] = [[self.add_assist(mat)
                              for mat in mats]
                             for mats in sims['intra']]
        if len(sims['inter']) > 0:
            # 'inter': [a]
            assist['inter'] = [self.add_assist(mat)
                            for mat in sims['inter']]
        return assist

    def train_intra(self):
        """
        Reconstruct intra-network similarity matrices.
        """
        loss_intra = []
        for i in range(len(self.assist['intra'])):
            loss_a = 0.
            for j in range(len(self.assist['intra'][i])):
                pairs, weights, mat = self.assist['intra'][i][j]
                batches = self.load_data(
                    pairs, cfg.batch_size)
                N = len(batches)
                loss_c = 0.
                for batch in batches:
                    f_s, f_t = self.model['embs']((j, j))
                    i_s, i_t = batch
                    # asymmetric similarity
                    loss = self.loss(
                        f_s, f_t,
                        i_s, i_t,
                        lambda x: x,
                        self.model['intra'][i],
                        cfg.neg,
                        weights,
                        mat
                    )
                    loss_c += loss
                    self.optimize(self.opt_emb, loss)
                loss_c /= N
                loss_a += loss_c
            loss_a /= len(self.assist['intra'][i])
            loss_intra.append(loss_a)
        return loss_intra

    def train_inter(self):
        """
        Reconstruct inter-network similarity matrices.
        """
        # pairs: positive samples
        # weights: negative sampling weights
        # mat: ground truth similarity matrix
        loss_inter = []
        for i in range(len(self.assist['inter'])):
            pairs, weights, mat = self.assist['inter'][i]
            batches = self.load_data(
                pairs, cfg.batch_size)
            N = len(batches)
            loss_c = 0.
            for batch in batches:
                f_s, f_t = self.model['embs']((0, 1))
                i_s, i_t = batch
                # symmetric similarity
                loss = self.loss(
                    f_s, f_t,
                    i_s, i_t,
                    self.model['inter'][i],
                    self.model['inter'][i],
                    cfg.neg,
                    weights,
                    mat
                )
                loss_c += loss
                self.optimize(self.opt_emb, loss)
            loss_c /= N
            loss_inter.append(loss_c)
        return loss_inter

    def train_labels(self):
        """
        Supverised training.
        """
        # pairs: positive samples
        # weights: negative sampling weights
        # mat: ground truth similarity matrix
        pairs, weights, mat = self.assist['labels']
        batches = self.load_data(
            pairs, cfg.batch_size)
        N = len(batches)
        loss_c = 0.
        for batch in batches:
            f_s, f_t = self.model['embs']((0, 1))
            i_s, i_t = batch
            loss = self.loss(
                f_s, f_t,
                i_s, i_t,
                self.model['common'][0],
                self.model['common'][1],
                cfg.neg,
                weights,
                mat
            )
            loss_c += loss
            self.optimize(self.opt_labels, loss)
        loss_c /= N
        return loss_c

    def train(self):
        """
        Main function for training.
        """
        for epoch in range(1, cfg.epochs + 1):
            if len(self.assist['intra']) > 0:
                # reconstruct intra-network similarity matrix
                losses = self.train_intra()
                losses = sum(losses) / len(losses)
                self.log.info('epoch {:03d} loss_intra {:.4f}'.format(
                    epoch, losses))

            if len(self.assist['inter']) > 0:
                # reconstruct inter-network similarity matrix
                losses = self.train_inter()
                losses = sum(losses) / len(losses)
                self.log.info('epoch {:03d} loss_inter {:.4f}'.format(
                    epoch, losses))

            if cfg.supervised:
                # supervised training
                if self.assist.get('labels') is not None:
                    loss = self.train_labels()
                    self.log.info('epoch {:03d} loss_labels {:.4f}'.format(
                        epoch, loss))
            _ = self.eval_bi()
        self.save_emb('data/MNE/emb_p_{}.pkl'.format(cfg.log.split('/')[1]))

    def get_emb_cat(self) -> tuple:
        """
        Get concatenated user embeddings of different feature spaces. The embeddings may not lie in a common space.

        Return:
            x_s/x_t: user embeddings in the source/target network
        """
        f_s, f_t = self.model['embs']((0, 1))
        x_s = torch.cat([mapping(f_s)
                         for mapping in self.model['intra']] + \
                        [mapping(f_s)
                         for mapping in self.model['inter']]
                        , dim=1)
        x_t = torch.cat([mapping(f_t)
                         for mapping in self.model['intra']] + \
                        [mapping(f_t)
                         for mapping in self.model['inter']]
                        , dim=1)
        return x_s, x_t

    def save_emb_cat(self, path: str) -> None:
        """
        Save user embeddings.

        Args:
            path: path to save user embeddings
        """
        embs = [i.cpu() for i in self.get_emb_cat()]
        write_pickle(embs, path)

    def get_emb(self) -> tuple:
        """
        Get user embeddings.

        Return:
            f_s/f_t: user embeddings in the source/target network
        """
        f_s, f_t = self.model['embs']((0, 1))
        if cfg.supervised:
            # map users to a common space if supervised information is provided
            f_s = self.model['common'][0](f_s)
            f_t = self.model['common'][1](f_t)
        return f_s, f_t
