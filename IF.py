import argparse

import os
from utils.general import read_pickle
import config as cfg
from instructors.IF import IFIns
import torch

# os.chdir('../')


def load_data(ratio: float) -> list:
    """
    Given a ratio of training data, get the corr
    """
    return read_pickle('data/train_test_{:.1f}.pkl'.format(ratio))


def match(train_test: list, k: int, option: str = '') -> None:
    """
    Main function for training.

    Args:
        train_test: matched pairs for training and testing, [[(i, j), ...], [(i, j), ...]]
        k: number of candidates, default: 30
        option: a string indicating what kinds of user information to use, typically structure, profile and content
    """

    # assert valid types of information
    options = set(option.split(' '))
    all_options = {'structure', 'profile', 'content'}
    if len(options - all_options) > 0 or len(options) == 0:
        raise ValueError('options: structure, profile, content')

    # create path for data if not existed.
    dir = 'data'
    if not os.path.exists(dir):
        os.makedirs(dir)

    # place holders for pre-computed intra-network and inter-network similarity matrices
    sims = {'intra': [], 'inter': []}

    if 'structure' in options:
        # adjacency matrix of the source network
        adj_s = read_pickle('{}/adj_s.pkl'.format(dir))
        # adjacency matrix of the target network
        adj_t = read_pickle('{}/adj_t.pkl'.format(dir))
        # intra-network similarity matrices are assumed in pairs
        sims['intra'].append([adj_s, adj_t])
    if 'profile' in options:
        # inter-network profile similarity matrix
        sim = read_pickle('{}/sims_p.pkl'.format(dir))
        sims['inter'].append(sim)
    if 'content' in options:
        # inter-network content similarity matrix
        sim = read_pickle('{}/sims_c.pkl'.format(dir))
        sims['inter'].append(sim)

    # set cuda device, multiple gpus are not supported yet.
    torch.cuda.set_device(cfg.cuda)
    # initialize an instructor that makes necessary reports.
    ins = IFIns(sims, train_test, k)
    # train the model
    ins.train()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', help='cuda id.', default=0, type=int)
    parser.add_argument('--epochs', help='number of training epochs.', default=120, type=int)
    parser.add_argument('--model', help='name of the model, used in logging.', default='IF', type=str)
    parser.add_argument('--ratio', help='ratio of training set, e.g. 0.1-0.9.', default=cfg.ratio, type=float)
    parser.add_argument('--options', help='types of user information to use, typically \'structure\', \'profile\' and \'content\', separated by space', default='structure profile content', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    print(args)
    cfg.init_args(args)
    match(load_data(cfg.ratio), k=cfg.k, option=cfg.options)
