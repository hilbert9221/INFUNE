import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Emb(nn.Module):
    """
    User embedding matrices of different social networks.
    """
    def __init__(self, sizes: list, dim: int) -> None:
        """
        Args:
            sizes: numbers of users in different social networks
            dim: dimension of user embeddings
        """
        super(Emb, self).__init__()
        self.embs = nn.ParameterList(
            [self.init_randn_uni(size, dim)
             for size in sizes])

    @staticmethod
    def init_randn_uni(size: int, dim: int) -> nn.Parameter:
        """
        Initialize user embeddings via normalized Gaussian so that the embeddings of two users are approximately orthogonal.
        Important for optimization.

        Args:
            size: number of users
            dim: dimension of user embeddings

        Returns:
            initialized user embeddings
        """
        emb = nn.Parameter(torch.randn(size, dim))
        # l2 normalization
        emb.data = F.normalize(emb.data)
        return emb

    def forward(self, index: tuple) -> list:
        """
        Args:
            index: indices of social networks

        Returns:
            user embedding matrices of given indices
        """
        return [self.embs[i] for i in index]


class MLP(nn.Module):
    """
    2-layer MLP
    """
    def __init__(self, dim_in: int, dim_out: int,
                 dim_hid: int = None, act: nn.Module = None) -> None:
        """
        Args:
            dim_in: input dimension
            dim_out: output dimension
            dim_hid: hidden layer dimension
            act: activation function
        """
        super(MLP, self).__init__()
        if act is None:
            act = nn.Tanh()
        if dim_hid is None:
            dim_hid = dim_in * 2
        self.model = nn.Sequential(
            nn.Linear(dim_in, dim_hid),
            act,
            nn.Linear(dim_hid, dim_out)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
