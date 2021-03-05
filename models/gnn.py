import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from models.base import MLP


class GNNRaw(MessagePassing):
    """
    A wrapper for torch_geometric.nn.MessagePassing class to allow some self-definition.
    """
    def __init__(self):
        super(GNNRaw, self).__init__('add')
        # IMPORTANT: define the direction of message passing as from the target node to the source node
        self.flow = 'target_to_source'

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # edge_index, _ = remove_self_loops(edge_index)
        # edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, x=x)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def update(self, aggr_out: Tensor) -> Tensor:
        return aggr_out


class TriGNN(nn.Module):
    """
    A graph neural network to compute the neighborhood embeddings.
    """
    def __init__(self, dim: int) -> None:
        """
        Args:
            dim: dimension of user embeddings
        """
        super(TriGNN, self).__init__()
        # Make sure gnn.flow = 'target_to_source'
        self.gnn = GNNRaw()
        dim_in = 3 * dim
        self.mlp = MLP(dim_in, dim_in, dim_in * 2)
        self.linear = nn.Linear(dim, dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x: Tensor, y: Tensor, i: list, j: list, edges: list) -> tuple:
        """
        Args:
            x/y: user embeddings of the source/target network
            i/j: indices of users in the source/target network
            edges: indices of matched and unmatched neighbors in the source/target network, [[...], [...], [...], [...]]

        Return:
            xx/yy: neighborhood embeddings in the source/target network
        """
        x.data = F.normalize(x.data)
        y.data = F.normalize(y.data)
        x = self.tanh(self.linear(x))
        y = self.tanh(self.linear(y))
        x.data = F.normalize(x.data)
        y.data = F.normalize(y.data)
        i_s, i_t, j_s, j_t = edges
        # apply gnn to matched and unmatched neighbors in the source network
        x_s = self.gnn(x, i_s) * (len(i_s)) / (len(i_s) + len(i_t))
        x_t = self.gnn(x, i_t) * (len(i_t)) / (len(i_s) + len(i_t))
        # apply gnn to matched and unmatched neighbors in the target network
        y_s = self.gnn(y, j_s) * (len(j_s)) / (len(j_s) + len(j_t))
        y_t = self.gnn(y, j_t) * (len(j_t)) / (len(j_s) + len(j_t))
        ii = torch.tensor(i)
        jj = torch.tensor(j)
        # neighborhood embeddings in the source network
        xx = torch.cat([x[ii], x_s[ii], x_t[ii]], dim=1)
        xx = self.mlp(xx)
        # neighborhood embeddings in the target network
        yy = torch.cat([y[jj], y_s[jj], y_t[jj]], dim=1)
        yy = self.mlp(yy)
        return xx, yy
