import torch
import torch.nn as nn

# def dirichlet_energy(M):

def batch_stable_rank(M):
    # Takes the stable rank of the full batch matrix, i.e. with the graph dimension and batch dimension stacked on top of each other
    # The matrix is of dimension (V_1 + ... + V_B) x F where V_i is the number of nodes in the ith graph in the batch of size B
    # and F is the feature dimension of the convolutional layer
    # Returns a scalar

    return (torch.linalg.norm(M, ord='fro')**2) / (torch.linalg.norm(M, ord=2)**2).item()

def batch_algebraic_rank(M):
    # Takes the algebraic rank of the full batch matrix, i.e. with the graph dimension and batch dimension stacked on top of each other
    # The matrix is of dimension (V_1 + ... + V_B) x F where V_i is the number of nodes in the ith graph in the batch of size B
    # and F is the feature dimension of the convolutional layer
    # Returns a scalar
    rank = torch.linalg.matrix_rank(M).type(torch.FloatTensor)
    return rank.item()

def graph_rank(M, batch):
    # Takes the stable ranks of the matrices defined by each graph in a batch, i.e. of the matrix of size V x F, where
    # V is the number of nodes in the graph and F is the size of the feature dimension of the convolution layer.
    # The stable ranks are then averaged over all of the graphs in the batch.
    # Returns a scalar

    batch_size = max(batch) + 1
    dim1 = M.shape[0]
    features = M.shape[-1]
    M = M.reshape(batch_size, int(dim1/batch_size), features)

    graph_ranks = []
    for graph in range(batch_size - 1):
        graph_matrix = M[graph, :, :]

        # D = torch.matmul(graph_matrix, graph_matrix.T).type(torch.FloatTensor)
        # tr = torch.diag(D).sum()
        # rank = tr**2 / torch.linalg.norm(D, ord='fro')**2
        rank = torch.linalg.matrix_rank(graph_matrix).type(torch.FloatTensor)
        graph_ranks.append(rank.item())

    mean_graph_rank = sum(graph_ranks) / batch_size
    return mean_graph_rank

def feature_rank(M, batch):
    # Takes the stable ranks of the matrices defined by each feature for all graphs in a batch, i.e. of the matrix of size
    # V x B, where V is the number of nodes in each graph and B is the number of graphs in the batch.
    # The stable ranks are tracked for each separate feature.
    # Returns an array of length == number of features

    batch_size = max(batch) + 1
    dim1 = M.shape[0]
    features = M.shape[-1]
    M = M.reshape(batch_size, int(dim1/batch_size), features)

    feature_ranks = []
    for feat in range(features):
        feature_matrix = M[:, :, feat]
        # D = torch.matmul(feature_matrix, feature_matrix.T).type(torch.FloatTensor)
        # tr = torch.diag(D).sum()
        # rank = tr**2 / torch.linalg.norm(D, ord='fro')**2

        rank = torch.linalg.matrix_rank(feature_matrix).type(torch.FloatTensor)
        feature_ranks.append(rank.item())

    return feature_ranks


def activation_norm(M, batch):
    # Takes the norm of the feature vector for each node, and then takes the average of these norms
    # across all vectors for all graphs in a batch. Each vector is normalized by the number of nodes
    # in the graph that the vector belongs to.

    all_norms = torch.linalg.norm(M.type(torch.FloatTensor), dim=[1]) # take the norm of each output feature vector
    num_nodes = batch.bincount()
    batch_size = batch.max() # largest node number in the batch; zero indexed
    weighted_norm_sum_per_graph = []
    for b in range(batch_size):
        b_norms = all_norms[batch == b].sum() / num_nodes[b]
        weighted_norm_sum_per_graph.append(b_norms)
    norm_mean = sum(weighted_norm_sum_per_graph) / (batch_size+1)
    return norm_mean.item()

def gradient_norm(M, batch):
    # Takes the norm of the feature vector for each node, and then takes the average of these norms
    # across all vectors for all graphs in a batch. Each vector is normalized by the number of nodes
    # in the graph that the vector belongs to.

    all_norms = torch.linalg.norm(M.type(torch.FloatTensor), dim=[1])
    num_nodes = self.batch.bincount()
    batch_size = self.batch.max() # zero indexed
    weighted_norm_sum_per_graph = []
    for b in range(batch_size):
        b_norms = all_norms[self.batch == b].sum() / num_nodes[b]
        weighted_norm_sum_per_graph.append(b_norms)
    norm_mean = sum(weighted_norm_sum_per_graph) / (batch_size+1)
    return norm_mean.item()


class ConvolutionProbe(nn.Module):

    def __init__(self):
        super(ConvolutionProbe, self).__init__()
        self.batch = None

        # Grads
        self.gradient_norms = []

        # Ranks
        # self.batch_algebraic_ranks = []
        self.batch_stable_ranks = []
        # self.graph_mean_ranks = []

        # feature_ranks will be a matrix of size num_training_steps x hidden_channels
        # self.feature_ranks = []

        class sim_grads(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input, batch):
                if not self.training:
                    return input.clone()
                else:
                    M = input.clone()

                    # Activation Rank
                    # Stable rank is more suitable for numerics: https://arxiv.org/pdf/1501.01571.pdf
                    stable_rank = batch_stable_rank(M)
                    # algebraic_rank = batch_algebraic_rank(M)
                    # graph_mean_rank = graph_rank(M, batch)
                    # f_rank = feature_rank(M, batch)
                    self.batch_stable_ranks.append(stable_rank)
                    # self.batch_algebraic_ranks.append(algebraic_rank)

                    # self.graph_mean_ranks.append(graph_mean_rank)
                    # self.feature_ranks.append(f_rank)

                    return input.clone()

            @staticmethod
            def backward(ctx, grad_output):
                if not self.training:
                    return grad_output.clone(), None
                else:
                    M = grad_output.view(grad_output.shape[0], -1)
                    self.gradient_norms.append(M.norm().item())

                    return grad_output.clone(), None

        self.cal_prop = sim_grads.apply

    def forward(self, input, batch):
        self.batch = batch
        if not torch.is_grad_enabled():
            return input
        else:
            return self.cal_prop(input, batch)

class ActivationProbe(nn.Module):
    def __init__(self):
        super(ActivationProbe, self).__init__()
        # Activs
        self.activation_norm = []
        self.activation_correlation = []

        # self.batch_algebraic_ranks = []
        self.batch_stable_ranks = []
        # self.graph_mean_ranks = []

        # feature_ranks will be a matrix of size num_training_steps x hidden_channels
        # self.feature_ranks = []

        class sim_activs(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input, batch):
                if not self.training:
                    return input.clone()
                else:
                    M = input.clone()

                    # Activation Norm
                    norm_mean = activation_norm(M, batch)
                    self.activation_norm.append(norm_mean)

                    # Activation Rank
                    # Stable rank is more suitable for numerics: https://arxiv.org/pdf/1501.01571.pdf
                    stable_rank = batch_stable_rank(M)
                    # algebraic_rank = batch_algebraic_rank(M)
                    # graph_mean_rank = graph_rank(M, batch)
                    # f_rank = feature_rank(M, batch)
                    self.batch_stable_ranks.append(stable_rank)
                    # self.batch_algebraic_ranks.append(algebraic_rank)
                    # self.graph_mean_ranks.append(graph_mean_rank)
                    # self.feature_ranks.append(f_rank)

                    return input.clone()

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output.clone(), None

        self.cal_prop = sim_activs.apply

    def forward(self, input, batch):
        if not torch.is_grad_enabled():
            return input
        else:
            return self.cal_prop(input, batch)
