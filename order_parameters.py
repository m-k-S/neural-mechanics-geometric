import torch
import torch.nn as nn

class Conv_prober(nn.Module):

    def __init__(self):
        super(Conv_prober, self).__init__()
        self.batch = None
        self.std_list = []
        # Grads
        self.grads_norms = []

        class sim_grads(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input, batch):
                # self.std_list.append(input.std(dim=[0,2,3]).mean().item())
                return input.clone()

            @staticmethod
            def backward(ctx, grad_output):
                M = grad_output.view(grad_output.shape[0], -1)
                self.grads_norms.append(M.norm().item())

                # all_norms = torch.linalg.norm(M.type(torch.FloatTensor), dim=[1])
                # num_nodes = self.batch.bincount()
                # batch_size = self.batch.max() # zero indexed
                # weighted_norm_sum_per_graph = []
                # for b in range(batch_size):
                #     b_norms = all_norms[self.batch == b].sum() / num_nodes[b]
                #     weighted_norm_sum_per_graph.append(b_norms)
                # norm_mean = sum(weighted_norm_sum_per_graph) / (batch_size+1)
                # self.grads_norms.append(norm_mean.item())

                return grad_output.clone(), None

        self.cal_prop = sim_grads.apply

    def forward(self, input, batch):
        self.batch = batch
        if not torch.is_grad_enabled():
            return input
        else:
            return self.cal_prop(input, batch)

class Activs_prober(nn.Module):
    def __init__(self):
        super(Activs_prober, self).__init__()
        # Activs
        self.activs_norms = []
        self.activs_corr = []
        # self.activs_ranks = []
        # self.activs_variance = []


        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        class sim_activs(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input, batch):
                M = input.clone()

                # Activation Norm
                all_norms = torch.linalg.norm(M.type(torch.FloatTensor), dim=[1])
                num_nodes = batch.bincount()
                batch_size = batch.max() # zero indexed
                weighted_norm_sum_per_graph = []
                for b in range(batch_size):
                    b_norms = all_norms[batch == b].sum() / num_nodes[b]
                    weighted_norm_sum_per_graph.append(b_norms)
                norm_mean = sum(weighted_norm_sum_per_graph) / (batch_size+1)
                self.activs_norms.append(norm_mean.item())

                # These functions need fixing to process graphs properly (i.e. weight by the number of nodes in a graph, using the batch input variable)

                # Activation Correlations
                # M = (M / anorms).reshape(M.shape[0], -1)
                # M = torch.matmul(M, M.T)
                # self.activs_corr.append(((M.sum(dim=1) - 1) / (M.shape[0]-1)).mean().item())

                # Activation Ranks (calculates stable rank)
                # tr = torch.diag(M).sum()
                # opnom = torch.linalg.norm(M + 1e-8 * M.mean() * torch.rand(M.shape).to(self.device), ord=2)
                # self.activs_ranks.append((tr / opnom).item())

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

def test2():
    print ('test')
