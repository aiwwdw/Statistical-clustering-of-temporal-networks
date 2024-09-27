import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops

class TAU(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=16):
        super(TAU, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  
        self.fc3 = nn.Linear(hidden_dim, output_dim) 
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def tau_init_new(Y_latent, model):
    Y_latent = torch.tensor(Y_latent, dtype=torch.float32)  
    output = F.softmax(model(Y_latent))
    return output

def tau_transition_new(Y_latent, model):
    N, L = Y_latent.shape
    Q = 2 
    Y_latent = torch.tensor(Y_latent, dtype=torch.float32) 
    output = model(Y_latent)  
    output = F.softmax(output.view(N, Q, Q), dim=2)
    return output


class GNNModel(nn.Module):
    def __init__(self, num_nodes, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.embedding = nn.Embedding(num_nodes, hidden_dim)  
        self.conv1 = GCNConv(hidden_dim, hidden_dim, add_self_loops=False) 
        self.conv2 = GCNConv(hidden_dim, output_dim, add_self_loops=False)  

    def forward(self, node_ids, edge_index):
        x = self.embedding(node_ids)
        x = self.conv1(x, edge_index)
        x = F.relu(x)  
        x = self.conv2(x, edge_index)
        return x
    

num_nodes = 100
num_latent = 2
node_latent= 5

time_stamp = 5
stability = 0.75
iteration = 64
trial = 2


# bernoulli_case = 'low_minus'
# bernoulli_case = 'low_plus'
# bernoulli_case = 'medium_minus'
bernoulli_case = 'medium_plus'
# bernoulli_case = 'medium_with_affiliation's
# bernoulli_case = 'large'

str_stability = str(stability).replace('0.', '0p')
Y = torch.load(f'parameter/{num_nodes}_{time_stamp}_{str_stability}/adjacency/{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}/Y_{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}_{iteration}.pt')

TAU_init_new = TAU(input_dim=node_latent, output_dim= num_latent)  
TAU_transition_new = TAU(input_dim=node_latent, output_dim= num_latent * num_latent)  
GNN_latent = GNNModel(num_nodes=num_nodes, hidden_dim=8, output_dim=node_latent)

edge_index = torch.nonzero(Y[0].clone().detach(), as_tuple=False).t().contiguous()
node_ids = torch.arange(num_nodes)

node_embeddings = GNN_latent(node_ids, edge_index)

# print(node_embeddings.size())
# print(tau_init_new(node_embeddings, TAU_init_new))
# print(tau_transition_new(node_embeddings, TAU_transition_new))

