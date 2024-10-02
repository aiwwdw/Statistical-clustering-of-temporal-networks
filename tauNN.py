import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data, DataLoader
from model_compact import tau_margin_generator

class TAU_init(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=16):
        super(TAU_init, self).__init__()
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

class TAU_transition(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, hidden_dim=16):
        super(TAU_transition, self).__init__()
        self.fc_x = nn.Linear(input_dim, hidden_dim)
        self.fc_pre_tau = nn.Linear(latent_dim**2, hidden_dim)
        self.fc_output = nn.Linear(hidden_dim + hidden_dim, output_dim)


    def forward(self, x, pre_tau_transition):
        x_transformed = self.fc_x(x)  
        pre_tau_transformed = self.fc_pre_tau(pre_tau_transition)  
        
        concatenated = torch.cat((x_transformed, pre_tau_transformed), dim=1) 
        output = self.fc_output(concatenated)  

        return output

def tau_init_new(Y_latent, model):

    output = F.softmax(model(Y_latent), dim=1)
    return output

def tau_transition_new(Y_latent,pre_tau_transition, model):
    N, L = Y_latent.shape
    Q = 2 

    output = model(Y_latent, pre_tau_transition)  
    output = F.softmax(output.view(N, Q, Q), dim=2)
    return output


class GNNModel(nn.Module):
    def __init__(self, num_nodes, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.embedding = nn.Embedding(num_nodes, hidden_dim)  
        self.conv1 = GCNConv(hidden_dim, hidden_dim, add_self_loops=False) 
        self.conv2 = GCNConv(hidden_dim, hidden_dim, add_self_loops=False) 
        self.conv3 = GCNConv(hidden_dim, output_dim, add_self_loops=False)  

    def forward(self, node_ids, edge_index):
        x = self.embedding(node_ids)
        x = self.conv1(x, edge_index)
        x = F.relu(x)  
        x = self.conv2(x, edge_index)
        x = F.relu(x)  
        x = self.conv3(x, edge_index)
        return x
    
def adjacency_to_edge_index(Y):
    T, N, _ = Y.shape  # Y는 (T, N, N) 크기
    edge_indices = []

    for t in range(T):
        adj_matrix = Y[t]
        edges = torch.nonzero(adj_matrix, as_tuple=False).t().contiguous()
        edge_indices.append(edges)
    return edge_indices

num_nodes = 100
num_latent = 2
node_latent= 4

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
Y = torch.load(f'parameter/{num_nodes}_{time_stamp}_{str_stability}_4/adjacency/{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}/Y_{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}_{iteration}.pt')

TAU_init_model = TAU_init(input_dim=node_latent, output_dim= num_latent)  
TAU_transition_model = TAU_transition(input_dim=node_latent,latent_dim= num_latent, output_dim= num_latent * num_latent)  
GNN_latent = GNNModel(num_nodes=num_nodes, hidden_dim=16, output_dim=node_latent)

edge_indices_list = adjacency_to_edge_index(Y)

node_ids = torch.arange(num_nodes)
data_list = []

for edge_index in edge_indices_list:
    data = Data(x=node_ids, edge_index=edge_index)
    data_list.append(data)

loader = DataLoader(data_list, batch_size=10, shuffle=True)
for batch in loader:
    output = GNN_latent(batch.x, batch.edge_index)
    print(output.size())
splits = torch.split(output, 100)

tau_transition = torch.zeros(time_stamp,num_nodes,num_latent,num_latent)

for t, node_embedding in enumerate(splits):
    if t == 0:
        tau_init = tau_init_new(node_embedding, TAU_init_model)
    result = tau_transition[0].view(num_nodes,num_latent**2)
    for t_prime in range (t):
        result = TAU_transition_model(splits[t_prime], result)  
    tau_transition[t] = F.softmax(result.view(num_nodes, num_latent, num_latent), dim=2)

