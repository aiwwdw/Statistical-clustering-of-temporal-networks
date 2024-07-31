import numpy as np
from scipy.stats import gamma
import networkx as nx
import matplotlib.pyplot as plt
import torch

def create_transition_matrix(n, stability):
    # Initialize an n x n matrix with zeros
    matrix = np.zeros((n, n))
    
    # Set the diagonal elements to 0.9
    np.fill_diagonal(matrix, stability)
    
    # Set the off-diagonal elements
    off_diagonal_value = (1-stability) / (n - 1)
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i, j] = off_diagonal_value
                
    return matrix


def weight_geneator(Z, distribution = 'Bernoulli'):
    weight = np.zeros((num_nodes,num_nodes))
    for i in range(num_nodes):
        for j in range(i):
            if i == j:
                continue
            beta = Bernoulli_parameter[Z[i],Z[j]]
            if np.random.rand() > beta:
                weight[i,j] = 0
                weight[j,i] = 0
            else:
                if distribution == 'Bernoulli':
                    weight[i,j] = 1
                    weight[j,i] = 1
                if distribution == 'gamma':
                    weight[i,j] = gamma.rvs(1.0, scale=1)
                    weight[j,i] = gamma.rvs(1.0, scale=1)
    return weight 

def init_latent():
    latent_variables = np.zeros(num_nodes, dtype=int)
    
    for i in range(num_nodes):
        latent_variables[i] = np.random.choice(num_latent, p=init_dist[i])
    return latent_variables

def transition(Z):
    latent_variables = np.zeros(num_nodes, dtype=int)

    for i in range(num_nodes):
        latent_variables[i] = np.random.choice(num_latent,p=transition_matrix[Z[i]])

    return latent_variables


Z = []
Y = []

time_stamp = 5
num_latent = 10
num_nodes = 1000
stability = 0.9
distribution = 'Bernoulli'


init_dist = np.ones((num_nodes,num_latent))/num_latent
transition_matrix = create_transition_matrix(num_latent, stability)
if num_latent == 2:
    Bernoulli_parameter =np.matrix([[0.25,0.1],
                                    [0.1,0.2]])   
elif num_latent == 3:              
    Bernoulli_parameter =np.matrix([[0.3,0.1,0.1],
                                    [0.1,0.25,0.1],
                                    [0.1,0.1,0.2]])
elif num_latent == 10:
    Bernoulli_parameter = np.full((num_latent, num_latent), 0.1)
    diagonal_values = np.linspace(0.2, 0.4, num_latent)
    np.fill_diagonal(Bernoulli_parameter, diagonal_values)

for t in range(time_stamp):
    if t == 0:
        Z.append(init_latent())
    else:
        Z.append(transition(Z[t-1]))
    Y.append(weight_geneator(Z[t],distribution))

name = 'large'
torch.save(Y, 'Y_large.pt')
torch.save((init_dist,transition_matrix,Bernoulli_parameter,Z),"true_para_large.pt")

# Plot the graph for each timestamp and save to file
# for t in range(time_stamp):
#     G = nx.Graph()
#     weight_matrix = Y[t]
#     for i in range(num_nodes):
#         G.add_node(i)  # Ensure all nodes are added
#         for j in range(i):
#             if weight_matrix[i, j] > 0:
#                 G.add_edge(i, j, weight=weight_matrix[i, j])
    
#     pos = nx.spring_layout(G)  # Positioning of nodes
#     plt.figure(figsize=(20, 20))
    
#     node_labels = {i: f"{i} (Z={Z[t][i]})" for i in range(num_nodes)}
#     nx.draw(G, pos, with_labels=True, labels=node_labels, node_size=10, node_color='lightblue', font_size=10, font_weight='bold')
#     plt.title(f"Graph at Timestamp {t}")
#     plt.savefig(f"graph_timestamp_{t}.png")  # Save the plot as a file
#     plt.close()  # Close the figure to avoid display