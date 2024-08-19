import numpy as np
from scipy.stats import gamma
import networkx as nx
import matplotlib.pyplot as plt
import torch
import os

def create_transition_matrix(num_nodes, stability):
    # Initialize an n x n matrix with zeros
    matrix = np.zeros((num_nodes, num_nodes))
    
    # Set the diagonal elements to 0.9
    np.fill_diagonal(matrix, stability)
    
    # Set the off-diagonal elements
    off_diagonal_value = (1-stability) / (num_nodes - 1)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                matrix[i, j] = off_diagonal_value
                
    return matrix


def weight_geneator(Z, num_nodes, Bernoulli_parameter,distribution = 'Bernoulli'):
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

def init_latent(num_nodes,num_latent,init_dist):
    latent_variables = np.zeros(num_nodes, dtype=int)
    
    for i in range(num_nodes):
        latent_variables[i] = np.random.choice(num_latent, p=init_dist[i])
    return latent_variables

def transition(Z,num_nodes,num_latent,transition_matrix):
    latent_variables = np.zeros(num_nodes, dtype=int)

    for i in range(num_nodes):
        latent_variables[i] = np.random.choice(num_latent,p=transition_matrix[Z[i]])

    return latent_variables

def generate_data(time_stamp = 10, num_nodes = 100, num_latent = 2,  stability = 0.9, total_iteration = 0, distribution = 'Bernoulli',bernoulli_case = 'low_plus'):

    Z = []
    Y = []


    init_dist = np.ones((num_nodes,num_latent))/num_latent
    transition_matrix = create_transition_matrix(num_latent, stability)
    if num_latent == 2:
        if bernoulli_case == 'low_minus':
            Bernoulli_parameter =np.matrix([[0.2,0.1],
                                            [0.1,0.15]]) 
        if bernoulli_case == 'low_plus':
            Bernoulli_parameter =np.matrix([[0.25,0.1],
                                            [0.1,0.2]]) 
        if bernoulli_case == 'medium_minus':
            Bernoulli_parameter =np.matrix([[0.3,0.1],
                                            [0.1,0.2]]) 
        if bernoulli_case == 'medium_plus':
            Bernoulli_parameter =np.matrix([[0.4,0.1],
                                            [0.1,0.2]]) 
        if bernoulli_case == 'medium_with_affiliation':
            Bernoulli_parameter =np.matrix([[0.3,0.1],
                                            [0.1,0.3]]) 
            
    elif num_latent == 3:              
        Bernoulli_parameter =np.matrix([[0.3,0.1,0.1],
                                        [0.1,0.25,0.1],
                                        [0.1,0.1,0.2]])

    else:
        Bernoulli_parameter = np.full((num_latent, num_latent), 0.1)
        diagonal_values = np.linspace(0.2, 0.4, num_latent)
        np.fill_diagonal(Bernoulli_parameter, diagonal_values)

    for t in range(time_stamp):
        if t == 0:
            Z.append(init_latent(num_nodes,num_latent,init_dist))
        else:
            Z.append(transition(Z[t-1],num_nodes, num_latent, transition_matrix))
        Y.append(weight_geneator(Z[t],num_nodes, Bernoulli_parameter,distribution))
    
    Y = torch.tensor(Y)

    str_stability = str(stability).replace('0.', '0p')

    torch.save(Y, f'parameter/adjacency/{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}/Y_{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}_{total_iteration}.pt')
    torch.save((init_dist,transition_matrix,Bernoulli_parameter,Z),f'parameter/true/{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}/true_para_{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}_{total_iteration}.pt')

    return Y
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

if __name__ == "__main__":
    time_stamp = 10
    num_latent = 2
    num_nodes = 100
    stability = 0.9
    iteration = 0
    distribution = 'Bernoulli'

    # bernoulli_case = 'low_plus'
    # bernoulli_case = 'low_minus'
    # bernoulli_case = 'low_plus'
    # bernoulli_case = 'medium_minus'
    bernoulli_case = 'medium_plus'
    # bernoulli_case = 'medium_with_affiliation'
    # bernoulli_case = 'large'

    generate_data(time_stamp = time_stamp, num_nodes = num_nodes, num_latent = num_latent, stability = stability, total_iteration = iteration, distribution = 'Bernoulli',bernoulli_case = bernoulli_case)