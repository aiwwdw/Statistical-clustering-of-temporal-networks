import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import adjusted_rand_score
import os

def tau_margin_generator(tau_init, tau_transition):
    T,N,Q,_ = tau_transition.shape
    tau = torch.zeros((T, N, Q))
    for t in range(T):
        result = tau_init[:, :]
        for t_prime in range(t):
            # print(result,tau_transition[t_prime, :, :,:])
            result = torch.einsum('ij,ijk->ik', result, tau_transition[t_prime+1, :, :,:])
            # print(result)
        tau[t,:,:] = result
    return tau

def eval(bernoulli_case, num_nodes, time_stamp, stability, total_iteration, trial = 0, mode = 'new_random'):
    """""
    compute ARI, J value,
    """""
    # Load the parameters from the file
    str_stability = str(stability).replace('0.', '0p')
    pi, alpha, beta, tau_init, tau_transition = torch.load(f'parameter/{num_nodes}_{time_stamp}_{str_stability}/{mode}_estimation/{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}/estimate_{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}_{total_iteration}_{trial}.pt')
    init_dist,transition_matrix, Bernoulli_parameter, Z = torch.load(f'parameter/{num_nodes}_{time_stamp}_{str_stability}/true/{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}/true_para_{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}_{total_iteration}.pt')
    Z = torch.tensor(Z, dtype=torch.int64)

    ### In case of My model
    if mode == 'new_random' or mode == 'new_kmeans':
        tau_init = F.softmax(tau_init, dim=1)
        tau_transition = F.softmax(tau_transition, dim=3)
        alpha = F.softmax(alpha, dim=0)
        pi = F.softmax(pi,dim=1)
        beta = F.sigmoid(beta)

    tau_marg = tau_margin_generator(tau_init, tau_transition)

    # Print each parameter
    # print("pi:", pi)
    # print("alpha:", alpha)
    # print("beta:", beta)

    # print("tau_init:", tau_init)
    # print("tau_transition:", tau_transition)
    # print("tau_marg:", tau_marg)

    all_pred = []
    all_true = [] 
    ARI = []
    for time in range(time_stamp):
        print("time is", time)
        indices = torch.argmax(tau_marg[time], dim=1)
        pred = indices
        true = Z[time]
        all_pred.extend(pred)
        all_true.extend(true)
        print(pred)
        print(true)
        difference = torch.sum(pred != true).item()
        print("Difference count:", difference)
        ari_score = adjusted_rand_score(true, pred)
        ARI.append(ari_score)
        print("Adjusted Rand Index:", ari_score)  

    global_ARI = adjusted_rand_score(all_true, all_pred)
    print("Global Adjusted Rand Index:", global_ARI)  
    average_ARI = np.mean(ARI)
    print("Average Adjusted Rand Index:", average_ARI)  
    
    return global_ARI, average_ARI

if __name__ == "__main__":
    num_nodes = 100
    time_stamp = 5
    stability = 0.75
    iteration = 0
    trial = 0
    
    # mode = 'prior_random'
    mode = 'prior_kmeans'
    # mode = 'new'

    # bernoulli_case = 'low_plus'
    # bernoulli_case = 'low_minus'
    bernoulli_case = 'low_plus'
    # bernoulli_case = 'medium_minus'
    # bernoulli_case = 'medium_plus'
    # bernoulli_case = 'medium_with_affiliation'
    # bernoulli_case = 'large'
    
    eval(bernoulli_case =bernoulli_case, num_nodes = num_nodes, time_stamp = time_stamp , stability = stability, total_iteration = iteration, trial = trial, mode = mode)
