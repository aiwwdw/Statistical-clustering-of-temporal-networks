import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import copy
from tqdm import tqdm
import time
from sklearn.cluster import KMeans

torch.autograd.set_detect_anomaly(True)

def J(tau_init, tau_transition, alpha, pi,beta, Y):
    N, Q = tau_init.shape
    T = Y.shape[0]
    
    tau_marg = tau_margin_generator(tau_init,tau_transition)
    # print(tau_init,tau_transition,tau_marg[1])
    
    q_indices = torch.arange(Q).view(1, 1, Q, 1, 1).expand(T, Q, Q, N, N)
    l_indices = torch.arange(Q).view(1, Q, 1, 1, 1).expand(T, Q, Q, N, N)
    t_indices = torch.arange(T).view(T, 1, 1, 1, 1).expand(T, Q, Q, N, N)
    Y_indices = Y.view(T,1,1,N,N).expand(T, Q, Q, N, N)
    phi_values_new = torch.zeros((T, Q, Q, N, N))
    phi_values_new = phi_vectorized(q_indices,l_indices ,t_indices, Y_indices,beta)


    log_phi_values = torch.log(phi_values_new)
    # Nan 방지용
    epsilon = 1e-50

    # First term
    term1 = torch.sum(tau_init * (torch.log(alpha) - torch.log(tau_init+epsilon)))
    # Second term
   
    term2=0
    for t in range(1, T):
        term2 += (torch.einsum('ij,ijk->ijk', tau_marg[t-1, :, :], tau_transition[t, :, :,:]) * (torch.log(pi[:, :]).unsqueeze(0).expand(N,Q,Q) - torch.log(tau_transition[t, :, :, :]+epsilon))).sum()

    # Third term
    indices = torch.triu_indices(N, N, offset=1)    
    term3 = torch.mul(torch.einsum('aik,ajl->aklij', tau_marg, tau_marg), log_phi_values)[:,:, :, indices[0], indices[1]].sum()

    
    J_value = term1 + term2 + term3
    return J_value  # Since we are using gradient ascent

        
def phi_vectorized(q, l, t, y, beta, distribution='Bernoulli'):
    max_q_l = torch.max(q, l)
    min_q_l = torch.min(q, l)
    prob = torch.where(q == l, beta[0, min_q_l, max_q_l], beta[t, min_q_l, max_q_l])
    result = torch.where(y == 0, 1 - prob, prob)
    return result

def tau_margin_generator(tau_init, tau_transition):
    tau = torch.zeros((time_stamp, num_nodes, num_latent))
    for t in range(time_stamp):
        result = tau_init[:, :]
        for t_prime in range(t):
            # print(result,tau_transition[t_prime, :, :,:])
            result = torch.einsum('ij,ijk->ik', result, tau_transition[t_prime+1, :, :,:])
            # print(result)
        tau[t,:,:] = result
    return tau


def true_tau(Z):
    tau_init_true = torch.zeros(num_nodes, num_latent,dtype=torch.float64)
    tau_transition_true = torch.zeros(time_stamp, num_nodes, num_latent, num_latent,dtype=torch.float64)
    for t in range(time_stamp):
        if t == 0:
            tau_init_true = torch.nn.functional.one_hot(Z[0] , num_classes=num_latent)
            tau_init_true = tau_init_true.to(dtype=torch.float64)
        else:
            for i in range(num_nodes):
                tau_transition_true[t,i,Z[t-1,i],Z[t,i]] = 1
    return tau_init_true, tau_transition_true

def true_beta(Bernoulli_parameter):
    beta = torch.zeros(time_stamp, num_latent, num_latent)
    for t in range(time_stamp):
        if t == 0:
            for q in range(num_latent):
                beta[t,q,q] = Bernoulli_parameter[q,q]
        for q in range(num_latent):
            for q_prime in range(q,num_latent):
                beta[t,q,q_prime] = Bernoulli_parameter[q,q_prime]
    return beta



num_nodes, num_latent, time_stamp = 100, 2, 5
distribution = 'Bernoulli'
Y = torch.tensor(torch.load('Y_MG_LPB.pt'))
init_dist,transition_matrix, Bernoulli_parameter, Z = torch.load('true_para_MG_LPB.pt')

init_dist = torch.tensor(init_dist, dtype=torch.float64)
transition_matrix = torch.tensor(transition_matrix, dtype=torch.float64)
Bernoulli_parameter = torch.tensor(Bernoulli_parameter, dtype=torch.float64)
Z = torch.tensor(Z, dtype=torch.int64)

tau_init, tau_transition = true_tau(Z)
beta = true_beta(Bernoulli_parameter)

# # initialization version
# tau_init = torch.zeros(num_nodes, num_latent)  # Example tau matrix
# tau_transition = torch.zeros(time_stamp, num_nodes, num_latent, num_latent)
# alpha = torch.full((num_latent,), 1/num_latent)
# pi = torch.rand(num_latent, num_latent)
# beta = torch.rand(time_stamp, num_latent, num_latent)


# load version 
# pi, alpha, beta, tau_init, tau_transition = torch.load('parameter1.pt')



loss = - J(tau_init,tau_transition, init_dist, transition_matrix,beta, Y)
print(f"True Objective function: Loss = {-loss.item()}")
   


