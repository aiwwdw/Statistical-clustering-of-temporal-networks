import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import copy
from tqdm import tqdm
import time
from sklearn.cluster import KMeans
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import os
import torch.nn.functional as F
from torch.distributions import Dirichlet
from utility import initial_before_softmax, inital_gradient

torch.autograd.set_detect_anomaly(True)

def J(tau_init, tau_transition, alpha, pi, beta, Y, device):
    N, Q = tau_init.shape
    T = Y.shape[0]
    
    
    tau_marg = tau_margin_generator(tau_init,tau_transition).to(device)

    q_indices = torch.arange(Q).view(1, 1, Q, 1, 1).expand(T, Q, Q, N, N).to(device)
    l_indices = torch.arange(Q).view(1, Q, 1, 1, 1).expand(T, Q, Q, N, N).to(device)
    t_indices = torch.arange(T).view(T, 1, 1, 1, 1).expand(T, Q, Q, N, N).to(device)
    Y_indices = Y.view(T,1,1,N,N).expand(T, Q, Q, N, N)

    phi_values_new = phi_vectorized(q_indices,l_indices ,t_indices, Y_indices, beta)


    log_phi_values = torch.log(phi_values_new)
    
    # 작은 값을 클리핑하여 NaN 방지
    eps = 1e-10
    tau_init = torch.clamp(tau_init, min=eps)
    alpha = torch.clamp(alpha, min=eps)
    pi = torch.clamp(pi, min=eps)
    tau_transition = torch.clamp(tau_transition, min=eps)

    # First term
    term1 = torch.sum(tau_init * (torch.log(alpha) - torch.log(tau_init)))
   

    # Second term
    term2=0
    for t in range(1, T):
        term2 += (torch.einsum('ij,ijk->ijk', tau_marg[t-1, :, :], tau_transition[t, :, :,:]) * (torch.log(pi[:, :]).unsqueeze(0).expand(N,Q,Q) - torch.log(tau_transition[t, :, :, :]))).sum()

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


# def tau_margin_generator(tau_init, tau_transition):
#     time_stamp, num_nodes, num_latent, _ = tau_transition.shape
#     tau = torch.zeros((time_stamp, num_nodes, num_latent))
#     for t in range(time_stamp):
#         result = tau_init[:, :]
#         for t_prime in range(t):
#             result = torch.einsum('ij,ijk->ik', result, tau_transition[t_prime+1, :, :,:])
#         tau[t,:,:] = result
#     return tau
 
def tau_margin_generator(tau_init, tau_transition):
    time_stamp, num_nodes, num_latent, _ = tau_transition.shape
    tau = torch.zeros((time_stamp, num_nodes, num_latent))
    result = tau_init[:, :]
    for t in range(time_stamp):
        if t == 0:
            tau[t,:,:] = result
            continue
        result= torch.einsum('ij,ijk->ik', result, tau_transition[t, :, :,:])
        tau[t,:,:] = result
    return tau


def estimate(adjacency_matrix, 
             initialization,
             num_latent = 2 , 
             stability = 0.9, 
             total_iteration = 0 ,
             distribution = 'Bernoulli', 
             bernoulli_case = 'low_plus', 
             trial = 0,
             num_iterations = 10000,
             device = 'cpu',
             mode = 'new_random'):
    
    time_stamp, num_nodes , _ = adjacency_matrix.shape
    adjacency_matrix = adjacency_matrix.to(torch.int32)

    print("time_stamp:", time_stamp)
    print("num_nodes:", num_nodes)
    print("num_latent:", num_latent)

    scale_factor = 5

    # # initialization version
    # tau_init_pre = torch.rand(num_nodes, num_latent, requires_grad=True) 
    # tau_transition_pre = torch.rand(time_stamp, num_nodes, num_latent, num_latent, requires_grad=True) 
    # alpha_pre = torch.rand(num_latent, requires_grad=True) 
    # # torch.full((num_latent,), 1/num_latent, requires_grad=True) 
    # pi_pre = torch.rand(num_latent, num_latent, requires_grad=True) 
    # beta_pre = torch.rand(time_stamp, num_latent, num_latent, requires_grad=True) 

    # tau_init_pre = (tau_init_pre * 10 - 5).detach().requires_grad_(True)
    # tau_transition_pre = (tau_transition_pre * 10 - 5).detach().requires_grad_(True)
    # alpha_pre = (alpha_pre * 10 - 5).detach().requires_grad_(True)
    # pi_pre = (pi_pre * 10 - 5).detach().requires_grad_(True)
    # beta_pre = (beta_pre * 10 - 5).detach().requires_grad_(True)

    tau_init_pre, tau_transition_pre, pi_pre, beta_pre, alpha_pre= initialization

    tau_init_pre = tau_init_pre.to(device)
    tau_transition_pre = tau_transition_pre.to(device)
    pi_pre = pi_pre.to(device)
    beta_pre = beta_pre.to(device)
    alpha_pre = alpha_pre.to(device)
    adjacency_matrix = adjacency_matrix.to(device)
    
    tau_init_pre.requires_grad_()
    tau_transition_pre.requires_grad_()
    pi_pre.requires_grad_()
    beta_pre.requires_grad_()
    alpha_pre.requires_grad_()

    # tau_init_pre, tau_transition_pre, pi_pre, beta_pre, alpha_pre= inital_parameter(adjacency_matrix, num_latent)


    # load version 
    # pi, alpha, beta, tau_init, tau_transition = torch.load('para_model_MP_LPB.pt', weights_only=True)


    # Optimizer
    optimizer_theta = optim.Adam([pi_pre, alpha_pre, beta_pre], lr=5e-2)
    optimizer_tau = optim.Adam([tau_init_pre, tau_transition_pre], lr=5e-2)

    # Learning rate scheduler
    scheduler_theta = StepLR(optimizer_theta, step_size=200, gamma=0.9)
    scheduler_tau = StepLR(optimizer_tau, step_size=200, gamma=0.9)

    # Stopping criteria parameters
    patience = 10
    threshold = 1e-4
    no_improve_count = 0 
    best_loss = float('inf') 

    str_stability = str(stability).replace('0.', '0p')

    # tau_init = torch.softmax(tau_init_pre, dim=1)
    # tau_transition = torch.softmax(tau_transition_pre, dim=3)
    # alpha = torch.softmax(alpha_pre, dim=0)
    # pi = torch.softmax(pi_pre,dim=1)
    # beta = torch.sigmoid(beta_pre)

    # print(tau_init, tau_transition, pi, beta, alpha)
    start_time = time.time()
    for iter in range(num_iterations):


        optimizer_theta.zero_grad()
        optimizer_tau.zero_grad()
       

        tau_init = torch.softmax(tau_init_pre, dim=1)
        tau_transition = torch.softmax(tau_transition_pre, dim=3)
        alpha = torch.softmax(alpha_pre, dim=0)
        pi = torch.softmax(pi_pre,dim=1)
        beta = torch.sigmoid(beta_pre)

        # tau_init = tau_init.to(device)
        # tau_transition = tau_transition.to(device)
        # pi = pi.to(device)
        # beta = beta.to(device)
        # alpha = alpha.to(device)
        # adjacency_matrix = adjacency_matrix.to(device)
        
        # tau_init.requires_grad_()
        # tau_transition.requires_grad_()
        # pi.requires_grad_()
        # beta.requires_grad_()
        # alpha.requires_grad_()

        loss = -J(tau_init, tau_transition, alpha, pi, beta, adjacency_matrix, device)


        loss.backward()

        
        optimizer_theta.step()
        optimizer_tau.step()

        # scheduler_theta.step()
        # scheduler_tau.step()
        
        # Stopping criteria check
       
        # Check for stopping criteria every 100 iterations
       
        if iter % 100 == 0:
            # end_time = time.time()
            # print(f"Execution Time: {(end_time - start_time)/100} seconds")
            current_loss = -loss.item()
            if np.isnan(current_loss):
                print(f"Stopping early at iteration {iter} due to Nan VALUE.")
                torch.save([pi_pre, alpha_pre, beta_pre, tau_init_pre, tau_transition_pre], f'parameter/{num_nodes}_{time_stamp}_{str_stability}/{mode}_estimation/{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}/estimate_{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}_{total_iteration}_{trial}.pt')
                break
            print(f"Iteration {iter}: Loss = {current_loss}")
            # print(best_loss, loss)
            if loss >= best_loss:
                no_improve_count += 1
                print(no_improve_count)
            else:
                best_loss = loss
                no_improve_count = 0
            
            if no_improve_count >= patience:
                print(f"Stopping early at iteration {iter} due to no improvement.")
                torch.save([pi_pre, alpha_pre, beta_pre, tau_init_pre, tau_transition_pre], f'parameter/{num_nodes}_{time_stamp}_{str_stability}/{mode}_estimation/{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}/estimate_{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}_{total_iteration}_{trial}.pt')
                break  
            # start_time = time.time()
    torch.save([pi_pre, alpha_pre, beta_pre, tau_init_pre, tau_transition_pre], f'parameter/{num_nodes}_{time_stamp}_{str_stability}/{mode}_estimation/{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}/estimate_{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}_{total_iteration}_{trial}.pt')

    return loss
    

if __name__ == "__main__":
    num_nodes = 100
    num_latent = 2

    time_stamp = 5
    stability = 0.75
    iteration = 92
    trial = 5


    # bernoulli_case = 'low_minus'
    # bernoulli_case = 'low_plus'
    bernoulli_case = 'medium_minus'
    # bernoulli_case = 'medium_plus'
    # bernoulli_case = 'medium_with_affiliation's
    # bernoulli_case = 'large'

    str_stability = str(stability).replace('0.', '0p')
    Y = torch.load(f'parameter/{num_nodes}_{time_stamp}_{str_stability}/adjacency/{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}/Y_{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}_{iteration}.pt')
    
    # initialization = inital_random(Y,num_latent)
    # torch.save(initialization,f"initalization.pt")
    
    initialization = torch.load(f"initalization.pt")
    
    intitalization_new_pre = initial_before_softmax(initialization)
    inital_gradient(intitalization_new_pre)

    estimate(adjacency_matrix = Y, initialization = intitalization_new_pre, num_latent = num_latent, stability = stability, total_iteration = iteration, bernoulli_case = bernoulli_case, trial = trial)
