import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import copy
from tqdm import tqdm
import time
from sklearn.cluster import KMeans
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

torch.autograd.set_detect_anomaly(True)

def J(tau_init, tau_transition, alpha, pi, beta, Y):
    N, Q = tau_init.shape
    T = Y.shape[0]
    

    tau_marg = tau_margin_generator(tau_init,tau_transition)
    
    q_indices = torch.arange(Q).view(1, 1, Q, 1, 1).expand(T, Q, Q, N, N)
    l_indices = torch.arange(Q).view(1, Q, 1, 1, 1).expand(T, Q, Q, N, N)
    t_indices = torch.arange(T).view(T, 1, 1, 1, 1).expand(T, Q, Q, N, N)
    Y_indices = Y.view(T,1,1,N,N).expand(T, Q, Q, N, N)
    phi_values_new = torch.zeros((T, Q, Q, N, N))
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

def tau_margin_generator(tau_init, tau_transition):
    time_stamp, num_nodes, num_latent, _ = tau_transition.shape
    tau = torch.zeros((time_stamp, num_nodes, num_latent))
    for t in range(time_stamp):
        result = tau_init[:, :]
        for t_prime in range(t):
            result = torch.einsum('ij,ijk->ik', result, tau_transition[t_prime+1, :, :,:])
        tau[t,:,:] = result
    return tau

def initial_clustering(adj_matrices, num_latent):
    # 인접 행렬을 쌓아 하나의 큰 행렬로 만듭니다.
    stacked_matrix = torch.hstack([matrix for matrix in adj_matrices])
    # k-means 알고리즘 적용
    kmeans = KMeans(n_clusters=num_latent, random_state=0)
    initial_labels = kmeans.fit_predict(stacked_matrix)
    initial_labels_tensor = torch.tensor(initial_labels, dtype=torch.int64)
    one_hot_labels = torch.nn.functional.one_hot(initial_labels_tensor, num_classes=num_latent)
    one_hot_labels = one_hot_labels.to(dtype=torch.float64)
    return one_hot_labels

def inital_parameter(adj_matrices,num_latent):
    time_stamp, num_nodes , _ = adj_matrices.shape
    pi = torch.eye(num_latent, dtype=torch.float64, requires_grad=True)
    alpha = torch.full((num_latent,), 1/num_latent, dtype=torch.float64, requires_grad=True)
    beta = torch.rand(time_stamp, num_latent, num_latent, dtype=torch.float64, requires_grad=True)

    tau_init = initial_clustering(adj_matrices, num_latent)
    tau_init.requires_grad_()
    
    tau_transition = torch.eye(num_latent, dtype=torch.float64).expand(time_stamp, num_nodes, num_latent, num_latent).clone().requires_grad_()
    
    return tau_init, tau_transition, pi, beta, alpha



def estimate(adjacency_matrix, num_latent = 2 ,stability = 0.9, total_iteration = 0 ,distribution = 'Bernoulli', bernoulli_case = 'low_plus', trial = 0):
    
    time_stamp, num_nodes , _ = adjacency_matrix.shape

    print("time_stamp:", time_stamp)
    print("num_nodes:", num_nodes)
    print("num_latent:", num_latent)

    scale_factor = 5

    # initialization version
    tau_init_pre = torch.rand(num_nodes, num_latent, requires_grad=True) 
    tau_transition_pre = torch.rand(time_stamp, num_nodes, num_latent, num_latent, requires_grad=True) 
    alpha_pre = torch.full((num_latent,), 1/num_latent, requires_grad=True)
    pi_pre = torch.rand(num_latent, num_latent, requires_grad=True) 
    beta_pre = torch.rand(time_stamp, num_latent, num_latent, requires_grad=True)

    # tau_init, tau_transition, pi, beta, alpha= inital_parameter(adjacency_matrix, num_latent)

    # load version 
    # pi, alpha, beta, tau_init, tau_transition = torch.load('para_model_MP_LPB.pt', weights_only=True)


    # Optimizer
    optimizer_theta = optim.Adam([pi_pre, alpha_pre, beta_pre], lr=1e-1)
    optimizer_tau = optim.Adam([tau_init_pre, tau_transition_pre], lr=1e-1)

    # Learning rate scheduler
    scheduler_theta = StepLR(optimizer_theta, step_size=200, gamma=0.9)
    scheduler_tau = StepLR(optimizer_tau, step_size=200, gamma=0.9)

    # Stopping criteria parameters
    patience = 10
    threshold = 1e-6  
    no_improve_count = 0 
    best_loss = float('inf') 

    str_stability = str(stability).replace('0.', '0p')
    # Gradient ascent
    num_iterations = 10000
    for iter in range(num_iterations):

        optimizer_theta.zero_grad()
        optimizer_tau.zero_grad()
        
        tau_init = F.softmax(tau_init_pre, dim=1)
        tau_transition = F.softmax(tau_transition_pre, dim=3)
        alpha = F.softmax(alpha_pre, dim=0)
        pi = F.softmax(pi_pre,dim=1)
        beta = F.sigmoid(beta_pre)

        loss = -J(tau_init, tau_transition, alpha, pi, beta, adjacency_matrix)
        loss.backward()
        
        optimizer_theta.step()
        optimizer_tau.step()


        scheduler_theta.step()
        scheduler_tau.step()
        
        # Check for stopping criteria every 100 iterations
        if iter % 100 == 0:
            current_loss = -loss.item()
            print(f"Iteration {iter}: Loss = {current_loss}")

            # Stopping criteria check
            if best_loss - current_loss < threshold:
                no_improve_count += 1
            else:
                best_loss = current_loss
                no_improve_count = 0

            if no_improve_count >= patience:
                print(f"Stopping early at iteration {iter} due to no improvement.")
                break

        if iter % 500 == 0:
            torch.save([pi, alpha, beta, tau_init, tau_transition], f'parameter/estimation/estimate_{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}_{total_iteration}_{trial}.pt')
    
    torch.save([pi, alpha, beta, tau_init, tau_transition], f'parameter/estimation/estimate_{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}_{total_iteration}_{trial}.pt')
    return loss
    

if __name__ == "__main__":
    num_nodes = 100
    num_latent = 2

    time_stamp = 10
    stability = 0.9
    iteration = 0
    
    bernoulli_case = 'low_plus'
    # bernoulli_case = 'low_minus'
    # bernoulli_case = 'low_plus'
    # bernoulli_case = 'medium_minus'
    # bernoulli_case = 'medium_plus'
    # bernoulli_case = 'medium_with_affiliation'
    # bernoulli_case = 'large'

    str_stability = str(stability).replace('0.', '0p')
    Y = torch.load(f'parameter/adjacency/Y_{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}_{iteration}.pt')
    estimate(adjacency_matrix = Y, num_latent = num_latent, stability = stability, total_iteration = iteration, bernoulli_case = bernoulli_case)