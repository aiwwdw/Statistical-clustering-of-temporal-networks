import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import copy
from tqdm import tqdm
import time
from sklearn.cluster import KMeans
import os

torch.autograd.set_detect_anomaly(True)

def J(tau_init, tau_transition, alpha, pi, beta, Y):
    N, Q = tau_init.shape
    T = Y.shape[0]

    tau_marg = tau_margin_generator(tau_init,tau_transition)
    # print(tau_marg[0])
    
    q_indices = torch.arange(Q).view(1, 1, Q, 1, 1).expand(T, Q, Q, N, N)
    l_indices = torch.arange(Q).view(1, Q, 1, 1, 1).expand(T, Q, Q, N, N)
    t_indices = torch.arange(T).view(T, 1, 1, 1, 1).expand(T, Q, Q, N, N)
    Y_indices = Y.view(T,1,1,N,N).expand(T, Q, Q, N, N)
    phi_values_new = torch.zeros((T, Q, Q, N, N))
    phi_values_new = phi_vectorized(q_indices,l_indices ,t_indices, Y_indices, beta)

    log_phi_values = torch.log(phi_values_new)
    
    # 작은 값을 클리핑하여 NaN 방지
    eps = 1e-15
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
    tau = torch.zeros((time_stamp, num_nodes, num_latent), dtype= torch.float64)
    for t in range(time_stamp):
        result = tau_init[:, :]
        for t_prime in range(t):
            result = torch.einsum('ij,ijk->ik', result, tau_transition[t_prime+1, :, :,:])
        tau[t,:,:] = result
    return tau

def initial_clustering(adj_matrices, k):
    # 인접 행렬을 쌓아 하나의 큰 행렬로 만듭니다.
    stacked_matrix = torch.hstack([matrix for matrix in adj_matrices])
    # k-means 알고리즘 적용
    kmeans = KMeans(n_clusters=k, random_state=0)
    initial_labels = kmeans.fit_predict(stacked_matrix)
    initial_labels_tensor = torch.tensor(initial_labels, dtype=torch.int64)
    one_hot_labels = torch.nn.functional.one_hot(initial_labels_tensor, num_classes=k)
    one_hot_labels = one_hot_labels.to(dtype=torch.float64)
    return one_hot_labels

def inital_parameter(adj_matrices,num_latent):
    # stability = 0.9
    # pi = torch.eye(num_latent, dtype=torch.float64)
    # pi.fill_diagonal_(stability)
    # pi[pi == 0] = (1-stability)/(num_latent-1)

    # alpha = torch.full((num_latent,), 1/num_latent, dtype=torch.float64)
    # beta = torch.rand(time_stamp, num_latent, num_latent, dtype=torch.float64)
    time_stamp, num_nodes , _ = adj_matrices.shape
    alpha = torch.rand(num_latent) * 10
    alpha = F.softmax(alpha,dim=0)

    pi = torch.rand(num_latent, num_latent, dtype=torch.float64) * 10
    pi = F.softmax(pi,dim=1)

    beta = torch.rand(time_stamp, num_latent, num_latent, dtype=torch.float64)
    
    tau_init = initial_clustering(adj_matrices, num_latent)
    tau_transition = torch.eye(num_latent, dtype=torch.float64).expand(time_stamp, num_nodes, num_latent, num_latent)
    return tau_init, tau_transition, pi, beta, alpha



def VE_step(tau_init_pre, tau_transition_pre, alpha_pre, pi_pre, beta, Y):
    N, Q = tau_init_pre.shape
    T = Y.shape[0]
    epsilon =0
    
    tau_marg = tau_margin_generator(tau_init_pre,tau_transition_pre)
    
    tau_init = torch.zeros(tau_init_pre.size(),dtype=torch.float64)
    tau_transition = torch.zeros(tau_transition_pre.size(),dtype=torch.float64)

    q_indices = torch.arange(Q).view(1, 1, Q, 1, 1).expand(T, Q, Q, N, N)
    l_indices = torch.arange(Q).view(1, Q, 1, 1, 1).expand(T, Q, Q, N, N)
    t_indices = torch.arange(T).view(T, 1, 1, 1, 1).expand(T, Q, Q, N, N)
    Y_indices = Y.view(T,1,1,N,N).expand(T, Q, Q, N, N)
    phi_values_new = torch.zeros((T, Q, Q, N, N))
    phi_values_new = phi_vectorized(q_indices,l_indices ,t_indices, Y_indices, beta)



    tau_init_pre_expanded = tau_init_pre.permute(1,0).unsqueeze(0).unsqueeze(2).expand(Q, Q, N, N)
    result = pow(phi_values_new[0], tau_init_pre_expanded)

    result_prod = torch.prod(result, dim = 3)

    result_prod = torch.prod(result_prod, dim = 1)


    result_prod = result_prod.permute(1,0)


    tau_init = alpha_pre.unsqueeze(0).expand(N, -1) * result_prod

    tau_init = tau_init / (tau_init.sum(dim=1, keepdim=True) + epsilon)
  



    tau_marg_expanded = tau_marg.permute(0, 2, 1).unsqueeze(1).unsqueeze(3).expand(T, Q, Q, N, N)
    
    result = pow(phi_values_new , tau_marg_expanded)

    result_prod = torch.prod(result, dim = 4)

    result_prod = torch.prod(result_prod, dim = 2)


    pi_pre_expanded = pi_pre.unsqueeze(0).unsqueeze(0).expand(T, N, -1, -1)
    result_prod_expanded = result_prod.permute(0, 2, 1).unsqueeze(2).expand(-1, -1, Q, -1)
    tau_transition = pi_pre_expanded * result_prod_expanded

    tau_transition = tau_transition / (tau_transition.sum(dim=3, keepdim=True)+epsilon)



    return tau_init, tau_transition
        

def M_step(tau_init_pre, tau_transition_pre, alpha_pre, pi_pre, beta, Y):
    N, Q = tau_init_pre.shape
    T = Y.shape[0]
    epsilon = 0

    tau_marg = tau_margin_generator(tau_init_pre,tau_transition_pre)
    pi = torch.zeros(pi_pre.size(),dtype=torch.float64)

    alpha = tau_marg.sum(dim=(0, 1))
    alpha /= N * T

    for q in range(Q):
        for q_prime in range(Q):
            for t in range(1,T):
                for i in range(N):
                    pi[q,q_prime] += tau_marg[t-1,i,q] * tau_transition_pre[t,i,q,q_prime]
        pi[q,:] = pi[q,:] / (pi[q,:].sum())
    
    tau_prod = torch.einsum("tiq,tjq->tijq",tau_marg,tau_marg)
    tau_prod_Y = tau_prod * Y.unsqueeze(-1).expand(-1,-1,-1,Q)
    beta_q= (tau_prod_Y.sum(dim=(0,1,2)) )/ (tau_prod.sum(dim=(0,1,2))+ epsilon)
    for q in range(Q):
        beta[0,q,q] = beta_q[q]

    tau_prod_1 = torch.einsum("tiq,tjl->tijql",tau_marg,tau_marg)
    tau_prod_Y_1 = tau_prod_1 * Y.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,-1,Q,Q)
    beta_tql = (tau_prod_Y_1.sum(dim=(1,2)) ) / (tau_prod_1.sum(dim=(1,2)) + epsilon)
    upper_tri_indices = torch.triu_indices(Q, Q, offset=1)
    beta[:,upper_tri_indices[0], upper_tri_indices[1]] = beta_tql[:,upper_tri_indices[0], upper_tri_indices[1]]

    
    return alpha, pi, beta



def estimate_old(adjacency_matrix,
                 initialization,
                 num_latent = 2 ,
                 stability = 0.9, 
                 total_iteration = 0 ,
                 distribution = 'Bernoulli', 
                 bernoulli_case = 'low_plus', 
                 trial = 0,
                 mode = 'prior_random'):
    
    time_stamp, num_nodes , _ = adjacency_matrix.shape

    tau_init, tau_transition, pi, beta, alpha= initialization

    # # initialization version
    # if mode == 'prior_random':
    #     tau_init = torch.rand(num_nodes, num_latent, dtype=torch.float64) * 10 - 5
    #     tau_transition = torch.rand(time_stamp, num_nodes, num_latent, num_latent, dtype=torch.float64) * 10 - 5
    #     # alpha = torch.full((num_latent,), 1/num_latent, dtype=torch.float64)
    #     alpha = torch.rand(num_latent) * 10 - 5
    #     pi = torch.rand(num_latent, num_latent, dtype=torch.float64) * 10 - 5
    #     beta = torch.rand(time_stamp, num_latent, num_latent, dtype=torch.float64)

    #     tau_init = F.softmax(tau_init, dim=1)
    #     tau_transition = F.softmax(tau_transition, dim=3)
    #     pi = F.softmax(pi,dim=1)
    #     alpha = F.softmax(alpha,dim=0)

    # #K-means initialization
    # elif mode == 'prior_kmeans':
    #     tau_init, tau_transition, pi, beta, alpha= inital_parameter(adjacency_matrix,num_latent)

    # load version 
    # pi, alpha, beta, tau_init, tau_transition = torch.load('para_old_3.pt')

    no_improve_count = 0 
    best_loss = float('inf') 

    str_stability = str(stability).replace('0.', '0p')

    loss = - J(tau_init,tau_transition, alpha, pi, beta, adjacency_matrix)
    current_loss = -loss.item()
    print(f"Iteration {0}: Loss = {current_loss}")

    # Gradient ascent
    num_iterations = 5000
    for iteration in range(num_iterations):
        tau_init, tau_transition = VE_step(tau_init, tau_transition, alpha, pi, beta, adjacency_matrix)
        alpha, pi, beta = M_step(tau_init, tau_transition, alpha, pi, beta, adjacency_matrix)
        loss = - J(tau_init,tau_transition, alpha, pi, beta, adjacency_matrix)

        if iteration % 10 == 9:
            current_loss = -loss.item()
            print(f"Iteration {iteration+1}: Loss = {current_loss}")
            if np.isnan(current_loss):
                break
            # print(best_loss, loss)
            if loss >= best_loss-1e-10:
                no_improve_count += 1
                print(no_improve_count)
            else:
                best_loss = loss
                no_improve_count = 0
            
            if no_improve_count >= 3:
                print(f"Stopping early at iteration {iteration} due to no improvement.")
                break

            if iteration % 50 == 0:
                torch.save([pi, alpha, beta, tau_init, tau_transition], f'parameter/{num_nodes}_{time_stamp}_{str_stability}/{mode}_estimation/{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}/estimate_{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}_{total_iteration}_{trial}.pt')
    
    torch.save([pi, alpha, beta, tau_init, tau_transition], f'parameter/{num_nodes}_{time_stamp}_{str_stability}/{mode}_estimation/{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}/estimate_{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}_{total_iteration}_{trial}.pt')
    return loss

if __name__ == "__main__":
    num_nodes = 100
    num_latent = 2

    time_stamp = 5
    stability = 0.751
    iteration = 0
    trial = 0

    mode = 'prior_kmeans'
    # mode = 'prior_random'
    
    
    # bernoulli_case = 'low_plus'
    # bernoulli_case = 'low_minus'
    # bernoulli_case = 'low_plus'
    bernoulli_case = 'medium_minus'
    # bernoulli_case = 'medium_plus'
    # bernoulli_case = 'medium_with_affiliation'
    # bernoulli_case = 'large'

    str_stability = str(stability).replace('0.', '0p')
    Y = torch.load(f'parameter/{num_nodes}_{time_stamp}_{str_stability}/adjacency/{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}/Y_{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}_{iteration}.pt')

    directory_prior_est = f"parameter/{num_nodes}_{time_stamp}_{str_stability}/{mode}_estimation/{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}"
    if not os.path.exists(directory_prior_est):
        os.makedirs(directory_prior_est)
    

    estimate_old(adjacency_matrix = Y, num_latent = num_latent, stability = stability, total_iteration = iteration, bernoulli_case = bernoulli_case, trial = trial, mode = mode)

