import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import copy
from tqdm import tqdm
import time
from sklearn.cluster import KMeans
# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


dtype = torch.float64
index_dtype = torch.int32
torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda")


def J(tau_init, tau_transition, alpha, pi, beta,Y):
    N, Q = tau_init.shape
    T = Y.shape[0]
    Y = Y.to(device)

    tau_marg = tau_margin_generator(tau_init,tau_transition).to(device)

    q_indices = torch.arange(Q,dtype=index_dtype).view(1, 1, Q, 1, 1).expand(T, Q, Q, N, N)
    l_indices = torch.arange(Q,dtype=index_dtype).view(1, Q, 1, 1, 1).expand(T, Q, Q, N, N)
    t_indices = torch.arange(T,dtype=index_dtype).view(T, 1, 1, 1, 1).expand(T, Q, Q, N, N)
    Y_indices = Y.view(T,1,1,N,N).expand(T, Q, Q, N, N)
    phi_values_new = phi_vectorized(q_indices,l_indices ,t_indices, Y_indices, beta)
    del q_indices,l_indices,t_indices,Y_indices

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
    indices = torch.triu_indices(N, N, offset=1).to(device)
    term3 = torch.mul(torch.einsum('aik,ajl->aklij', tau_marg, tau_marg), log_phi_values)[:,:, :, indices[0], indices[1]].sum()
    
    J_value = term1 + term2 + term3
    return J_value  # Since we are using gradient ascent

        
def phi_vectorized(q, l, t, y, beta, distribution='Bernoulli'):

    max_q_l = torch.max(q, l)
    min_q_l = torch.min(q, l)
    
    indentication = (q == l).to(device)
    outer = beta[t, min_q_l, max_q_l]
    inner = beta[0, min_q_l, max_q_l]

    prob = torch.where(indentication, inner, outer)

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

def initial_clustering(adj_matrices, k):
    # 인접 행렬을 쌓아 하나의 큰 행렬로 만듭니다.
    stacked_matrix = torch.hstack([matrix for matrix in adj_matrices])
    # k-means 알고리즘 적용
    kmeans = KMeans(n_clusters=k, random_state=0)
    initial_labels = kmeans.fit_predict(stacked_matrix)
    initial_labels_tensor = torch.tensor(initial_labels, dtype=torch.int64)
    one_hot_labels = torch.nn.functional.one_hot(initial_labels_tensor, num_classes=k)
    one_hot_labels = one_hot_labels.to(dtype=dtype)
    return one_hot_labels

def inital_parameter(adj_matrices,k):
    # pi = torch.eye(num_latent, dtype=torch.float64, requires_grad=True)
    # alpha = torch.full((num_latent,), 1/num_latent, dtype=torch.float64, requires_grad=True)
    # beta = torch.rand(time_stamp, num_latent, num_latent, dtype=torch.float64, requires_grad=True)

    alpha_pre = torch.rand(num_latent, device=device,dtype=dtype, requires_grad=True) 
    # torch.full((num_latent,), 1/num_latent, requires_grad=True) 
    pi_pre = torch.rand(num_latent, num_latent,device=device,dtype=dtype, requires_grad=True) 
    beta_pre = torch.rand(time_stamp, num_latent, num_latent, device=device,dtype=dtype, requires_grad=True) 

    alpha_pre = (alpha_pre * 10 - 5).detach().requires_grad_(True)
    pi_pre = (pi_pre * 10 - 5).detach().requires_grad_(True)
    beta_pre = (beta_pre * 10 - 5).detach().requires_grad_(True)

    tau_init = initial_clustering(adj_matrices, k).to(device)
    tau_init.requires_grad_()
    
    tau_transition = torch.eye(num_latent, device= device,dtype=torch.float64).expand(time_stamp, num_nodes, num_latent, num_latent).clone().requires_grad_()

    return tau_init, tau_transition, pi_pre, beta_pre, alpha_pre


def estimate_gpu(adjacency_matrix, 
                 num_latent = 2 ,
                 stability = 0.9, 
                 total_iteration = 0 ,
                 distribution = 'Bernoulli', 
                 bernoulli_case = 'low_plus', 
                 trial = 0,
                 mode = 'new_kmeans'):

    time_stamp, num_nodes , _ = adjacency_matrix.shape

    print("time_stamp:", time_stamp)
    print("num_nodes:", num_nodes)
    print("num_latent:", num_latent)

    # # initialization version
    # tau_init = torch.rand(num_nodes, num_latent, requires_grad=True)  # Example tau matrix
    # tau_transition = torch.rand(time_stamp, num_nodes, num_latent, num_latent, requires_grad=True)
    # alpha = torch.full((num_latent,), 1/num_latent, requires_grad=True)
    # pi = torch.rand(num_latent, num_latent, requires_grad=True)
    # beta = torch.rand(time_stamp, num_latent, num_latent, requires_grad=True)

    # initialization version
    tau_init_pre = torch.rand(num_nodes, num_latent, device=device,dtype=dtype, requires_grad=True) 
    tau_transition_pre = torch.rand(time_stamp, num_nodes, num_latent, num_latent, device=device,dtype=dtype, requires_grad=True) 
    alpha_pre = torch.rand(num_latent, device=device,dtype=dtype, requires_grad=True) 
    # torch.full((num_latent,), 1/num_latent, requires_grad=True) 
    pi_pre = torch.rand(num_latent, num_latent,device=device,dtype=dtype, requires_grad=True) 
    beta_pre = torch.rand(time_stamp, num_latent, num_latent, device=device,dtype=dtype, requires_grad=True) 

    tau_init_pre = (tau_init_pre * 10 - 5).detach().requires_grad_(True)
    tau_transition_pre = (tau_transition_pre * 10 - 5).detach().requires_grad_(True)
    alpha_pre = (alpha_pre * 10 - 5).detach().requires_grad_(True)
    pi_pre = (pi_pre * 10 - 5).detach().requires_grad_(True)
    beta_pre = (beta_pre * 10 - 5).detach().requires_grad_(True)

    # tau_init_pre, tau_transition_pre, pi_pre, beta_pre, alpha_pre= inital_parameter(Y, num_latent)

    # load version 
    # pi, alpha, beta, tau_init, tau_transition = torch.load('para_model_2.pt')


    # Optimizer
    optimizer_theta = optim.Adam([pi_pre,alpha_pre,beta_pre], lr=1e-4)
    optimizer_tau =  optim.Adam([tau_init_pre, tau_transition_pre], lr=1e-4)

    # Stopping criteria parameters
    patience = 15
    threshold = 1e-4
    no_improve_count = 0 
    best_loss = float('inf') 

    str_stability = str(stability).replace('0.', '0p')
    # Gradient ascent
    num_iterations = 50000
    for iter in tqdm(range(num_iterations)):

        optimizer_theta.zero_grad()
        optimizer_tau.zero_grad()
        
        tau_init = F.softmax(tau_init_pre, dim=1)
        tau_transition = F.softmax(tau_transition_pre, dim=3)
        alpha = F.softmax(alpha_pre, dim=0)
        pi = F.softmax(pi_pre,dim=1)
        beta = torch.sigmoid(beta_pre)
        loss = - J(tau_init,tau_transition, alpha, pi, beta, adjacency_matrix)
        loss.backward()

        optimizer_theta.step()
        optimizer_tau.step()

        if iter % 100 == 0:
            current_loss = -loss.item()
            print(f"Iteration {iter}: Loss = {current_loss}")
            # print(best_loss, loss)
            if loss > best_loss + threshold :
                no_improve_count += 1
                print(no_improve_count)
            else:
                best_loss = loss
                no_improve_count = 0
            
            if no_improve_count >= patience:
                print(f"Stopping early at iteration {iter} due to no improvement.")
                break

            if iter % 500 == 0:
                torch.save([pi_pre, alpha_pre, beta_pre, tau_init_pre, tau_transition_pre], f'parameter/{num_nodes}_{time_stamp}_{str_stability}/{mode}_estimation/{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}/estimate_{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}_{total_iteration}_{trial}.pt')
    
    torch.save([pi_pre, alpha_pre, beta_pre, tau_init_pre, tau_transition_pre], f'parameter/{num_nodes}_{time_stamp}_{str_stability}/{mode}_estimation/{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}/estimate_{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}_{total_iteration}_{trial}.pt')
    return loss


if __name__ == "__main__":
    
    num_nodes = 100
    num_latent = 2

    time_stamp = 5
    stability = 0.75
    total_iteration = 0
    trial = 2

    
    # bernoulli_case = 'low_minus'
    bernoulli_case = 'low_plus'
    # bernoulli_case = 'medium_minus'
    # bernoulli_case = 'medium_plus'
    # bernoulli_case = 'medium_with_affiliation's
    # bernoulli_case = 'large'

    distribution = 'Bernoulli'

    str_stability = str(stability).replace('0.', '0p')
    Y = torch.load(f'parameter/{num_nodes}_{time_stamp}_{str_stability}/adjacency/{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}/Y_{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}_{total_iteration}.pt')
    estimate_gpu(adjacency_matrix = Y, num_latent = num_latent, stability = stability, total_iteration = total_iteration, bernoulli_case = bernoulli_case, trial = trial)