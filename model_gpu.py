import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import copy
from tqdm import tqdm
import time
from sklearn.cluster import KMeans

torch.autograd.set_detect_anomaly(True)
device = torch.device("mps")

def J(tau_init_pre, tau_transition_pre, alpha_pre, pi_pre, Y):
    N, Q = tau_init_pre.shape
    T = Y.shape[0]
    
    tau_init = F.softmax(tau_init_pre, dim=1)
    tau_transition = F.softmax(tau_transition_pre, dim=3)
    alpha = F.softmax(alpha_pre, dim=0)
    pi = F.softmax(pi_pre,dim=1)
    tau_marg = tau_margin_generator(tau_init,tau_transition).to(device)
    
    q_indices = torch.arange(Q).view(1, 1, Q, 1, 1).expand(T, Q, Q, N, N).to(device)
    l_indices = torch.arange(Q).view(1, Q, 1, 1, 1).expand(T, Q, Q, N, N).to(device)
    t_indices = torch.arange(T).view(T, 1, 1, 1, 1).expand(T, Q, Q, N, N).to(device)
    Y_indices = Y.view(T,1,1,N,N).expand(T, Q, Q, N, N)
    phi_values_new = torch.zeros((T, Q, Q, N, N))
    phi_values_new = phi_vectorized(q_indices,l_indices ,t_indices, Y_indices)

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

        
def phi_vectorized(q, l, t, y, distribution='Bernoulli'):
    max_q_l = torch.max(q, l).to(device)
    min_q_l = torch.min(q, l).to(device)
    prob = torch.sigmoid(torch.where(q == l, beta[0, min_q_l, max_q_l], beta[t, min_q_l, max_q_l]))
    result = torch.where(y == 0, 1 - prob, prob)
    
    return result

def tau_margin_generator(tau_init, tau_transition):
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
    one_hot_labels = one_hot_labels.to(dtype=torch.float64)
    return one_hot_labels

def inital_parameter(adj_matrices,k):
    pi = torch.eye(num_latent, dtype=torch.float64, requires_grad=True)
    alpha = torch.full((num_latent,), 1/num_latent, dtype=torch.float64, requires_grad=True)
    beta = torch.rand(time_stamp, num_latent, num_latent, dtype=torch.float64, requires_grad=True)

    tau_init = initial_clustering(adj_matrices, k)
    tau_init.requires_grad_()
    
    tau_transition = torch.eye(num_latent, dtype=torch.float64).expand(time_stamp, num_nodes, num_latent, num_latent).clone().requires_grad_()
    
    return tau_init, tau_transition, pi, beta, alpha


distribution = 'Bernoulli'
Y = torch.tensor(torch.load('Y_large.pt'),dtype=torch.float32)
time_stamp, num_nodes , _ = Y.shape
num_latent = 10


# # initialization version
# tau_init = torch.rand(num_nodes, num_latent, requires_grad=True)  # Example tau matrix
# tau_transition = torch.rand(time_stamp, num_nodes, num_latent, num_latent, requires_grad=True)
# alpha = torch.full((num_latent,), 1/num_latent, requires_grad=True)
# pi = torch.rand(num_latent, num_latent, requires_grad=True)
# beta = torch.rand(time_stamp, num_latent, num_latent, requires_grad=True)

# initialization GPU version
tau_init = torch.rand(num_nodes, num_latent, dtype=torch.float32, device=device)
tau_transition = torch.rand(time_stamp, num_nodes, num_latent, num_latent, dtype=torch.float32, device=device)
alpha = torch.full((num_latent,), 1/num_latent, dtype=torch.float32, device=device)
pi = torch.rand(num_latent, num_latent, dtype=torch.float32, device=device)
beta = torch.rand(time_stamp, num_latent, num_latent, dtype=torch.float32, device=device)
Y = Y.to(device)


# tau_init, tau_transition, pi, beta, alpha= inital_parameter(Y, num_latent)

# load version 
# pi, alpha, beta, tau_init, tau_transition = torch.load('para_model_2.pt')


# Optimizer
optimizer_theta = optim.Adam([pi,alpha,beta], lr=0.01)
optimizer_tau =  optim.Adam([tau_init, tau_transition], lr=0.01)

# Gradient ascent
num_iterations = 50000
for iteration in tqdm(range(num_iterations)):
    optimizer_theta.zero_grad()
    optimizer_tau.zero_grad()
    start_time = time.time()  # 시작 시간 기록
    loss = - J(tau_init,tau_transition, alpha, pi, Y)
    end_time = time.time()  # 끝 시간 기록
    elapsed_time = end_time - start_time  # 경과 시간 계산
    print(f"Iteration {iteration}:Time per calculation = {elapsed_time:.4f} seconds")
    start_time = time.time()  # 시작 시간 기록
    loss.backward()
    optimizer_theta.step()
    optimizer_tau.step()
    end_time = time.time()  # 끝 시간 기록
    elapsed_time = end_time - start_time  # 경과 시간 계산
    print(f"Iteration {iteration}:Time per gradient = {elapsed_time:.4f} seconds")
    # if iteration % 100 ==0:
    print(f"Iteration {iteration}: Loss = {-loss.item()}")
    if iteration % 500 == 0:
        torch.save([pi,alpha,beta,tau_init, tau_transition], "para_model_large.pt")


