import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import copy
from tqdm import tqdm
import time

torch.autograd.set_detect_anomaly(True)

def J(tau_init_pre, tau_transition_pre, alpha_pre, pi_pre, Y):
    N, Q = tau_init_pre.shape
    T = Y.shape[0]
    
    # start_time = time.time() 

    tau_init = F.softmax(tau_init_pre, dim=1)
    tau_transition = F.softmax(tau_transition_pre, dim=3)
    alpha = F.softmax(alpha_pre, dim=0)
    pi = F.softmax(pi_pre,dim=0)
    tau_marg = tau_margin_generator(tau_init,tau_transition)

    # phi_values = torch.zeros((T, Q, Q, N, N))
    # for t in range(T):
    #     for q in range(Q):
    #         for l in range(Q):
    #             for i in range(N):
    #                 for j in range(N):
    #                     phi_values[t, q, l, i, j] = phi(q, l, t, Y[t, i, j])
    
    q_indices = torch.arange(Q).view(1, 1, Q, 1, 1).expand(T, Q, Q, N, N)
    l_indices = torch.arange(Q).view(1, Q, 1, 1, 1).expand(T, Q, Q, N, N)
    t_indices = torch.arange(T).view(T, 1, 1, 1, 1).expand(T, Q, Q, N, N)
    Y_indices = Y.view(T,1,1,N,N).expand(T, Q, Q, N, N)
    phi_values_new = torch.zeros((T, Q, Q, N, N))
    phi_values_new = phi_vectorized( q_indices,l_indices ,t_indices, Y_indices)
    # end_time = time.time()
    # step1_duration = end_time - start_time
    # print(f'Step 1 took {step1_duration :.2f} seconds')

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
    # start_time = time.time()
    term2 = 0
    for t in range(1, T):
        for i in range(N):
            for q in range(Q):
                for q_prime in range(Q):
                    term2 =term2 + tau_marg[t-1, i, q] * tau_transition[t, i, q, q_prime] * (
                        torch.log(pi[q, q_prime]) - torch.log(tau_transition[t, i, q, q_prime])
                    )
    # end_time = time.time()
    # step2_duration = end_time - start_time
    # print(f'Step 2 took {step2_duration :.2f} seconds')
    
    
    # Third term
    # start_time = time.time()
    # term3 = 0
    # for t in range(T):
    #     for i in range(N):
    #         for j in range(i+1, N):
    #             for q in range(Q):
    #                 for l in range(Q):
    #                     term3 =term3 + tau_marg[t, i, q] * tau_marg[t, j, l] * log_phi_values[t, q,l ,i,j]
    
    indices = torch.triu_indices(N, N, offset=1)
    term3_re = 0
    for t in range(T):
        tau_mul = torch.mul(torch.einsum('ik,jl->klij', tau_marg[t,:,:], tau_marg[t,:,:]), log_phi_values[t,:,:,:,:])
        term3_re += tau_mul[:, :, indices[0], indices[1]].sum()
    # end_time = time.time()
    # step3_duration = end_time - start_time
    # print(f'Step 3 took {step3_duration :.2f} seconds')
    
    J_value = term1 + term2 + term3_re
    return J_value  # Since we are using gradient ascent

def phi(q, l, t, y, distribution = 'Bernoulli'):
    if q < l:
        q, l = l, q

    if q == l:
        prob = torch.sigmoid(beta[0, q, l])
    else:
        prob = torch.sigmoid(beta[t, q, l])

    if y == 0:
        return 1 - prob
    else:
        if distribution == 'Bernoulli':
            return prob
        elif distribution == 'gamma':
            return "wrong"
def phi_vectorized(q, l, t, y, distribution='Bernoulli'):
    max_q_l = torch.max(q, l)
    min_q_l = torch.min(q, l)
    
    prob = torch.sigmoid(torch.where(q == l, beta[0, max_q_l, min_q_l], beta[t, max_q_l, min_q_l]))
    
    result = torch.where(y == 0, 1 - prob, prob)
    
    return result

def tau_margin_generator(tau_init, tau_transition):
    tau = torch.zeros((time_stamp, num_nodes, num_latent))
    for t in range(time_stamp):
        for i in range(num_nodes):
            result = tau_init[i, :]
            for t_prime in range(time_stamp):
                result = result @ tau_transition[t_prime, i, :,:]
            tau[t,i,:] = result
            # for q in range(num_latent):
            #     if t == 0:
            #         tau[t, i, q] = tau_init[i, q]
            #     elif t == 1:
            #         tau[t,i,q] = torch.sum(tau_init[i, :] * tau_transition[t, i, :, q])
            #     elif t == 2:
            #         for q_1 in range(num_latent):
            #             for q_2 in range(num_latent):
            #                 tau[t, i, q] = tau[t, i, q] + tau_init[i, q_2] * tau_transition[0, i, q_2, q_1] * tau_transition[1, i, q_1, q]
            #     else:
            #         temp_tau = torch.zeros_like(tau[t, i, q])
            #         for q_iter in range(num_latent):
            #             temp_tau = temp_tau + tau[t-1, i, q_iter] * tau_transition[t, i, q_iter, q]
            #         tau[t, i, q] = temp_tau
    return tau


num_nodes, num_latent, time_stamp = 10, 2, 5
distribution = 'Bernoulli'

tau_marg = torch.zeros((time_stamp, num_nodes, num_latent))
tau_init = torch.rand(num_nodes, num_latent, requires_grad=True)  # Example tau matrix
tau_transition = torch.rand(time_stamp, num_nodes, num_latent, num_latent, requires_grad=True)
alpha = torch.rand(num_latent, requires_grad=True)
pi = torch.rand(num_latent, num_latent, requires_grad=True)
beta = torch.rand(time_stamp, num_latent, num_latent, requires_grad=True)
Y = torch.tensor(torch.load('Y_parameter.pt'))



# Optimizer
optimizer_theta = optim.Adam([pi,alpha,beta], lr=0.001)
optimizer_tau =  optim.Adam([tau_init, tau_transition], lr=0.001)

# Gradient ascent
num_iterations = 10000
for iteration in tqdm(range(num_iterations)):
    optimizer_theta.zero_grad()
    optimizer_tau.zero_grad()

    loss = - J(tau_init,tau_transition, alpha, pi, Y)
    
    # start_time = time.time()
    loss.backward()
    optimizer_theta.step()
    optimizer_tau.step()
    # end_time = time.time()
    # step4_duration = end_time - start_time
    # print(f'Step 4 took {step4_duration :.2f} seconds')
    
    print(f"Iteration {iteration}: Loss = {-loss.item()}")

torch.save([pi,alpha,beta,tau_init, tau_transition], "parameter.pt")

