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
from torch_geometric.data import Data, DataLoader as GeoDataLoader
from torch.utils.data import DataLoader, TensorDataset
from tauNN import GNNModel, TAU_init,TAU_transition, tau_init_new, tau_transition_new,adjacency_to_edge_index

torch.autograd.set_detect_anomaly(True)

def J(tau_inits, tau_transitions, alpha, pi, beta, Y):
    batch_size, N, Q = tau_inits.shape
    batch_size, T, _,_ = Y.shape

    tau_marg = tau_margin_generator(tau_inits,tau_transitions)
    print(tau_marg.shape)
    q_indices = torch.arange(Q).view(1, 1, 1, Q, 1, 1).expand(batch_size, T, Q, Q, N, N).to("cuda:0")
    l_indices = torch.arange(Q).view(1, 1, Q, 1, 1, 1).expand(batch_size, T, Q, Q, N, N).to("cuda:0")
    t_indices = torch.arange(T).view(1, T, 1, 1, 1, 1).expand(batch_size, T, Q, Q, N, N).to("cuda:0")
    Y_indices = Y.view(batch_size,T,1,1,N,N).expand(batch_size,T, Q, Q, N, N).to("cuda:0")
    phi_values_new = torch.zeros((batch_size, T, Q, Q, N, N)).to("cuda:0")
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
    batch_size, time_stamp, num_nodes, num_latent, _ = tau_transition.shape
    tau = torch.zeros((batch_size, time_stamp, num_nodes, num_latent)).to('cuda:0')
    for t in range(time_stamp):
        result = tau_init[:, :, :]
        for t_prime in range(t):
            result = torch.einsum('bij,bijk->bik', result, tau_transition[: t_prime+1, :, :,:])
        tau[:,t,:,:] = result
    return tau


def inital_random(adj_matrices,num_latent):
    time_stamp, num_nodes , _ = adj_matrices.shape
    epsilon = 1e-2

    concentration = torch.ones(num_latent)
    tau_init = torch.stack([Dirichlet(concentration).sample() for _ in range(num_nodes)])
    tau_transition = torch.stack([
                        torch.stack([
                            torch.stack([Dirichlet(concentration).sample() for _ in range(num_latent)])
                                for __ in range(num_nodes)
                            ]) for ___ in range(time_stamp)
                        ])
    alpha = Dirichlet(concentration).sample()
    pi = torch.stack([Dirichlet(concentration).sample() for _ in range(num_latent)])
    beta = torch.rand(time_stamp, num_latent, num_latent)


    tau_init = torch.clamp(tau_init, min=epsilon, max=1 - epsilon)
    tau_transition = torch.clamp(tau_transition, min=epsilon, max=1 - epsilon)
    alpha = torch.clamp(alpha, min=epsilon, max=1 - epsilon)
    pi = torch.clamp(pi, min=epsilon, max=1 - epsilon)
    beta = torch.clamp(beta, min=epsilon, max=1 - epsilon)

    tau_init = tau_init / tau_init.sum(dim=1, keepdim=True)
    tau_transition = tau_transition/tau_transition.sum(dim=3, keepdim=True)
    alpha =alpha/alpha.sum(dim=0, keepdim=True)
    pi = pi / pi.sum(dim=1, keepdim=True)
    
    return tau_init, tau_transition, pi, beta, alpha

def initial_before_softmax(intitalization):
    tau_init, tau_transition, pi, beta, alpha = intitalization
    epsilon = 1e-2
    # 로그를 취할 때 양수인 값만 허용≈
    tau_init = torch.log(torch.clamp(tau_init, min=epsilon, max = 1-epsilon))
    tau_transition = torch.log(torch.clamp(tau_transition, min=epsilon, max = 1-epsilon))
    pi = torch.log(torch.clamp(pi, min=epsilon, max = 1-epsilon))
    alpha = torch.log(torch.clamp(alpha, min=epsilon, max = 1-epsilon))

    beta = np.log(beta/ (1 - beta))

    return tau_init, tau_transition, pi, beta, alpha

def inital_gradient(intitalization):
    tau_init, tau_transition, pi, beta, alpha = intitalization
    tau_init.requires_grad_()
    tau_transition.requires_grad_()
    pi.requires_grad_()
    beta.requires_grad_()
    alpha.requires_grad_()



def GNN_estimate(adjacency_matrices,  # 이제 DataLoader가 됩니다
                 initialization,
                 num_latent=2, 
                 stability=0.9, 
                 total_iteration=0,
                 distribution='Bernoulli', 
                 bernoulli_case='low_plus', 
                 trial=0,
                 num_iterations=10000,
                 mode='GNN',
                 node_latent=4,
                 device='cpu',
                 batch_size=10):  # 배치 크기 추가

    first_batch = next(iter(adjacency_matrices))[0]
    batch_size, time_stamp, num_nodes, _ = first_batch.shape
    print("Batch training mode")
    print(f"Batch size: {batch_size}, time_stamp: {time_stamp}, num_nodes: {num_nodes}, num_latent: {num_latent}")

    tau_init_pre, tau_transition_pre, pi_pre, beta_pre, alpha_pre = initialization

    # 모델 초기화
    TAU_init_model = TAU_init(input_dim=node_latent, output_dim=num_latent).to(device)
    TAU_transition_model = TAU_transition(input_dim=node_latent, latent_dim=num_latent, output_dim=num_latent * num_latent).to(device)
    GNN_latent = GNNModel(num_nodes=num_nodes, hidden_dim=16, output_dim=node_latent).to(device)

    # Load pretrained parameters if they exist
    loaded_parameters = torch.load('parameter/GNN_model.pt')
    
    TAU_init_model.load_state_dict(loaded_parameters['TAU_init_model'])
    TAU_transition_model.load_state_dict(loaded_parameters['TAU_transition_model'])
    GNN_latent.load_state_dict(loaded_parameters['GNN_latent'])

    alpha = F.softmax(alpha_pre, dim=0).to(device)
    pi = F.softmax(pi_pre, dim=1).to(device)
    beta = F.sigmoid(beta_pre).to(device)

    # Optimizer
    params = list(TAU_init_model.parameters()) + \
             list(TAU_transition_model.parameters()) + \
             list(GNN_latent.parameters())
    optimizer_theta = optim.Adam([pi_pre, alpha_pre, beta_pre], lr=5e-4)
    optimizer_tau = optim.Adam(params, lr=5e-5)    

    # Stopping criteria parameters
    patience = 10
    threshold = 1e-4
    no_improve_count = 0 
    best_loss = float('inf') 

    str_stability = str(stability).replace('0.', '0p')


    # 여러 adjacency_matrices에 대해 배치로 학습 진행
    for itertation in range(num_iterations):
        total_loss = 0

        # DataLoader에서 배치 단위로 데이터를 가져옵니다.
        for batch_data in adjacency_matrices:
            adjacency_batch = batch_data[0]
            print(adjacency_batch.shape)
            batch_size, time_stamp, num_nodes, _ = adjacency_batch.shape

            # 그래디언트 초기화
            optimizer_theta.zero_grad()
            optimizer_tau.zero_grad()
            tau_inits = []
            tau_transitions = []

            for batch_idx in range(batch_size):
                edge_indices_list = adjacency_to_edge_index(adjacency_batch[batch_idx])

                node_ids = torch.arange(num_nodes).to(device)
                data_list = []

                for edge_index in edge_indices_list:
                    data = Data(x=node_ids, edge_index=edge_index)
                    data_list.append(data)

                loader = GeoDataLoader(data_list, batch_size=20, shuffle=True)
                for batch in loader:
                    output = GNN_latent(batch.x, batch.edge_index)
                splits = torch.split(output, 100)

                tau_transition = torch.zeros(time_stamp,num_nodes,num_latent,num_latent).to('cuda:0')

                for t, node_embedding in enumerate(splits):
                    if t == 0:
                        tau_init = F.softmax(tau_init_new(node_embedding, TAU_init_model),dim=1)
                    else:
                        result = torch.zeros(num_nodes,num_latent*num_latent).to(device)
                        for t_prime in range (t):
                            result = TAU_transition_model(splits[t_prime], result)  
                        tau_transition[t] = F.softmax(result.view(num_nodes, num_latent, num_latent), dim=2)
                tau_inits.append(tau_init)
                tau_transitions.append(tau_transition)
            
            tau_inits_tensor = torch.stack(tau_inits)  # [batch_size, num_nodes, num_latent]
            tau_transitions_tensor = torch.stack(tau_transitions)


            # 파라미터 갱신
            alpha = F.softmax(alpha_pre, dim=0).to(device)
            pi = F.softmax(pi_pre, dim=1).to(device)
            beta = F.sigmoid(beta_pre).to(device)
            print(tau_init.shape)
            # 손실 계산 (예: 음의 로그 우도)
            loss = -J(tau_inits_tensor, tau_transitions_tensor, alpha, pi, beta, adjacency_batch)
            loss.backward()

            # 옵티마이저를 통해 파라미터 업데이트
            optimizer_theta.step()
            optimizer_tau.step()

            # 배치의 손실 합산
            total_loss += loss.item()

        # 평균 손실 출력
        if iter % 100 == 0:
            avg_loss = total_loss / len(adjacency_matrices)
            print(f"Iteration {itertation}: Average Loss = {avg_loss}")

            # 모델 저장 (필요 시)
            if itertation % 500 == 0:
                model_parameters = {
                    'TAU_init_model': TAU_init_model.state_dict(),
                    'TAU_transition_model': TAU_transition_model.state_dict(),
                    'GNN_latent': GNN_latent.state_dict()
                }
                torch.save(model_parameters, f'parameter/GNN_model.pt')
    return loss
    

if __name__ == "__main__":
    num_nodes = 100
    num_latent = 2
    node_latent = 5

    time_stamp = 5
    stability = 0.75
    iteration = 0

    trial = 100000
    batch_size = 10  # 배치 사이즈 추가

    device = 'cuda:0'

    # bernoulli_case = 'low_minus'
    # bernoulli_case = 'low_plus'
    bernoulli_case = 'medium_minus'
    # bernoulli_case = 'medium_plus'
    # bernoulli_case = 'medium_with_affiliation'
    # bernoulli_case = 'large'
    
    str_stability = str(stability).replace('0.', '0p')

    Y_list = []
    for iteration in range(100):
        Y = torch.load(f'parameter/{num_nodes}_{time_stamp}_{str_stability}/adjacency/{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}/Y_{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}_{iteration}.pt')
        Y_list.append(Y)

    adjacency_matrices = torch.stack(Y_list, dim=0).to(device)

    # 데이터를 DataLoader로 묶기
    dataset = TensorDataset(adjacency_matrices)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # initialization = inital_random(Y, num_latent)
    # torch.save(initialization, f"initalization.pt")
    
    initialization = torch.load(f"initalization.pt")
    
    intitalization_new_pre = initial_before_softmax(initialization)
    inital_gradient(intitalization_new_pre)

    # GNN_estimate 함수 호출
    GNN_estimate(adjacency_matrices=data_loader,  # data_loader로 전달
                 initialization=intitalization_new_pre, 
                 num_latent=num_latent, 
                 stability=stability, 
                 total_iteration=iteration, 
                 bernoulli_case=bernoulli_case, 
                 trial=trial,
                 node_latent=node_latent,
                 device=device,
                 batch_size=batch_size)  # 배치 사이즈 추가
 