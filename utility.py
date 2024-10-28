import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.distributions import Dirichlet


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

def inital_random(adj_matrices,num_latent):
    time_stamp, num_nodes , _ = adj_matrices.shape
    epsilon = 1e-2

    concentration = torch.ones(num_latent, dtype=torch.float64)
    tau_init = torch.stack([Dirichlet(concentration).sample() for _ in range(num_nodes)])
    tau_transition = torch.stack([
                        torch.stack([
                            torch.stack([Dirichlet(concentration).sample() for _ in range(num_latent)])
                                for __ in range(num_nodes)
                            ]) for ___ in range(time_stamp)
                        ])
    alpha = Dirichlet(concentration).sample()
    pi = torch.stack([Dirichlet(concentration).sample() for _ in range(num_latent)])
    beta = torch.rand(time_stamp, num_latent, num_latent, dtype=torch.float64)


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