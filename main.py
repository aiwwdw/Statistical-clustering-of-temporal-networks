from adj_generator import generate_data
from model_compact import estimate
from evaluation import eval
from true_model import true_J
from model_gpu import estimate_gpu
from GNN_model import GNN_estimate
from old_model_compact import estimate_old
import numpy as np
import torch
from tqdm import tqdm
from sklearn.cluster import KMeans
import pandas as pd
import os
import torch.nn.functional as F
from torch.distributions import Dirichlet

def compare_variables(var1, var2, var_name):
    if torch.equal(var1, var2):
        print(f"{var_name}: 두 값이 동일합니다.")
    else:
        difference = (var1 - var2).abs().mean()  # 두 값의 차이의 절대값 평균 계산
        print(f"{var_name}: 값이 다릅니다. 평균 차이: {difference.item()}")

def logit_safe(y, epsilon=1e-50):
    # y 값을 [epsilon, 1 - epsilon]으로 제한하여 극단값을 피함
    y = np.clip(y, epsilon, 1 - epsilon)
    return np.log(y / (1 - y))

def inital_kmeans(adj_matrices,num_latent):
    epsilon = 1e-2
    time_stamp, num_nodes , _ = adj_matrices.shape
    concentration = torch.ones(num_latent, dtype=torch.float64)

    alpha = Dirichlet(concentration).sample()
    pi = torch.stack([Dirichlet(concentration).sample() for _ in range(num_latent)])
    beta = torch.rand(time_stamp, num_latent, num_latent, dtype=torch.float64)

    tau_init = initial_clustering(adj_matrices, num_latent)
    tau_transition = torch.eye(num_latent, dtype=torch.float64).expand(time_stamp, num_nodes, num_latent, num_latent)
    
    tau_init = torch.clamp(tau_init, min=epsilon, max=1 - epsilon)
    tau_transition = torch.clamp(tau_transition, min=epsilon, max=1 - epsilon)
    alpha = torch.clamp(alpha, min=epsilon, max=1 - epsilon)
    pi = torch.clamp(pi, min=epsilon, max=1 - epsilon)
    beta = torch.clamp(beta, min=epsilon, max=1 - epsilon)

    return tau_init, tau_transition, pi, beta, alpha

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
    

    # tau_init = torch.rand(num_nodes, num_latent, dtype=torch.float64) * 10 - 5
    # tau_transition = torch.rand(time_stamp, num_nodes, num_latent, num_latent, dtype=torch.float64) * 10 - 5
    # # alpha = torch.full((num_latent,), 1/num_latent, dtype=torch.float64)
    # alpha = torch.rand(num_latent) * 10 - 5
    # pi = torch.rand(num_latent, num_latent, dtype=torch.float64) * 10 - 5
    # beta = torch.rand(time_stamp, num_latent, num_latent, dtype=torch.float64)
    
    return tau_init, tau_transition, pi, beta, alpha

def inital_prior(intitalization):
    tau_init, tau_transition, pi, beta, alpha = intitalization
    tau_init = F.softmax(tau_init, dim=1)
    tau_transition = F.softmax(tau_transition, dim=3)
    pi = F.softmax(pi,dim=1)
    alpha = F.softmax(alpha,dim=0)
    beta = F.sigmoid(beta)

    return tau_init, tau_transition, pi, beta, alpha

# def initial_before_softmax(intitalization):
#     tau_init, tau_transition, pi, beta, alpha = intitalization
#     epsilon = 1e-2

#     # 각 텐서를 float32로 변환한 후 clamp와 log 적용
#     tau_init = torch.log(torch.clamp(tau_init.to(torch.float32), min=epsilon, max=1-epsilon))
#     tau_transition = torch.log(torch.clamp(tau_transition.to(torch.float32), min=epsilon, max=1-epsilon))
#     pi = torch.log(torch.clamp(pi.to(torch.float32), min=epsilon, max=1-epsilon))
#     alpha = torch.log(torch.clamp(alpha.to(torch.float32), min=epsilon, max=1-epsilon))
#     beta = torch.log(torch.clamp(beta.to(torch.float32), min=epsilon, max=1-epsilon))

#     return tau_init, tau_transition, pi, beta, alpha

def initial_before_softmax(intitalization):
    tau_init, tau_transition, pi, beta, alpha = intitalization
    epsilon = 1e-2
    # 로그를 취할 때 양수인 값만 허용
    tau_init = torch.log(torch.clamp(tau_init, min=epsilon, max = 1-epsilon))
    tau_transition = torch.log(torch.clamp(tau_transition, min=epsilon, max = 1-epsilon))
    pi = torch.log(torch.clamp(pi, min=epsilon, max = 1-epsilon))
    alpha = torch.log(torch.clamp(alpha, min=epsilon, max = 1-epsilon))
    beta = torch.log(torch.clamp(beta, min=epsilon, max = 1-epsilon))

    return tau_init, tau_transition, pi, beta, alpha

def inital_gradient(intitalization):
    tau_init, tau_transition, pi, beta, alpha = intitalization
    tau_init.requires_grad_()
    tau_transition.requires_grad_()
    pi.requires_grad_()
    beta.requires_grad_()
    alpha.requires_grad_()

def main(time_stamp = 10, num_latent = 2, num_nodes = 100, stability = 0.9, total_iteration = 0, distribution = 'Bernoulli', bernoulli_case = 'medium_plus', num_trials = 1, mode = 'new'):

    # 데이터 생성
    Y = generate_data(time_stamp = time_stamp, 
                      num_nodes = num_nodes, 
                      num_latent = num_latent, 
                      stability = stability, 
                      total_iteration = total_iteration, 
                      distribution = distribution,
                      bernoulli_case = bernoulli_case)
    
    true_loss= true_J(Y, num_nodes = num_nodes,
                        num_latent = num_latent, 
                        stability = stability, 
                        total_iteration = total_iteration, 
                        distribution = distribution, 
                        bernoulli_case = bernoulli_case)
    
    initialization_kmeans = inital_kmeans(Y,num_latent)
    torch.save(initialization_kmeans, f"parameter/{num_nodes}_{time_stamp}_{str_stability}/initialization/{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}/initialization_kmeans_{total_iteration}.pt")
    initialization_kmeans_new = tuple(tensor.clone() for tensor in initialization_kmeans)

    print("prior kmeans ------------------------------------------------------------------------------")
    prior_kmeans_loss = estimate_old(adjacency_matrix = Y, 
                                    initialization = initialization_kmeans,
                                    num_latent = num_latent, 
                                    stability = stability, 
                                    total_iteration = total_iteration, 
                                    bernoulli_case = bernoulli_case, 
                                    trial = 0,
                                    mode= 'prior_kmeans')
    
    prior_kmeans_global_ARI, prior_kmeans_average_ARI = eval(bernoulli_case =bernoulli_case, 
                                                            num_nodes = num_nodes,
                                                            time_stamp = time_stamp, 
                                                            stability = stability, 
                                                            total_iteration = total_iteration,
                                                            trial = 0,
                                                            mode = 'prior_kmeans')
    
    intitalization_kmeans_new_pre = initial_before_softmax(initialization_kmeans_new)
    inital_gradient(intitalization_kmeans_new_pre)

    print("ours kmeans ------------------------------------------------------------------------------")
    kmeans_loss = estimate(adjacency_matrix = Y, 
                        initialization = intitalization_kmeans_new_pre,
                        num_latent = num_latent, 
                        stability = stability, 
                        total_iteration = total_iteration, 
                        bernoulli_case = bernoulli_case,
                        trial = 0,
                        num_iterations = 10000,
                        mode = 'new_kmeans')
    
    kmeans_global_ARI, kmeans_average_ARI = eval(bernoulli_case =bernoulli_case, 
                                        num_nodes = num_nodes,
                                        time_stamp = time_stamp, 
                                        stability = stability, 
                                        total_iteration = total_iteration,
                                        trial = 0,
                                        mode = 'new_kmeans')
        
    
    trial_global_ARI = []
    trial_average_ARI = []
    trial_loss = []

    prior_trial_global_ARI = []
    prior_trial_average_ARI = []
    prior_trial_loss = []

    for j in range(num_trials):
        # 추정
        initialization = inital_random(Y,num_latent)
        torch.save(initialization, f"parameter/{num_nodes}_{time_stamp}_{str_stability}/initialization/{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}/initialization_{total_iteration}.pt")
    
        initialization_new = tuple(tensor.clone() for tensor in initialization)
        
        print(f"prior {j} times ------------------------------------------------------------------------------")
        prior_random_loss = estimate_old(adjacency_matrix = Y, 
                                        initialization = initialization,
                                        num_latent = num_latent, 
                                        stability = stability, 
                                        total_iteration = total_iteration, 
                                        bernoulli_case = bernoulli_case, 
                                        trial = j,
                                        mode= 'prior_random')
        
        prior_random_global_ARI, prior_random_average_ARI = eval(bernoulli_case =bernoulli_case, 
                                                                num_nodes = num_nodes,
                                                                time_stamp = time_stamp, 
                                                                stability = stability, 
                                                                total_iteration = total_iteration,
                                                                trial = j,
                                                                mode = 'prior_random')
        
        intitalization_new_pre = initial_before_softmax(initialization_new)
        inital_gradient(intitalization_new_pre)

        print(f"ours {j} times ------------------------------------------------------------------------------")
        loss = estimate(adjacency_matrix = Y, 
                        initialization = intitalization_new_pre,
                        num_latent = num_latent, 
                        stability = stability, 
                        total_iteration = total_iteration, 
                        bernoulli_case = bernoulli_case,
                        trial = j,
                        num_iterations = 5000,
                        mode = 'new_random')
    
        # loss = estimate_gpu(adjacency_matrix = Y, 
                            # num_latent = num_latent, 
                            # stability = stability, 
                            # total_iteration = total_iteration, 
                            # bernoulli_case = bernoulli_case,
                            # trial = j)

        global_ARI, average_ARI = eval(bernoulli_case =bernoulli_case, 
                                        num_nodes = num_nodes,
                                        time_stamp = time_stamp, 
                                        stability = stability, 
                                        total_iteration = total_iteration,
                                        trial = j,
                                        mode = 'new_random')
        
        trial_global_ARI.append(global_ARI)
        trial_average_ARI.append(average_ARI)
        trial_loss.append(loss.item())

        prior_trial_global_ARI.append(prior_random_global_ARI)
        prior_trial_average_ARI.append(prior_random_average_ARI)
        prior_trial_loss.append(prior_random_loss.item())


    print("True loss: ", true_loss.item())
    print("new: " ,trial_global_ARI,trial_average_ARI,trial_loss)
    print("prior: ",prior_trial_global_ARI,prior_trial_average_ARI,prior_trial_loss)

    max_index = trial_global_ARI.index(max(trial_global_ARI))
    prior_max_index = prior_trial_global_ARI.index(max(prior_trial_global_ARI))
    

        
    # 추정 결과 평가
    
    return (
        trial_global_ARI[max_index], trial_average_ARI[max_index], 
        kmeans_global_ARI, kmeans_average_ARI,
        prior_trial_global_ARI[prior_max_index], prior_trial_average_ARI[prior_max_index],  
        prior_kmeans_global_ARI, prior_kmeans_average_ARI, 
        trial_loss[max_index], true_loss.item(), kmeans_loss.item(),
        prior_trial_loss[prior_max_index], prior_kmeans_loss.item()
        )
    # return global_ARI, average_ARI, loss, true_loss
    

if __name__ == "__main__":
    time_stamp = 5
    num_latent = 2
    num_nodes = 100
    stability = 0.6
    total_iteration = 0
    distribution = 'Bernoulli'
    num_trials = 8
    
    # bernoulli_case = 'low_minus'
    # bernoulli_case = 'low_plus'
    # bernoulli_case = 'medium_minus'
    # bernoulli_case = 'medium_plus'
    bernoulli_case = 'medium_with_affiliation'
    # bernoulli_case = 'large'

    str_stability = str(stability).replace('0.', '0p')

    iterations = 100
    

    directory_output = f'output/{num_nodes}_{time_stamp}_{str_stability}'
    directory_adj = f"parameter/{num_nodes}_{time_stamp}_{str_stability}/adjacency/{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}"
    directory_true = f"parameter/{num_nodes}_{time_stamp}_{str_stability}/true/{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}"
    directory_random_est = f"parameter/{num_nodes}_{time_stamp}_{str_stability}/new_random_estimation/{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}"
    directory_kmeans_est = f"parameter/{num_nodes}_{time_stamp}_{str_stability}/new_kmeans_estimation/{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}"
    directory_prior_random_est = f"parameter/{num_nodes}_{time_stamp}_{str_stability}/prior_random_estimation/{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}"
    directory_prior_kmeans_est = f"parameter/{num_nodes}_{time_stamp}_{str_stability}/prior_kmeans_estimation/{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}"
    directory_initialization_est = f"parameter/{num_nodes}_{time_stamp}_{str_stability}/initialization/{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}"

    if not os.path.exists(directory_output):
        os.makedirs(directory_output)
    
    if not os.path.exists(directory_adj):
        os.makedirs(directory_adj)

    if not os.path.exists(directory_true):
        os.makedirs(directory_true)
    
    if not os.path.exists(directory_random_est):
        os.makedirs(directory_random_est)

    if not os.path.exists(directory_kmeans_est):
        os.makedirs(directory_kmeans_est)

    if not os.path.exists(directory_prior_random_est):
        os.makedirs(directory_prior_random_est)

    if not os.path.exists(directory_prior_kmeans_est):
        os.makedirs(directory_prior_kmeans_est)
        
    if not os.path.exists(directory_initialization_est):
        os.makedirs(directory_initialization_est)

   

    csv_file_path = f'output/{num_nodes}_{time_stamp}_{str_stability}/result_{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}.csv'
    df = pd.DataFrame(columns=["iteration", "global_ari", "average_ari", "kmeans_global_ari", "kmeans_average_ari","prior_random_global_ari", "prior_random_average_ari","prior_kmeans_global_ari", "prior_kmeans_average_ari", "true_loss", "loss", "kmeans_loss","prior_random_loss", "prior_kmeans_loss" ])
    df.to_csv(csv_file_path, index=False)
    
    for i in tqdm(range(iterations)):
        print("-----------------------------------------------------------------------------------------")
        print("-----------------------------------------------------------------------------------------")
        print(f"This iteration is {i}")
        print("-----------------------------------------------------------------------------------------")
        print("-----------------------------------------------------------------------------------------")

        global_ARI, average_ARI, \
        kmeans_global_ARI, kmeans_average_ARI, \
        prior_random_global_ARI, prior_random_average_ARI, \
        prior_kmeans_global_ARI, prior_kmeans_average_ARI, \
        loss, true_loss, kmeans_loss,\
        prior_random_loss, prior_kmeans_loss = main(time_stamp = time_stamp, 
                                                    num_latent = num_latent, 
                                                    num_nodes = num_nodes, 
                                                    stability = stability, 
                                                    total_iteration = i, 
                                                    distribution = distribution, 
                                                    bernoulli_case = bernoulli_case,
                                                    num_trials = num_trials)
        
        iteration_data = pd.DataFrame([{"iteration": i,
                                        "global_ari": global_ARI,
                                        "average_ari": average_ARI,
                                        "kmeans_global_ari": kmeans_global_ARI,
                                        "kmeans_average_ari": kmeans_average_ARI,
                                        "prior_random_global_ari": prior_random_global_ARI,
                                        "prior_random_average_ari": prior_random_average_ARI,
                                        "prior_kmeans_global_ari": prior_kmeans_global_ARI,
                                        "prior_kmeans_average_ari": prior_kmeans_average_ARI,
                                        "true_loss": -true_loss,
                                        "loss": -loss,
                                        "kmeans_loss" : -kmeans_loss,
                                        "prior_random_loss" : -prior_random_loss,
                                        "prior_kmeans_global_ARI" : -prior_kmeans_loss
                                        }])
        
        iteration_data.to_csv(csv_file_path, mode='a', header=False, index=False)
