from adj_generator import generate_data
from model_compact import estimate
from evaluation import eval
from true_model import true_J
from model_gpu import estimate_gpu
from old_model_compact import estimate_old
import numpy as np
from tqdm import tqdm
import pandas as pd
import os

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
    
    prior_random_loss = estimate_old(adjacency_matrix = Y, 
                        num_latent = num_latent, 
                        stability = stability, 
                        total_iteration = total_iteration, 
                        bernoulli_case = bernoulli_case, 
                        trial = 0,
                        mode= 'prior_random')
    
    prior_random_global_ARI, prior_random_average_ARI = eval(bernoulli_case =bernoulli_case, 
                                                            num_nodes = num_nodes,
                                                            time_stamp = time_stamp, 
                                                            stability = stability, 
                                                            total_iteration = total_iteration,
                                                            trial = 0,
                                                            mode = 'prior_random')
                        
    prior_kmeans_loss = estimate_old(adjacency_matrix = Y, 
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
    


    
    trial_global_ARI = []
    trial_average_ARI = []
    trial_loss = []

    for j in range(num_trials):
        # 추정
        loss = estimate(adjacency_matrix = Y, 
                        num_latent = num_latent, 
                        stability = stability, 
                        total_iteration = total_iteration, 
                        bernoulli_case = bernoulli_case,
                        trial = j)
    
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
                                        mode = 'new')
        
        trial_global_ARI.append(global_ARI)
        trial_average_ARI.append(average_ARI)
        trial_loss.append(loss.item())
    print(true_loss.item())
    print(trial_global_ARI,trial_average_ARI,trial_loss)

    max_index = trial_global_ARI.index(max(trial_global_ARI))
    

        
    # 추정 결과 평가
    
    return (
        trial_global_ARI[max_index], trial_average_ARI[max_index], 
        prior_random_global_ARI, prior_random_average_ARI,  
        prior_kmeans_global_ARI, prior_kmeans_average_ARI, 
        trial_loss[max_index], true_loss.item(),
        prior_random_loss.item(), prior_kmeans_loss.item()
        )
    # return global_ARI, average_ARI, loss, true_loss
    

if __name__ == "__main__":
    time_stamp = 5
    num_latent = 2
    num_nodes = 100
    stability = 0.75
    total_iteration = 0
    distribution = 'Bernoulli'
    num_trials = 4
    
    bernoulli_case = 'low_minus'
    # bernoulli_case = 'low_plus'
    # bernoulli_case = 'medium_minus'
    # bernoulli_case = 'medium_plus'
    # bernoulli_case = 'medium_with_affiliation'
    # bernoulli_case = 'large'

    str_stability = str(stability).replace('0.', '0p')

    iterations = 100
    

    directory_output = f'output/{num_nodes}_{time_stamp}_{str_stability}'
    directory_adj = f"parameter/{num_nodes}_{time_stamp}_{str_stability}/adjacency/{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}"
    directory_true = f"parameter/{num_nodes}_{time_stamp}_{str_stability}/true/{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}"
    directory_est = f"parameter/{num_nodes}_{time_stamp}_{str_stability}/new_estimation/{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}"
    directory_prior_random_est = f"parameter/{num_nodes}_{time_stamp}_{str_stability}/prior_random_estimation/{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}"
    directory_prior_kmeans_est = f"parameter/{num_nodes}_{time_stamp}_{str_stability}/prior_kmeans_estimation/{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}"

    if not os.path.exists(directory_adj):
        os.makedirs(directory_adj)

    if not os.path.exists(directory_true):
        os.makedirs(directory_true)
    
    if not os.path.exists(directory_est):
        os.makedirs(directory_est)

    if not os.path.exists(directory_prior_random_est):
        os.makedirs(directory_prior_random_est)

    if not os.path.exists(directory_prior_kmeans_est):
        os.makedirs(directory_prior_kmeans_est)

    if not os.path.exists(directory_output):
        os.makedirs(directory_output)

    csv_file_path = f'output/{num_nodes}_{time_stamp}_{str_stability}/result_{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}.csv'
    df = pd.DataFrame(columns=["iteration", "global_ari", "average_ari", "prior_random_global_ari", "prior_random_average_ari","prior_kmeans_global_ari", "prior_kmeans_average_ari", "true_loss", "loss", "prior_random_loss", "prior_kmeans_loss" ])
    df.to_csv(csv_file_path, index=False)
    
    for i in tqdm(range(iterations)):
        print("-----------------------------------------------------------------------------------------")
        print(f"This iteration is {i}")

        global_ARI, average_ARI, \
        prior_random_global_ARI, prior_random_average_ARI, \
        prior_kmeans_global_ARI, prior_kmeans_average_ARI, \
        loss, true_loss, \
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
                                        "prior_random_global_ari": prior_random_global_ARI,
                                        "prior_random_average_ari": prior_random_average_ARI,
                                        "prior_kmeans_global_ari": prior_kmeans_global_ARI,
                                        "prior_kmeans_average_ari": prior_kmeans_average_ARI,
                                        "true_loss": -true_loss,
                                        "loss": -loss,
                                        "prior_random_loss" : -prior_random_loss,
                                        "prior_kmeans_global_ARI" : -prior_kmeans_loss
                                        }])
        
        iteration_data.to_csv(csv_file_path, mode='a', header=False, index=False)