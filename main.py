from adj_generator import generate_data
from model_compact import estimate
from evaluation import eval
from true_model import true_J
from model_gpu import estimate_gpu
import numpy as np
from tqdm import tqdm
import pandas as pd
import os

def main(time_stamp = 10, num_latent = 2, num_nodes = 100, stability = 0.9, total_iteration = 0, distribution = 'Bernoulli', bernoulli_case = 'medium_plus', num_trials = 1):

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
        #                     num_latent = num_latent, 
        #                     stability = stability, 
        #                     iteration = total_iteration, 
        #                     bernoulli_case = bernoulli_case)

        global_ARI, average_ARI = eval(bernoulli_case =bernoulli_case, 
                                        num_nodes = num_nodes,
                                        time_stamp = time_stamp, 
                                        stability = stability, 
                                        total_iteration = total_iteration,
                                        trial = j)
        
        trial_global_ARI.append(global_ARI)
        trial_average_ARI.append(average_ARI)
        trial_loss.append(loss.item())
    print(true_loss.item())
    print(trial_global_ARI,trial_average_ARI,trial_loss)

    max_index = trial_global_ARI.index(max(trial_global_ARI))
    

        
    # 추정 결과 평가
    
    return trial_global_ARI[max_index], trial_average_ARI[max_index], trial_loss[max_index], true_loss.item()
    # return global_ARI, average_ARI, loss, true_loss
    

if __name__ == "__main__":
    time_stamp = 5
    num_latent = 2
    num_nodes = 100
    stability = 0.75
    total_iteration = 0
    distribution = 'Bernoulli'
    num_trials = 2

    # bernoulli_case = 'low_minus'
    # bernoulli_case = 'low_plus'
    # bernoulli_case = 'medium_minus'
    bernoulli_case = 'medium_plus'
    # bernoulli_case = 'medium_with_affiliation'
    # bernoulli_case = 'large'

    str_stability = str(stability).replace('0.', '0p')

    iterations = 100
    csv_file_path = f'output/result_{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}.csv'
    df = pd.DataFrame(columns=["iteration", "global_ari", "average_ari", "loss", "true_loss"])
    df.to_csv(csv_file_path, index=False)


    directory_adj = f"parameter/adjacency/{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}"
    directory_true = f"parameter/true/{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}"
    directory_est = f"parameter/estimation/{bernoulli_case}_{num_nodes}_{time_stamp}_{str_stability}"

    if not os.path.exists(directory_adj):
        os.makedirs(directory_adj)

    if not os.path.exists(directory_true):
        os.makedirs(directory_true)
    
    if not os.path.exists(directory_est):
        os.makedirs(directory_est)
    
    for i in tqdm(range(iterations)):
        print("-------------------------------------")
        print(f"This iteration is {i}")
        global_ARI, average_ARI, loss, true_loss = main(time_stamp = time_stamp, 
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
                                        "loss": -loss,
                                        "true_loss": -true_loss}])
        
        iteration_data.to_csv(csv_file_path, mode='a', header=False, index=False)