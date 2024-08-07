from adj_generator import generate_data
from model_compact import estimate
from configuration import eval
from true_model import true_J
import numpy as np
from tqdm import tqdm
import pandas as pd

def main(time_stamp = 10, num_latent = 2, num_nodes = 100, stability = 0.9, iteration = 0, distribution = 'Bernoulli', bernoulli_case = 'medium_plus'):

    # 데이터 생성
    Y = generate_data(time_stamp = time_stamp, 
                      num_latent = num_latent, 
                      num_nodes = num_nodes, 
                      stability = stability, 
                      iteration = iteration, 
                      distribution = distribution,
                      bernoulli_case = bernoulli_case)
    
    true_loss= true_J(Y ,num_latent = num_latent, 
                      stability = stability, 
                      iteration = iteration, 
                      distribution = distribution, 
                      bernoulli_case = bernoulli_case)
    # 데이터 추정
    loss = estimate(adjacency_matrix = Y, 
                    num_latent = num_latent, 
                    stability = stability, 
                    iteration = iteration, 
                    bernoulli_case = bernoulli_case)
   
    # 추정 결과 평가
    global_ARI, average_ARI  = eval(bernoulli_case =bernoulli_case, 
                             time_stamp = time_stamp , 
                             stability = stability, 
                             iteration = iteration)
    return global_ARI, average_ARI, loss, true_loss
    

if __name__ == "__main__":
    time_stamp = 10
    num_latent = 2
    num_nodes = 100
    stability = 0.9
    iteration = 0
    distribution = 'Bernoulli'

    # bernoulli_case = 'low_minus'
    # bernoulli_case = 'low_plus'
    # bernoulli_case = 'medium_minus'
    bernoulli_case = 'medium_plus'
    # bernoulli_case = 'medium_with_affiliation'
    # bernoulli_case = 'large'

    str_stability = str(stability).replace('0.', '0p')

    iterations = 100
    csv_file_path = f'output/result_{bernoulli_case}_{time_stamp}_{str_stability}.csv'
    df = pd.DataFrame(columns=["iteration", "global_ari", "average_ari", "loss", "true_loss"])
    df.to_csv(csv_file_path, index=False)

    for i in tqdm(range(iterations)):
        print("-------------------------------------")
        print(f"This iteration is {i}")
        global_ARI, average_ARI, loss, true_loss = main(time_stamp = time_stamp, 
                                                        num_latent = num_latent, 
                                                        num_nodes = num_nodes, 
                                                        stability = stability, 
                                                        iteration = iteration, 
                                                        distribution = distribution, 
                                                        bernoulli_case = bernoulli_case)
        iteration_data = pd.DataFrame([{"iteration": i,
                                        "global_ari": global_ARI,
                                        "average_ari": average_ARI,
                                        "loss": loss.item(),
                                        "true_loss": true_loss.item()}])
        
        iteration_data.to_csv(csv_file_path, mode='a', header=False, index=False)