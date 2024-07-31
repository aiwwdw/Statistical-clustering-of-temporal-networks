import torch
import torch.nn.functional as F
from sklearn.metrics import adjusted_rand_score

def tau_margin_generator(tau_init, tau_transition):
    T,N,Q,_ = tau_transition.shape
    tau = torch.zeros((T, N, Q))
    for t in range(T):
        result = tau_init[:, :]
        for t_prime in range(t):
            # print(result,tau_transition[t_prime, :, :,:])
            result = torch.einsum('ij,ijk->ik', result, tau_transition[t_prime+1, :, :,:])
            # print(result)
        tau[t,:,:] = result
    return tau


# Load the parameters from the file
pi, alpha, beta, tau_init, tau_transition = torch.load('para_old_MP_LPB.pt')
init_dist,transition_matrix, Bernoulli_parameter, Z = torch.load('true_para_MG_LPB.pt')
Z = torch.tensor(Z, dtype=torch.int64)

# ### In case of My model
# tau_init = F.softmax(tau_init, dim=1)
# tau_transition = F.softmax(tau_transition, dim=3)
# alpha = F.softmax(alpha, dim=0)
# pi = F.softmax(pi,dim=1)
# beta = F.sigmoid(beta)


tau_marg = tau_margin_generator(tau_init, tau_transition)


# Print each parameter
print("pi:", pi)
print("alpha:", alpha)
print("beta:", beta)

# print("tau_init:", tau_init)
# print("tau_transition:", tau_transition)
# print("tau_marg:", tau_marg)

# all_pred = []
# all_true = [] 
# for time in range(5):
#     print("time is", time)
#     indices = torch.argmax(tau_marg[time], dim=1)
#     pred = indices
#     true = Z[time]
#     all_pred.extend(pred)
#     all_true.extend(true)
#     print(pred)
#     print(true)
#     difference = torch.sum(pred != true).item()
#     print("Difference count:", difference)
#     ari_score = adjusted_rand_score(true, pred)
#     print("Adjusted Rand Index:", ari_score)  

# overall_ari_score = adjusted_rand_score(all_true, all_pred)
# print("Overall Adjusted Rand Index:", overall_ari_score)  
