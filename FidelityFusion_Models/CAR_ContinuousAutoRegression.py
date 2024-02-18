import sys
import os
# sys.path.append(r'H:\\eda\\mybranch')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import GaussianProcess.kernel as kernel
from GaussianProcess.gp_basic import GP_basic as CIGP
from FidelityFusion_Models.MF_data import MultiFidelityDataManager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn.functional as F

def warp_function(lf, hf):
    return lf, hf

class fidelity_kernel_MCMC(nn.Module):
    """
    fidelity kernel module base ARD and use MCMC to calculate the integral.

    Args:
        input_dim (int): The input dimension.
        initial_length_scale (float): The initial length scale value. Default is 1.0.
        initial_signal_variance (float): The initial signal variance value. Default is 1.0.
        eps (float): A small constant to prevent division by zero. Default is 1e-9.

    Attributes:
        length_scales (nn.Parameter): The length scales for each dimension.
        signal_variance (nn.Parameter): The signal variance.
        eps (float): A small constant to prevent division by zero.

    """

    def __init__(self, input_dim, kernel1, lf, hf, b, initial_length_scale=1.0, initial_signal_variance=1.0, eps=1e-3):
        super().__init__()
        self.kernel1 = kernel1
        self.b = b
        self.lf = lf
        self.hf = hf
        self.length_scales = nn.Parameter(torch.ones(input_dim) * initial_length_scale)
        self.signal_variance = nn.Parameter(torch.tensor([initial_signal_variance]))
        self.eps = eps
        self.seed = 105

    def forward(self, x1, x2):
        """
        Compute the covariance matrix using the ARD kernel.

        Args:
            x1 (torch.Tensor): The first input tensor.
            x2 (torch.Tensor): The second input tensor.

        Returns:
            torch.Tensor: The covariance matrix.

        """
        # length_scales = torch.abs(self.length_scales) + self.eps
        length_scales = torch.tensor(1.)
        N = 100
        torch.manual_seed(self.seed)
        # print(torch.rand(1))
        z1 = torch.rand(N) * (self.hf - self.lf) + self.lf # 这块需要用来调整z选点的范围
        z2 = torch.rand(N) * (self.hf - self.lf) + self.lf

        dist_z = (z1 / length_scales - z2 / length_scales) ** 2
        z_part1 = -self.b * (z1 - self.hf)
        z_part2 = -self.b * (z2 - self.hf)
        z_part  = (z_part1 + z_part2 - 0.5 * dist_z).exp()
        z_part_mc = z_part.mean() * (self.hf - self.lf) * (self.hf - self.lf)

        return self.signal_variance.abs() * z_part_mc * self.kernel1(x1, x2)

class ContinuousAutoRegression(nn.Module):
    # initialize the model
    def __init__(self, fidelity_num, kernel_list, b_init=1.0):
        super().__init__()
        self.fidelity_num = fidelity_num
        self.b = torch.nn.Parameter(torch.tensor(b_init))

        # create the model
        self.cigp_list=[]
        self.cigp_list.append(CIGP(kernel=kernel_list[0], noise_variance=1.0))

        for fidelity_low in range(self.fidelity_num - 1):
            low_fidelity_indicator, high_fidelity_indicator = warp_function(fidelity_low, fidelity_low+1)
            input_dim = kernel_list[0].length_scales.shape[0]
            kernel_residual = fidelity_kernel_MCMC(input_dim, kernel_list[fidelity_low+1],
                                                   low_fidelity_indicator, high_fidelity_indicator, self.b)
            self.cigp_list.append(CIGP(kernel=kernel_residual, noise_variance=1.0))
        
        self.cigp_list = torch.nn.ModuleList(self.cigp_list)

        # self.rho_list=[]
        # for _ in range(self.fidelity_num-1):
        #     self.rho_list.append(torch.nn.Parameter(torch.tensor(b_init)))
        # self.rho_list = torch.nn.ParameterList(self.rho_list)

    def forward(self, data_manager, x_test):
        # predict the model
        for f in range(self.fidelity_num):
            if f == 0:
                x_train,y_train = data_manager.get_data(f)
                y_pred_low, cov_pred_low = self.cigp_list[f](x_train,y_train,x_test)
                if self.fidelity_num == 1:
                    y_pred_high = y_pred_low
                    cov_pred_high = cov_pred_low
            else:
                x_train,y_train = data_manager.get_data(-f)
                y_pred_res, cov_pred_res= self.cigp_list[f](x_train,y_train,x_test)
                y_pred_high = y_pred_low + self.b * y_pred_res
                cov_pred_high = cov_pred_low + (self.b **2) * cov_pred_res

                ## for next fidelity
                y_pred_low = y_pred_high
                cov_pred_low = cov_pred_high

        # return the prediction
        return y_pred_high, cov_pred_high
    
def train_CAR(CARmodel, data_manager,max_iter=1000,lr_init=1e-1):
    
    for f in range(CARmodel.fidelity_num):
        optimizer = torch.optim.Adam(CARmodel.parameters(), lr=lr_init)
        if f == 0:
            x_low,y_low = data_manager.get_data(f)
            for i in range(max_iter):
                optimizer.zero_grad()
                loss = -CARmodel.cigp_list[f].log_likelihood(x_low, y_low)
                loss.backward()
                optimizer.step()
                print('fidelity:', f, 'iter', i, 'nll:{:.5f}'.format(loss.item()))
        else:
            _, y_low, subset_x,y_high = data_manager.get_overlap_input_data(f-1,f)
            for i in range(max_iter):
                optimizer.zero_grad()
                y_residual = y_high - CARmodel.b.exp() * y_low # 修改
                if i == max_iter-1:
                    data_manager.add_data(fidelity_index=-f,raw_fidelity_name='res-{}'.format(f),x=subset_x,y=y_residual)
                loss = -CARmodel.cigp_list[f].log_likelihood(subset_x, y_residual)
                loss.backward()
                optimizer.step()
                print('fidelity:', f, 'iter', i,'b:',CARmodel.b.item(), 'nll:{:.5f}'.format(loss.item()))



def precision_at_k(predicted_ranking, ground_truth, k):  # overlap precision
    predicted_ranking = predicted_ranking.flatten()
    ground_truth = ground_truth.flatten()

    # Get the top k predictions
    top_k_predictions = set(predicted_ranking[:k].tolist())

    # Convert ground_truth to a set of items for efficient checking
    ground_truth_set = set(ground_truth[:k].tolist())

    # Count the number of relevant items in the top k predictions
    num_relevant_items = sum([1 for item in top_k_predictions if item in ground_truth_set])

    # Calculate precision
    precision = num_relevant_items / k
    return precision

# demo 
if __name__ == "__main__":

    torch.manual_seed(1)
    hyperparameter_index = pd.read_csv(
        "/home/haolin/VSCode/automl2024/benchmarking/nursery/benchmark_new/data/fcnet-protein/hyperparameter_index.csv")
    objectives_evaluations = np.load(
        "/home/haolin/VSCode/automl2024/benchmarking/nursery/benchmark_new/data/fcnet-protein/objectives_evaluations.npy")
    print(objectives_evaluations.shape)
    tiny_hyperparameter_index = hyperparameter_index[:1000]

    tiny_objectives_default = objectives_evaluations[:1000, 0, :, 0]
    # np.save('/home/haolin/VSCode/tiny_objectives.npy', tiny_objectives_default)

    # replace "tanh", "relu", "cosine", "const" with numbers:
    tiny_hyperparameter_index = tiny_hyperparameter_index.replace("tanh", 0)
    tiny_hyperparameter_index = tiny_hyperparameter_index.replace("relu", 1)
    tiny_hyperparameter_index = tiny_hyperparameter_index.replace("cosine", 0)
    tiny_hyperparameter_index = tiny_hyperparameter_index.replace("const", 1)
    tiny_train_set = tiny_objectives_default
    # np.save('/home/haolin/VSCode/tiny_train.csv', tiny_train_set)

    tiny_train_x = tiny_hyperparameter_index
    tiny_train_y = tiny_train_set
    np_tiny_train_x = tiny_train_x.to_numpy()
    tensor_tiny_train_x = torch.tensor(np_tiny_train_x).float()

    tensor_y_list = []

    for i in range(tiny_train_y.shape[1]):  # np_array.shape[1] is 100
        # Extract the i-th column and convert it to a PyTorch tensor of shape (1000,)
        col_tensor = torch.tensor(tiny_train_y[:, i])

        # Reshape the tensor to (1, 1000) and add it to the list
        tensor_y_list.append(col_tensor.unsqueeze(1))

    sample_num = tiny_objectives_default.shape[0]

    normalized_x = F.softmax(tensor_tiny_train_x, dim=1)
    entropy = -torch.sum(normalized_x * torch.log(normalized_x), dim=1)
    num_rows_to_select = 50
    _, indices = torch.topk(entropy, num_rows_to_select)
    selected_rows = tensor_tiny_train_x[indices]
    selected_x = [tensor_tiny_train_x]
    selected_y = [tensor_y_list[0]]
    for i in range(tiny_train_y.shape[1]-1):
        selected_x.append(tensor_tiny_train_x[indices])
        selected_y.append(tensor_y_list[i+1][indices])
    # selected_y should be a list, list[k] is a tensor of tiny_train_y[indices, k]:
    # selected_y = [tensor_y[indices] for tensor_y in tensor_y_list[1:]]

    num_fidelity_indicators = len(selected_y)

    initial_data = [
        {
            'fidelity_indicator': k,
            'raw_fidelity_name': str(k),
            'X': selected_x[k],  # Assign the same 'X' value for each item
            'Y': selected_y[k]  # Assign 'Y' from the k-th item of selected_y
        }
        for k in range(num_fidelity_indicators)  # Iterate over the range of fidelity indicators
    ]
##################################################
    # generate the data
    # x_all = torch.rand(500, 1) * 20
    # xlow_indices = torch.randperm(500)[:300]
    # xlow_indices = torch.sort(xlow_indices).values
    # x_low = x_all[xlow_indices]
    # xhigh1_indices = torch.randperm(500)[:300]
    # xhigh1_indices = torch.sort(xhigh1_indices).values
    # x_high1 = x_all[xhigh1_indices]
    # xhigh2_indices = torch.randperm(500)[:250]
    # xhigh2_indices = torch.sort(xhigh2_indices).values
    # x_high2 = x_all[xhigh2_indices]
    # x_test = torch.linspace(0, 20, 100).reshape(-1, 1)
    #
    # y_low = torch.sin(x_low) - torch.rand(300, 1) * 0.2
    # y_high1 = torch.sin(x_high1) - torch.rand(300, 1) * 0.1
    # y_high2 = torch.sin(x_high2) + torch.rand(250, 1) * 0.1 - 0.05
    # y_test = torch.sin(x_test)
    #
    # initial_data = [
    #     {'fidelity_indicator': 0,'raw_fidelity_name': '0', 'X': x_low, 'Y': y_low},
    #     {'fidelity_indicator': 1, 'raw_fidelity_name': '1','X': x_high1, 'Y': y_high1},
    #     {'fidelity_indicator': 2, 'raw_fidelity_name': '2','X': x_high2, 'Y': y_high2},
    # ]

    fidelity_manager = MultiFidelityDataManager(initial_data)
    fidelity_num = 100
    kernel_list = [kernel.ARDKernel(selected_rows.shape[1]) for _ in range(fidelity_num)]
    # kernel_residual = fidelity_kernel_MCMC(x_low.shape[1], kernel.ARDKernel(x_low.shape[1]), 1, 2)
    CAR = ContinuousAutoRegression(fidelity_num=fidelity_num, kernel_list=kernel_list, b_init=1.0)

    train_CAR(CAR,fidelity_manager, max_iter=100, lr_init=1e-2)

    with torch.no_grad():
        ypred, ypred_var = CAR(fidelity_manager,tensor_tiny_train_x)
    ypred_reshape = ypred.unsqueeze(1)
    joined_y = torch.cat((tensor_y_list[-1], ypred_reshape), 1)
    joined_y = joined_y[indices]

    predict_y = ypred[indices]
    groundtruth = tensor_y_list[-1][indices]
    evaluation_num = len(groundtruth)
    sorted_predict_y, sorted_predict_y_indices = torch.sort(predict_y, dim=0)
    sorted_groundtruth, sorted_groundtruth_indices = torch.sort(groundtruth, dim=0)

    for k in [5, 10, 20]:
        top_k = int((k / 100) * evaluation_num)
        top_k_predict_indices = sorted_predict_y_indices[:top_k]
        top_k_groundtruth_indices = sorted_groundtruth_indices[:top_k]

        # # Compute the intersection of indices
        # _, indices_in_top_k_predict = torch.where(
        #     torch.isin(top_k_groundtruth_indices, top_k_predict_indices)
        # )
        similarity = sum(1 for i in range(top_k) if top_k_predict_indices[i] == top_k_groundtruth_indices[i])

        # similarity = indices_in_top_k_predict.numel()
        print(f"Strict Precision @{k}%: {similarity}")

    # Calculate overlap precision at 5%, 10%, 20% of all evaluations, evaluation number should be calculated first

    for k in [5, 10, 20]:
        top_k = int((k / 100) * evaluation_num)
        similarity = precision_at_k(predicted_ranking=sorted_predict_y_indices, ground_truth=sorted_groundtruth_indices,
                                    k=top_k)
        print(f"Overlap Precision @{k}%: {similarity}")

    mse = torch.mean((groundtruth - predict_y) ** 2)
    print("MSE:", mse)

    from scipy.stats import spearmanr
    from scipy.stats import kendalltau

    # Assuming groundtruth and predict_y are your rank vectors
    spearman_correlation, _ = spearmanr(groundtruth, predict_y)
    print("Spearman correlation:", spearman_correlation)

    # Assuming groundtruth and predict_y are your rank vectors
    kendall_tau, _ = kendalltau(groundtruth, predict_y)

    print(f"Kendall's Tau coefficient: {kendall_tau}")
    #
    # plt.figure()
    # plt.errorbar(tensor_tiny_train_x.flatten(), ypred.reshape(-1).detach(), ypred_var.diag().sqrt().squeeze().detach(), fmt='r-.' ,alpha = 0.2)
    # plt.fill_between(tensor_tiny_train_x.flatten(), ypred.reshape(-1).detach() - ypred_var.diag().sqrt().squeeze().detach(), ypred.reshape(-1).detach() + ypred_var.diag().sqrt().squeeze().detach(), alpha=0.2)
    # plt.plot(tensor_tiny_train_x.flatten(), tensor_y_list[-1], 'k+')
    # plt.show()