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
# selected_y should be a list, list[k] is a tensor of tiny_train_y[indices, k]:
selected_y = [tensor_y[indices] for tensor_y in tensor_y_list]

num_fidelity_indicators = len(selected_y)

initial_data = [
    {
        'fidelity_indicator': k,
        'raw_fidelity_name': str(k),
        'X': selected_rows,  # Assign the same 'X' value for each item
        'Y': selected_y[k]  # Assign 'Y' from the k-th item of selected_y
    }
    for k in range(num_fidelity_indicators)  # Iterate over the range of fidelity indicators
]