import os
import pandas as pd
import numpy as np
import argparse
import torch
import torch.nn.functional as F
import random
import pickle

from MFGP import *
from MFGP.utils.normalizer import Dateset_normalize_manager


def arg_parse():
    """Parsing arguments"""
    parser = argparse.ArgumentParser(
        description="AutoML Multi-Fidelity Optimization"
    )
    parser.add_argument("--cfg", required=True, help="path to config file", type=str)
    # parser.add_argument("--gpus", default=None, help="gpu id(s) to use", type=str)
    parser.add_argument("--resume", default="", type=str)
    args = parser.parse_args()
    return args


# def normalize_data(tr_x, eval_x, tr_y_list, eval_y_list):
#     # normalize
#     norm_tool = Dateset_normalize_manager(tr_x, tr_y_list)
#     tr_x = [norm_tool.normalize_input(_, 0) for _ in tr_x]
#     tr_y_list = norm_tool.normalize_outputs(tr_y_list)
#     eval_x = norm_tool.normalize_input(eval_x, 0)
#     eval_y_list = norm_tool.normalize_outputs(eval_y_list)
#
#     return tr_x, eval_x, tr_y_list, eval_y_list, norm_tool


def normalize_data(tr_x, tr_y_list):
    # normalize
    norm_tool = Dateset_normalize_manager(tr_x, tr_y_list)
    # tr_x = [norm_tool.normalize_input(_, 0) for _ in tr_x]
    tr_x = norm_tool.normalize_input(tr_x, 0)
    tr_y_list = norm_tool.normalize_outputs(tr_y_list)

    return tr_x, tr_y_list, norm_tool


model_dict = {
    'AR': AR,
    'CIGAR': CIGAR,
    'GAR': GAR,
    'CAR': CAR,
    'NAR': NAR,
    'ResGP': ResGP,
    'CIGP': CIGP,
    'HOGP': HOGP,
}


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


def find_matching_indices(x_1, x_2):
    # return the indices of x_1 that are in x_2
    matching_indices = []
    for i, row in enumerate(x_2):
        # Check if the current row of x_2 is in x_1
        matches = (x_1 == row).all(dim=1)
        if matches.any():
            matching_indices.extend(torch.where(matches)[0].tolist())
    return matching_indices

def load_data(data_root, benchmark="lcbench", task_name="airlines"):
    data_dir = os.path.join(data_root, "{}-{}".format(benchmark, task_name))

    param_cfgs = pd.read_csv("{}/hyperparameter_index.csv".format(data_dir))
    y_valid = pd.read_csv("{}/final_data.csv".format(data_dir), header=None, dtype=float)

    param_cfgs = param_cfgs
    # y_valid = y_valid_[:, 0, :, 0]
    # np.save('/home/haolin/VSCode/tiny_objectives.npy', y_valid)
    # y_valid = -1 * y_valid_  # why negative?
    # replace "tanh", "relu", "cosine", "const" with numbers:
    if benchmark == "fcnet":
        param_cfgs = param_cfgs.replace("tanh", 0)
        param_cfgs = param_cfgs.replace("relu", 1)
        param_cfgs = param_cfgs.replace("cosine", 0)
        param_cfgs = param_cfgs.replace("const", 1)

    param_cfgs_tensor = torch.tensor(param_cfgs.values).type(torch.float64)
    y_valid_tensor = torch.tensor(y_valid.values).type(torch.float64)

    return param_cfgs_tensor, y_valid_tensor


def dump_list(data, out_file_path):
    """
    Dumps a list to a file.
    Args:
        data: python list
        out_file_path: string

    Returns:

    """
    with open(out_file_path, 'wb') as f:
        pickle.dump(data, f)
    f.close()
