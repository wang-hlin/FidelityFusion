import sys

import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import random

from MFGP import *
from MFGP.utils.normalizer import Dateset_normalize_manager


# def get_testing_data(fidelity_num):
#     x = np.load(r'assets/sample_data/input.npy')
#     y_list = [np.load(r'assets/sample_data/output_fidelity_{}.npy'.format(i)) for i in range(3)]
#     y_list = y_list[:fidelity_num]
#
#     x = torch.tensor(x)
#     y_list = [torch.tensor(_) for _ in y_list]
#
#     sample_num = x.shape[0]
#     tr_x = x[:sample_num//2, ...].float()
#     eval_x = x[sample_num//2:, ...].float()
#     tr_y_list = [y[:sample_num//2, ...].float() for y in y_list]
#     eval_y_list = [y[sample_num//2:, ...].float() for y in y_list]
#
#     return tr_x, eval_x, tr_y_list, eval_y_list

def get_testing_data_hpo():
    hyperparameter_index = pd.read_csv(
        "/home/haolin/VSCode/automl2024/benchmarking/nursery/benchmark_new/data/fc_new/hyperparameter_index.csv")
    objectives_evaluations = pd.read_csv(
        "/home/haolin/VSCode/automl2024/benchmarking/nursery/benchmark_new/data/fc_new/final_data.csv", dtype=float)

    # tiny_hyperparameter_index = hyperparameter_index[:1000]
    tiny_hyperparameter_index = hyperparameter_index
    # tiny_objectives_default = objectives_evaluations[:1000, 0, :, 0]
    # tiny_objectives_default = objectives_evaluations[:, 0, :, 0]
    tiny_objectives_default = objectives_evaluations.to_numpy()
    # np.save('/home/haolin/VSCode/tiny_objectives.npy', tiny_objectives_default)
    print(tiny_objectives_default.shape)
    tiny_objectives_default = -tiny_objectives_default
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
    tensor_tiny_train_x = torch.tensor(np_tiny_train_x).type(torch.float64)

    tensor_y_list = []

    for i in range(tiny_train_y.shape[1]):  # np_array.shape[1] is 100
        # Extract the i-th column and convert it to a PyTorch tensor of shape (1000,)
        col_tensor = torch.tensor(tiny_train_y[:, i])

        # Reshape the tensor to (1, 1000) and add it to the list
        tensor_y_list.append(col_tensor.unsqueeze(1))

    sample_num = tiny_objectives_default.shape[0]

    # sample_num = tiny_objectives_default.shape[0]
    #
    # tr_x = torch.tensor(tensor_tiny_train_x[:sample_num//2, ...]).float()
    # eval_x = torch.tensor(tensor_tiny_train_x[sample_num//2:, ...]).float()
    # tr_y_list = [y[:sample_num//2, ...].float() for y in tensor_y_list]
    # eval_y_list = [y[sample_num//2:, ...].float() for y in tensor_y_list]
    #
    # # tr_x_sliced = tr_x[:, :50]
    # # tr_x = np.concatenate((tr_x_sliced, tr_x[:, -1:]), axis=1)
    # # tr_y_list = tr_y_list[:50] + [tr_y_list[-1]]
    # tr_y_list = tr_y_list[:50]
    # eval_y_list = eval_y_list[:50] + [eval_y_list[-1]]
    # # eval_y_list = eval_y_list[:50]
    # #########
    #
    # print(tiny_train_x.shape)
    # print(tiny_train_y.shape)
    fidelity_num = tiny_objectives_default.shape[1]
    return tensor_tiny_train_x, tensor_tiny_train_x, tensor_y_list, tensor_y_list, sample_num, fidelity_num


# ##############
# def normalize_tensors(tr_x_list, eval_x_list, tr_y_list, eval_y_list):
#     """
#     Normalize lists of tensors and return parameters for denormalization.
#
#     Args:
#     - tensor_lists (list of lists of Tensors): The lists containing the tensors to normalize.
#
#     Returns:
#     - normalized_lists (list of lists of Tensors): The normalized tensor lists.
#     - norm_tool (dict): A tool containing the means and stds for denormalization.
#     """
#     means = []
#     stds = []
#
#     concat_tr_x_list = [torch.cat(tr_x, dim=0) for tr_x in tr_x_list]
#     mean_tr_x = [concat_tr_x.mean() for concat_tr_x in concat_tr_x_list]
#     std_tr_x = [concat_tr_x.std() for concat_tr_x in concat_tr_x_list]
#     normalized_tr_x_list = [[(t - mean) / std for t in tr_x] for tr_x, mean, std in zip(tr_x_list, mean_tr_x, std_tr_x)]
#
#     concat_tr_y_list = [torch.cat(tr_y, dim=0) for tr_y in tr_y_list]
#     mean_tr_y_list = [concat_tr_y.mean() for concat_tr_y in concat_tr_y_list]
#     std_tr_y_list = [concat_tr_y.std() for concat_tr_y in concat_tr_y_list]
#     normalized_tr_y_list = [[(t - mean) / std for t in tr_y] for tr_y, mean, std in
#                             zip(tr_y_list, mean_tr_y_list, std_tr_y_list)]
#
#     eval_x_list = [eval_x_list]
#     normalized_eval_x_list = [(eval_x - mean) / std for eval_x, mean, std in
#                               zip(eval_x_list, [mean_tr_x[0]], [std_tr_x[0]])]
#
#     normalized_eval_y_list = [(eval_y - mean) / std for eval_y, mean, std in
#                               zip(eval_y_list, mean_tr_y_list, std_tr_y_list)]
#     norm_tool = {'means_x': mean_tr_x, 'means_y': mean_tr_y_list, 'stds_x': std_tr_x, 'stds_y': std_tr_y_list}
#
#     return normalized_tr_x_list, normalized_tr_y_list, normalized_eval_x_list[0], normalized_eval_y_list, norm_tool
#
#
# def denormalize_tensors(pred_y, norm_tool):
#     # denormalized_lists = []
#     for i, normalized_list in enumerate(normalized_lists):
#         mean = norm_tool['means'][i]
#         std = norm_tool['stds'][i]
#         denormalized_list = [(t * std) + mean for t in normalized_list]
#         denormalized_lists.append(denormalized_list)
#     mean = norm_tool['means_y'][-1]
#     std = norm_tool['stds_y'][-1]
#     denormalized_pred_y = (pred_y * std) + mean
#     return denormalized_pred_y


###################

def normalize_data(tr_x, eval_x, tr_y_list, eval_y_list):
    # normalize
    norm_tool = Dateset_normalize_manager(tr_x, tr_y_list)
    tr_x = [norm_tool.normalize_input(_, 0) for _ in tr_x]
    tr_y_list = norm_tool.normalize_outputs(tr_y_list)
    eval_x = norm_tool.normalize_input(eval_x, 0)
    eval_y_list = norm_tool.normalize_outputs(eval_y_list)

    return tr_x, eval_x, tr_y_list, eval_y_list, norm_tool


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


if __name__ == '__main__':
    support_model = list(model_dict.keys())
    if len(sys.argv) < 2:
        print('Usage: python mfgp_demo.py <model_name>')
        print('support model: {}'.format(support_model))
        exit()
    elif sys.argv[1] not in support_model:
        print('model_name must be one of {}'.format(support_model))
        print('Got {}'.format(sys.argv[1]))
        exit()

    model_name = sys.argv[1]

    tr_x, eval_x, tr_y_list, eval_y_list, sample_num, fidelity_num = get_testing_data_hpo()
    ################################
    normalized_x = F.softmax(tr_x, dim=1)
    entropy = -torch.sum(normalized_x * torch.log(normalized_x), dim=1)
    # pick top 10% of entropy
    num_rows_to_select = int(0.1 * sample_num)
    _, indices = torch.topk(entropy, num_rows_to_select)
    selected_rows = tr_x[indices]
    selected_x = [tr_x]
    selected_y = [tr_y_list[0]]
    full_selected_set = set(indices.numpy())

    num_fidelity_indicators = len(selected_y)

    # rest_of_y = selected_y[1:]
    # random.shuffle(rest_of_y)
    # selected_y = [selected_y[0]] + rest_of_y
    initial_data = [
        {
            'fidelity_indicator': k,
            'raw_fidelity_name': str(k),
            'X': selected_x[k],  # Assign the same 'X' value for each item
            'Y': selected_y[k]  # Assign 'Y' from the k-th item of selected_y
        }
        for k in range(num_fidelity_indicators)  # Iterate over the range of fidelity indicators
    ]

    ##############################
    groundtruth = eval_y_list[-1]
    evaluation_num = len(groundtruth)
    print(len(tr_y_list))

    fidelity_num = 1 if model_name in ['CIGP', 'HOGP'] else fidelity_num

    src_y_shape = tr_y_list[0].shape[1:]
    if model_name in ['AR', 'CIGAR', 'CAR', 'NAR', 'ResGP', 'CIGP']:
        flatten_output = True
        sample_num = tr_y_list[0].shape[0]
        tr_y_list = [_.reshape(_.shape[0], -1) for _ in tr_y_list]
        eval_y_list = [_.reshape(_.shape[0], -1) for _ in eval_y_list]

    # normalize data
    tr_x, full_x, tr_y_list, full_y_list, norm_tool = normalize_data(tr_x, eval_x, tr_y_list, eval_y_list)
    list_eval_index = [set(range(0, sample_num))] + [full_selected_set] * (fidelity_num - 1)
    # module for stopping trails:
    stopping_trail = [1, 3, 9, 27, 81]

    for e in range(len(stopping_trail)):
        if stopping_trail[e] > fidelity_num:
            break

        select_top_k = int(sample_num / pow(3, (e+1)))
        current_trail = stopping_trail[e]
        before_next_trail = min(stopping_trail[e + 1], fidelity_num)
        print("current stopping trial:", current_trail)
        tr_x = []
        tr_y = []
        for i in range(len(list_eval_index)):
            list_i = list(list_eval_index[i])
            tr_x.append(full_x[list_i])
            tr_y.append(full_y_list[i][list_i])

        # init model
        config = {
            'fidelity_shapes': [_y.shape[1:] for _y in tr_y_list],
        }
        model_define = model_dict[model_name]
        model = model_define(config)

        # print info
        print('model: {}'.format(model_name))
        print('fidelity num: {}'.format(fidelity_num))
        # print('x shape: {}'.format(tr_x.shape))
        # print('y shape: {}'.format([_.shape for _ in tr_y_list]))

        # enable to test cuda
        if True and torch.cuda.is_available():
            print('enable cuda')
            model = model.cuda()
            tr_x = tr_x.cuda()
            full_x = full_x.cuda()
            tr_y_list = [_.cuda() for _ in tr_y_list]
            full_y_list = [_.cuda() for _ in full_y_list]

        # training
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        max_epoch = 300 if model_name in ['CIGAR', 'GAR'] else 150

        train_each_fidelity_separately = True
        if train_each_fidelity_separately and model_name not in ['CIGP', 'HOGP']:
            '''
                Train method 1: train each fidelity separately
            '''
            for _fn in range(fidelity_num):
                for epoch in range(max_epoch):
                    optimizer.zero_grad()
                    if _fn == 0:
                        low_fidelity = None
                        high_fidelity = tr_y[_fn]
                    elif tr_y_list[_fn - 1].shape[0] != tr_y[_fn].shape[0]:
                        matching_indices = find_matching_indices(tr_x[_fn - 1], tr_x[_fn])
                        low_fidelity = tr_y[_fn - 1][matching_indices]
                        high_fidelity = tr_y[_fn]

                    else:
                        low_fidelity = tr_y[_fn - 1]
                        high_fidelity = tr_y[_fn]
                    nll = model.single_fidelity_compute_loss(tr_x[_fn], low_fidelity, high_fidelity, fidelity_index=_fn)
                    print('fidelity {}, epoch {}/{}, nll: {}'.format(_fn, epoch + 1, max_epoch, nll.item()), end='\r')
                    nll.backward()
                    optimizer.step()
                print('\n')
        else:
            '''
                Train method 2: train all fidelity at the same time
            '''
            for epoch in range(max_epoch):
                optimizer.zero_grad()
                nll = model.compute_loss(tr_x, tr_y_list)
                nll.backward()
                optimizer.step()
                print('epoch {}/{}, nll: {}'.format(epoch + 1, max_epoch, nll.item()), end='\r')

        # predict and plot result
        with torch.no_grad():
            if model_name in ['CAR']:
                # predict_y = model(eval_x, fidelity_num-1)[0]
                predict_y = model.forward(x=tr_x[current_trail-1], to_fidelity_n=-1)[0]
            else:
                predict_y = model(tr_x[current_trail-1])[0]

        from MFGP.utils.plot_field import plot_container

        # groundtruth = norm_tool.denormalize_output(eval_y_list[-1], len(tr_y_list)-1)
        predict_y = norm_tool.denormalize_output(predict_y, len(tr_y_list) - 1)
        # plot_container([groudtruth.reshape(-1, *src_y_shape), predict_y.reshape(-1, *src_y_shape)],
        #                ['ground truth', 'predict'], 0).plot(3)
        print("model_name:", model_name)
        matching_groundtruth_indices = find_matching_indices(full_x, tr_x[current_trail-1])
        evaluation_num = len(groundtruth)
        sorted_predict_y, sorted_predict_y_indices = torch.sort(predict_y, dim=0)
        sorted_groundtruth, sorted_groundtruth_indices = torch.sort(groundtruth[matching_groundtruth_indices], dim=0)

        # for k in [5, 10, 20]:
        #     top_k = int((k / 100) * evaluation_num)
        #     top_k_predict_indices = sorted_predict_y_indices[:top_k]
        #     top_k_groundtruth_indices = sorted_groundtruth_indices[:top_k]
        #
        #     # # Compute the intersection of indices
        #     # _, indices_in_top_k_predict = torch.where(
        #     #     torch.isin(top_k_groundtruth_indices, top_k_predict_indices)
        #     # )
        #     similarity = sum(1 for i in range(top_k) if top_k_predict_indices[i] == top_k_groundtruth_indices[i])
        #
        #     # similarity = indices_in_top_k_predict.numel()
        #     print(f"Strict Precision @{k}%: {similarity / top_k}")
        #
        # # Calculate overlap precision at 5%, 10%, 20% of all evaluations, evaluation number should be calculated first
        #
        # for k in [5, 10, 20]:
        #     top_k = int((k / 100) * evaluation_num)
        #     similarity = precision_at_k(predicted_ranking=sorted_predict_y_indices,
        #                                 ground_truth=sorted_groundtruth_indices,
        #                                 k=top_k)
        #     print(f"Overlap Precision @{k}%: {similarity}")
        #
        # mse = torch.mean((groundtruth[matching_groundtruth_indices] - predict_y) ** 2)
        # print("MSE:", mse)

        from scipy.stats import spearmanr
        from scipy.stats import kendalltau

        # Assuming groundtruth and predict_y are your rank vectors
        spearman_correlation, _ = spearmanr(groundtruth[matching_groundtruth_indices], predict_y)
        print("Spearman correlation:", spearman_correlation)

        # Assuming groundtruth and predict_y are your rank vectors
        kendall_tau, _ = kendalltau(groundtruth[matching_groundtruth_indices], predict_y)

        print(f"Kendall's Tau coefficient: {kendall_tau}")

        # return the top_k hyperparameters index, prepare for next round of evaluation
        top_k_hyperparameters_index = find_matching_indices(groundtruth, sorted_predict_y[:select_top_k,:])
        top_k_x = tr_x[current_trail-1][sorted_predict_y_indices[:select_top_k,:]]
        top_k_x_indices = find_matching_indices(full_x, top_k_x)
        # add these indices to set in list_eval_index, until current_trial in the list_eval_index
        for i in range(current_trail, before_next_trail):
            list_eval_index[i] = list_eval_index[i].union(set(top_k_x_indices))
