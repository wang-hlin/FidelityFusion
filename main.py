import os

import torch
import numpy as np
# import pandas as pd
# import torch.nn.functional as F
from utils import arg_parse, load_data, model_dict, normalize_data, dump_list
from config.config import get_cfg_defaults
from sklearn.model_selection import train_test_split
import pickle
# import random

# from MFGP import *
# from MFGP.utils.normalizer import Dateset_normalize_manager


def main():
    args = arg_parse()
    # ---- setup configs ----
    cfg = get_cfg_defaults()
    support_model = list(model_dict.keys())
    if cfg.MODEL.NAME not in support_model:
        print('model_name must be one of {}'.format(support_model))
        print('Got {}'.format(cfg.MODEL.NAME))
        exit()

    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    print(cfg)

    data_root = cfg.DATASET.ROOT
    benchmark = cfg.DATASET.BENCHMARK
    task = cfg.DATASET.TASK
    model_name = cfg.MODEL.NAME
    base = cfg.SOLVER.BASE
    seed = cfg.SOLVER.SEED
    x, y = load_data(data_root, benchmark, task)
    n_samples = y.shape[0]
    n_fidelity = y.shape[1]
    out_dir = cfg.OUTPUT.ROOT

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    ################################
    # pick top 10% of entropy, changed to random selection
    # normalized_x = F.softmax(x_norm_list, dim=1)
    # entropy = -torch.sum(normalized_x * torch.log(normalized_x), dim=1)
    #
    # num_rows_to_select = int(cfg.DATASET.TRAIN_RATIO * n_samples)
    # _, indices = torch.topk(entropy, num_rows_to_select)
    train_ratio = cfg.DATASET.TRAIN_RATIO
    train_ratio_str = str(train_ratio).replace('.', '_')
    train_idx, valid_idx = train_test_split(range(n_samples), test_size=1 - train_ratio,
                                            train_size=train_ratio, random_state=seed)
    train_size = len(train_idx)

    out_file_name_prefix = f"{benchmark}_{task}_{model_name}_{train_ratio_str}_{seed}"
    ##############################
    y_high_fid = y[:, -1]

    fidelity_num = 1 if model_name in ['CIGP', 'HOGP'] else n_fidelity

    # if model_name in ['AR', 'CIGAR', 'CAR', 'NAR', 'ResGP', 'CIGP']:
    #     flatten_output = True
    y_list = [y[:, _i].reshape(n_samples, -1) for _i in range(n_fidelity)]

    # normalize data
    x_normalized, y_normalized_list, norm_tool = normalize_data(x, y_list)
    x_normalized = x_normalized.float()
    list_eval_index = [np.concatenate([np.array(train_idx), np.array(valid_idx)])]
    for i in range(1, fidelity_num):
        list_eval_index.append(np.array(train_idx))
    # module for stopping trails:
    stopping_trail = cfg.SOLVER.STOPPING_TRAILS
    y_pred_all = []
    for stop_epoch_idx in range(len(stopping_trail)):
        if stopping_trail[stop_epoch_idx] > fidelity_num:
            break

        select_top_k = int(n_samples / pow(base, (stop_epoch_idx + 1)))
        current_epoch = stopping_trail[stop_epoch_idx]
        if stop_epoch_idx + 1 < len(stopping_trail):
            next_stop_epoch = stopping_trail[stop_epoch_idx + 1]
        else:
            next_stop_epoch = current_epoch
        before_next_trail = min(next_stop_epoch, fidelity_num)
        print("Current stopping trial (epoch):", current_epoch)

        out_model_path = os.path.join(out_dir,
                                      f"{out_file_name_prefix}_epoch_{current_epoch}.pth")
        if os.path.exists(out_model_path):
            model = torch.load(out_model_path)

        else:
            # init model
            config = {
                'fidelity_shapes': [_y.shape[1:] for _y in y_normalized_list],
            }
            model_define = model_dict[model_name]
            model = model_define(config)

            # print info
            print('model: {}'.format(model_name))
            print('fidelity num: {}'.format(fidelity_num))

            # enable to test cuda
            # if torch.cuda.is_available():
            #     print('enable cuda')
            #     model = model.cuda()
            #     x_normalized_list = [_.cuda() for _ in x_normalized_list]
            #     x_normalized = x_normalized.cuda()
            #     y_normalized_list = [_.cuda() for _ in y_normalized_list]
            #     full_y_list = [_.cuda() for _ in full_y_list]

            # ---------------------------------------- training ----------------------------------------
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            max_epoch = 300 if model_name in ['CIGAR', 'GAR'] else 150

            if cfg.SOLVER.TRAIN_FIDELITY_SEPARATELY and model_name not in ['CIGP', 'HOGP']:
                '''
                Train method 1: train each fidelity separately
                '''
                for _fn in range(fidelity_num):
                    x_normalized_fn = x_normalized[list_eval_index[_fn]].float()
                    for epoch in range(max_epoch):
                        optimizer.zero_grad()
                        if _fn == 0:
                            low_fidelity = None
                            high_fidelity = y_normalized_list[_fn]
                        # elif list_eval_index[_fn - 1].shape[0] != list_eval_index[_fn].shape[0]:
                        #
                        #     low_fidelity = y_normalized_list[_fn - 1][list_eval_index[_fn]]
                        #     high_fidelity = y_normalized_list[_fn][list_eval_index[_fn]]
                        else:
                            low_fidelity = y_normalized_list[_fn - 1][list_eval_index[_fn]]
                            high_fidelity = y_normalized_list[_fn][list_eval_index[_fn]]
                        if low_fidelity is not None:
                            low_fidelity = low_fidelity.float()
                        high_fidelity = high_fidelity.float()
                        nll = model.single_fidelity_compute_loss(x_normalized_fn,
                                                                 low_fidelity,
                                                                 high_fidelity,
                                                                 fidelity_index=_fn)
                        print('fidelity {}, epoch {}/{}, nll: {}\r'.format(_fn, epoch+1, max_epoch, nll.item()))
                        nll.backward()
                        optimizer.step()
                    print('\n')
            else:
                '''
                Train method 2: train all fidelity at the same time
                '''
                for epoch in range(max_epoch):
                    x_normalized_fn = x_normalized[list_eval_index[epoch], :].float()
                    optimizer.zero_grad()
                    nll = model.compute_loss(x_normalized_fn, y_normalized_list)
                    nll.backward()
                    optimizer.step()
                    print('epoch {}/{}, nll: {}'.format(epoch + 1, max_epoch, nll.item()), end='\r')

            # save model
            torch.save(model, out_model_path)

        # predict and plot result
        x_current_epoch = x_normalized[list_eval_index[current_epoch - 1]].float()
        with torch.no_grad():
            if model_name in ['CAR']:
                # y_pred = model(eval_x, fidelity_num-1)[0]
                y_pred = model.forward(x=x_current_epoch[train_size:], to_fidelity_n=-1)[0]
            else:
                y_pred = model(x_current_epoch[train_size:])[0]

        # from MFGP.utils.plot_field import plot_container

        # y_high_fid = norm_tool.denormalize_output(eval_y_list[-1], len(y_normalized_list)-1)
        y_pred = norm_tool.denormalize_output(y_pred, len(y_normalized_list) - 1)
        y_pred_all.append(y_pred)
        # plot_container([y_high_fid.reshape(-1, *src_y_shape), y_pred.reshape(-1, *src_y_shape)],
        #                ['ground truth', 'predict'], 0).plot(3)
        print("model_name:", model_name)
        valid_idx_iter = list_eval_index[current_epoch - 1][train_size:]

        if benchmark == "lcbench":
            y_pred_sort, y_pred_sort_idx = torch.sort(y_pred, dim=0, descending=True)
        else:
            y_pred_sort, y_pred_sort_idx = torch.sort(y_pred, dim=0, descending=False)
        # sorted_ground_truth, sorted_ground_truth_indices = torch.sort(y_high_fid[matching_gt_indices], dim=0)

        # for k in [5, 10, 20]:
        #     top_k = int((k / 100) * n_configs)
        #     top_k_predict_indices = y_pred_sort_idx[:top_k]
        #     top_k_ground_truth_indices = sorted_ground_truth_indices[:top_k]
        #
        #     # # Compute the intersection of indices
        #     # _, indices_in_top_k_predict = torch.where(
        #     #     torch.isin(top_k_ground_truth_indices, top_k_predict_indices)
        #     # )
        #     similarity = sum(1 for i in range(top_k) if top_k_predict_indices[i] == top_k_ground_truth_indices[i])
        #
        #     # similarity = indices_in_top_k_predict.numel()
        #     print(f"Strict Precision @{k}%: {similarity / top_k}")
        #
        # # Calculate overlap precision at 5%, 10%, 20% of all evaluations, evaluation number should be calculated first
        #
        # for k in [5, 10, 20]:
        #     top_k = int((k / 100) * n_configs)
        #     similarity = precision_at_k(predicted_ranking=y_pred_sort_idx,
        #                                 y_high_fid=sorted_ground_truth_indices,
        #                                 k=top_k)
        #     print(f"Overlap Precision @{k}%: {similarity}")
        #
        # mse = torch.mean((y_high_fid[matching_gt_indices] - y_pred) ** 2)
        # print("MSE:", mse)

        from scipy.stats import spearmanr
        from scipy.stats import kendalltau

        # Assuming y_high_fid and y_pred are your rank vectors
        spearman_correlation, _ = spearmanr(y_high_fid[valid_idx_iter], y_pred)
        print("Spearman correlation:", spearman_correlation)

        # Assuming y_high_fid and y_pred are your rank vectors
        kendall_tau, _ = kendalltau(y_high_fid[valid_idx_iter], y_pred)

        print(f"Kendall's Tau coefficient: {kendall_tau}")

        eval_index_new = list_eval_index[current_epoch-1][y_pred_sort_idx[:select_top_k]].reshape(-1)
        for i in range(current_epoch, before_next_trail):
            list_eval_index[i] = np.concatenate((list_eval_index[i], eval_index_new))

    # save the final indices of selected hyperparameters
    out_file_name = os.path.join(out_dir, f"{out_file_name_prefix}_selected_indices.pkl")
    dump_list(list_eval_index, out_file_name)
    print(f"Selected indices are saved to {out_file_name}")

    # save the final prediction
    out_file_name = os.path.join(out_dir, f"{out_file_name_prefix}_final_prediction.pkl")
    dump_list(y_pred_all, out_file_name)
    print(f"Final prediction is saved to {out_file_name}")


if __name__ == '__main__':
    main()
