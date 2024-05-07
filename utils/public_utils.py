import os
import random
from argparse import Namespace

import numpy as np
import torch
import yaml


def create_dir_if_not_exisit(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def fix_train_random_seed(seed=2021):
    # fix random seeds
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def setup_device(n_gpu_use):
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(
            "Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(
                n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def is_left_better_right(left_num, right_num, standard):
    '''

    :param left_num:
    :param right_num:
    :param standard: if max, left_num > right_num is true, if min, left_num < right_num is true.
    :return:
    '''
    assert standard in ["max", "min"]
    if standard == "max":
        return left_num > right_num
    elif standard == "min":
        return left_num < right_num


def get_tqdm_desc(dataset, epoch):
    tqdm_train_desc = "[train] dataset: {}; epoch: {}".format(dataset, epoch)
    tqdm_eval_train_desc = "[eval on train set] dataset: {}; epoch: {}".format(dataset, epoch)
    tqdm_eval_val_desc = "[eval on valid set] dataset: {}; epoch: {}".format(dataset, epoch)
    tqdm_eval_test_desc = "[eval on test set] dataset: {}; epoch: {}".format(dataset, epoch)
    return tqdm_train_desc, tqdm_eval_train_desc, tqdm_eval_val_desc, tqdm_eval_test_desc
