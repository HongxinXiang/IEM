import os

import numpy as np
import pandas as pd
import torch


def load_dual_data_files(dataroot, dataset, img_folder_name):
    processed_root = f"{dataroot}/{dataset}/processed"
    img_folder = f"{processed_root}/{img_folder_name}"
    graph_path = f"{processed_root}/geometric_data_processed.pt"
    align_path = f"{processed_root}/{dataset}_processed_ac.csv"
    return img_folder, graph_path, align_path


def check_processed_dataset(dataroot, dataset, img_folder_name):
    processed_path = os.path.join(dataroot, dataset, "processed")

    processed_file_path = os.path.join(processed_path, "{}_processed_ac.csv".format(dataset))
    graph_file = os.path.join(processed_path, "geometric_data_processed.pt")
    image_folder = os.path.join(processed_path, img_folder_name)

    if not (os.path.exists(processed_file_path) and os.path.exists(graph_file) and os.path.exists(image_folder)):
        return False

    # check file processed_ac.csv
    df = pd.read_csv(processed_file_path)
    cols = df.columns
    if not ("smiles" in cols and "index" in cols and "label" in cols):
        return False
    index = df["index"].values.astype(str).tolist()

    # check geometric_data_processed.pt and image folder
    graph_data = torch.load(graph_file)
    graph_index = [str(item) for item in graph_data[0].index]
    image_index = [str(os.path.splitext(item)[0]) for item in os.listdir(image_folder)]

    if not len(index) == len(graph_index) == len(image_index):
        return False
    if len(set(index) - set(graph_index)) != 0 or len(set(index) - set(image_index)) != 0:
        return False
    return True


def get_label_from_align_data(label_series, task_type="classification"):
    '''e.g. get_label_from_align_data(df["label"])'''
    if task_type == "classification":
        return np.array(label_series.apply(lambda x: np.array(str(x).split(" ")).astype(int).tolist()).tolist())
    elif task_type == "regression":
        return np.array(label_series.apply(lambda x: np.array(str(x).split(" ")).astype(float).tolist()).tolist())
    else:
        raise UserWarning("{} is undefined.".format(task_type))


def load_dual_align_data(dataroot, dataset, img_folder_name, task_type="classification", graph_feat="all", verbose=False, logger=None):
    log = print if logger is None else logger.info
    if not check_processed_dataset(dataroot, dataset, img_folder_name):
        raise ValueError("{}/{}: data is not completed.".format(dataroot, dataset))
    if verbose:
        log("checking processed dataset completed. ")

    img_folder, graph_path, align_path = load_dual_data_files(dataroot, dataset, img_folder_name)

    df = pd.read_csv(align_path)
    index = df["index"].astype(str).tolist()
    smiles = df["smiles"].astype(str).to_numpy()
    if verbose:
        log("reading align data completed. ")

    # load graph data
    graph_data, graph_slices = torch.load(graph_path)
    if graph_feat == "min":  # 和 pretrain-gnns 配置一样
        graph_data.edge_attr = graph_data.edge_attr[:, :2]
        graph_data.x = graph_data.x[:, :2]

    graph_index = np.array(graph_data.index).astype(str).tolist()
    assert (np.array(graph_index) == np.array(index)).sum() == len(index), "index from graph  and index from csv file is inconsistent"

    new_index = df["index"].astype(str).values
    label = get_label_from_align_data(df["label"], task_type=task_type)

    image_path_list = []
    for idx in new_index:
        image_path = []
        for filename in os.listdir(f"{img_folder}/{idx}/"):
            image_path.append(f"{img_folder}/{idx}/{filename}")
        image_path_list.append(image_path)

    return {
        "index": new_index,
        "smiles": smiles,
        "label": label,
        "graph_data": graph_data,
        "graph_slices": graph_slices,
        "image_path": image_path_list
    }

