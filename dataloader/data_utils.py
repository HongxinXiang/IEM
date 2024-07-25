import os.path
import pickle

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.transforms_factory import create_transform
from torchvision import transforms
from tqdm import tqdm


def transforms_for_train_aug(resize=224, mean_std=None, p=0.2):
    if mean_std is None:
        mean, std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    else:
        mean, std = mean_std[0], mean_std[1]
    img_transforms = transforms.Compose([transforms.CenterCrop(resize), transforms.RandomHorizontalFlip(),
                                         transforms.RandomGrayscale(p), transforms.RandomRotation(degrees=360),
                                         transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return img_transforms


def transforms_for_train(config, is_training=True):
    return create_transform(**config, is_training=is_training)


def transforms_for_eval(resize=(224, 224), img_size=(224, 224), mean_std=None):
    if mean_std is None:
        mean, std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    else:
        mean, std = mean_std[0], mean_std[1]
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def check_num_of_image3d(image3d_path_list, n_view):
    for idx, view_path_list in enumerate(image3d_path_list):
        assert len(view_path_list) == n_view, \
            "The view number of image3d {} is {}, not equal to the expected {}" \
                .format(idx, len(view_path_list), n_view)


def load_image3d_data_list(dataroot, dataset, image3d_type="processed", label_column_name="label",
                           image3d_dir_name="image3d", csv_suffix="", is_cache=False, logger=None):
    log = print if logger is None else logger.info
    if is_cache:
        cache_path = f"{dataroot}/{dataset}/{image3d_type}/cache_{dataset}_load_image3d_data_list.pkl"
        if os.path.exists(cache_path):
            log(f"load from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            return data["image3d_index_list"], data["image3d_path_list"], data["image3d_label_list"]

    csv_file_path = os.path.join(dataroot, dataset, image3d_type, "{}{}.csv".format(dataset, csv_suffix))
    df = pd.read_csv(csv_file_path)
    columns = df.columns.tolist()
    assert "index" in columns and label_column_name in columns
    image3d_root = f"{dataroot}/{dataset}/{image3d_type}/{image3d_dir_name}"

    image3d_index_list = df["index"].tolist()
    image3d_label_list = df[label_column_name].apply(lambda x: str(x).split(' ')).tolist()
    image3d_path_list = []

    for image3d_index in tqdm(image3d_index_list, desc="load_image3d_data_list"):
        image3d_path = []
        for filename in os.listdir(f"{image3d_root}/{image3d_index}"):
            image3d_path.append(f"{image3d_root}/{image3d_index}/{filename}")
        image3d_path_list.append(image3d_path)

    if is_cache:
        log(f"save to cache: {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump({
                "image3d_index_list": image3d_index_list,
                "image3d_path_list": image3d_path_list,
                "image3d_label_list": image3d_label_list},
                f)
    return image3d_index_list, image3d_path_list, image3d_label_list


def read_image(image_path, img_type="RGB"):
    """从 image_path 从读取图片
    如果 img_type="RGB"，则直接读取；
    如果 img_type="BGR"，则将 BGR 转换为 RGB
    """
    if img_type == "RGB":
        return Image.open(image_path).convert('RGB')
    elif img_type == "BGR":
        img = Image.open(image_path).convert('RGB')
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        return img
    else:
        raise NotImplementedError