import collections.abc as container_abcs
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Data, Batch
from torch_geometric.data import InMemoryDataset
from torchvision import transforms

from dataloader.data_utils import load_image3d_data_list, check_num_of_image3d
from dataloader.finetune_data_utils import load_dual_align_data
from utils.splitter import *

string_classes, int_classes = str, int


class DualCollater(object):
    def __init__(self, follow_batch, multigpu=False):
        self.follow_batch = follow_batch
        self.multigpu = multigpu

    def collate(self, batch):
        elem = batch[0]
        if isinstance(elem, Data):
            if self.multigpu:
                return batch
            else:
                return Batch.from_data_list(batch, self.follow_batch)
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, np.ndarray):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int_classes):
            return torch.tensor(batch)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, container_abcs.Mapping):
            return {key: self.collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self.collate(s) for s in zip(*batch)))
        elif isinstance(elem, container_abcs.Sequence):
            return [self.collate(s) for s in zip(*batch)]

        raise TypeError('DataLoader found invalid type: {}'.format(type(elem)))

    def __call__(self, batch):
        return self.collate(batch)


def get_supervised_singal_name():
    return {
        "basic_properties_info": ['molecular_weight', 'MolLogP', 'MolMR', 'BalabanJ', 'NumHAcceptors', 'NumHDonors',
                                  'NumValenceElectrons', 'TPSA'],
        "atom_count_dict": ['C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'P', 'Si', 'B', 'Se', 'Ge', 'As', 'H', 'Ti', 'Ga',
                            'Ca', 'Mg', 'Zn'],
        "bound_count_dict": ['SINGLE', 'AROMATIC', 'DOUBLE', 'TRIPLE'],  # {'SINGLE': 12, 'DOUBLE': 5}
        "normalized_points_flatten": []
    }


def get_supervised_singal_dim():
    return {
        "basic_properties_info": 8,
        "atom_count_dict": 19,
        "bound_count_dict": 4,
        "normalized_points_flatten": 60
    }


class PretrainDataset(Dataset):
    def __init__(self, dataroot, dataset, split="all", transforms=None, ret_index=False, args=None,
                 idx_image3d_list=[0, 1, 2, 3], logger=None):
        self.logger = logger
        self.log = print if logger is None else logger.info
        self.split = split
        assert split in ["all", "train", "valid"]
        # load data
        n_view = len(idx_image3d_list)
        index_list, image3d_path_list, image_path_list, info_path_list, label_list = self.load_data(dataroot, dataset, idx_image3d_list)

        self.args = args
        self.indexs = index_list
        self.image3d_path_list = image3d_path_list
        self.image_path_list = image_path_list
        self.info_path_list = info_path_list
        self.label_list = label_list
        self.max_points = 60

        self.total_image3d = len(self.image3d_path_list)
        self.total_view = len(self.image3d_path_list) * n_view
        self.transforms = transforms
        self.n_view = n_view
        self.ret_index = ret_index

    def load_data(self, dataroot, dataset, idx_image3d_list):
        if not os.path.exists("cache"):
            os.makedirs("cache")
        suffix = "_".join([str(item) for item in idx_image3d_list])
        cache_data_path = f"./cache/PretrainDataset@{dataset}@{self.split}@{suffix}.npz"
        if os.path.exists(cache_data_path):
            self.log(f"load cache from {cache_data_path}")
            data = np.load(cache_data_path, allow_pickle=True)
            return data["index_list"], data["image3d_path_list"], data["image_path_list"], data["info_path_list"], data["label_list"]
        else:
            # load multi-view data
            index_list, image3d_path_list, label_list = load_image3d_data_list(dataroot, dataset, label_column_name="label", is_cache=True, logger=self.logger)
            n_total = len(index_list)
            if self.split == "train":
                index_list, image3d_path_list, label_list = index_list[:int(n_total * 0.95)], image3d_path_list[:int(
                    n_total * 0.95)], label_list[:int(n_total * 0.95)]
            elif self.split == "valid":
                index_list, image3d_path_list, label_list = index_list[int(n_total * 0.95):], image3d_path_list[int(
                    n_total * 0.95):], label_list[int(n_total * 0.95):]
            tmp_image3d_path_list = []
            for tmp_list in tqdm(image3d_path_list, desc=f"check {idx_image3d_list}"):
                tmp_image3d_path_list.append(np.array(tmp_list)[idx_image3d_list].tolist())
            image3d_path_list = tmp_image3d_path_list
            n_view = len(idx_image3d_list)
            check_num_of_image3d(image3d_path_list, n_view)
            # load image generated by rdkit
            image_path_list = [f"{dataroot}/{dataset}/processed/image2d/{index}.png" for index in index_list]
            # load basic info
            info_path_list = [f"{dataroot}/{dataset}/processed/mol-basic-info/{index}.npy" for index in index_list]

            # align these data
            is_exists_on_image_and_info_path_list = [os.path.exists(image_path) and os.path.exists(info_path) for
                                                     image_path, info_path in
                                                     tqdm(zip(image_path_list, info_path_list), desc="align data",
                                                          total=len(image_path_list))]
            if np.sum(is_exists_on_image_and_info_path_list) != len(is_exists_on_image_and_info_path_list):
                index_list = np.array(index_list)[is_exists_on_image_and_info_path_list]
                image3d_path_list = np.array(image3d_path_list)[is_exists_on_image_and_info_path_list]
                image_path_list = np.array(image_path_list)[is_exists_on_image_and_info_path_list]
                info_path_list = np.array(info_path_list)[is_exists_on_image_and_info_path_list]
                label_list = np.array(label_list)[is_exists_on_image_and_info_path_list]
            self.log(f"save cache to {cache_data_path}")
            np.savez(cache_data_path, index_list=index_list, image3d_path_list=image3d_path_list,
                     label_list=label_list, image_path_list=image_path_list, info_path_list=info_path_list)
            return index_list, image3d_path_list, image_path_list, info_path_list, label_list

    def get_image(self, index):
        filename = self.image_path_list[index]
        img = Image.open(filename).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img

    def get_image3d(self, index):
        view_path_list = self.image3d_path_list[index]
        image3d = [Image.open(view_path).convert('RGB') for view_path in view_path_list]
        if self.transforms is not None:
            image3d = list(map(lambda img: self.transforms(img).unsqueeze(0), image3d))
            image3d = torch.cat(image3d)
        return image3d

    def get_info(self, index):
        filename = self.info_path_list[index]
        npy_data = np.load(filename, allow_pickle=True).item()

        n_dim_1 = len(get_supervised_singal_name()["basic_properties_info"])
        n_dim_2 = len(get_supervised_singal_name()["atom_count_dict"])
        n_dim_3 = len(get_supervised_singal_name()["bound_count_dict"])
        label_properties_dist, label_atom_dist, label_bound_dist, label_points_dist, label_points_mask = np.zeros(
            n_dim_1, dtype=np.double), np.zeros(n_dim_2, dtype=np.double), np.zeros(n_dim_3, dtype=np.double), np.zeros(
            self.max_points, dtype=np.double), np.zeros(self.max_points, dtype=np.int64)

        label_points_dist[:len(npy_data["normalized_points_flatten"])] = npy_data["normalized_points_flatten"]
        label_points_mask[:len(npy_data["normalized_points_flatten"])] = 1

        for i, key in enumerate(npy_data["basic_properties_info"]):
            if key in npy_data["basic_properties_info"].keys():
                label_properties_dist[i] = npy_data["basic_properties_info"][key]

        for i, key in enumerate(npy_data["atom_count_dict"]):
            if key in npy_data["atom_count_dict"].keys():
                label_atom_dist[i] = npy_data["atom_count_dict"][key]

        for i, key in enumerate(npy_data["bound_count_dict"]):
            if key in npy_data["bound_count_dict"].keys():
                label_bound_dist[i] = npy_data["bound_count_dict"][key]

        return label_properties_dist, label_atom_dist, label_bound_dist, label_points_dist, label_points_mask

    def __getitem__(self, index):
        image = self.get_image(index)
        image3d = self.get_image3d(index)
        label_properties_dist, label_atom_dist, label_bound_dist, label_points_dist, label_points_mask = self.get_info(index)

        if self.ret_index:
            return image, image3d, label_properties_dist, label_atom_dist, label_bound_dist, label_points_dist, label_points_mask, self.indexs[index]
        else:
            return image, image3d, label_properties_dist, label_atom_dist, label_bound_dist, label_points_dist, label_points_mask

    def __len__(self):
        return self.total_image3d


class GraphDataset(InMemoryDataset):
    def __init__(self, data, slices, transform=None, pre_transform=None, pre_filter=None, task_type="classification"):
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        super().__init__(transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)
        self.data, self.slices = data, slices
        self.num_tasks = self.data.y.shape[1]
        self.total = len(self)
        self.task_type = task_type


class FeatDataset(Dataset):
    def __init__(self, feats, labels):
        self.feats = feats
        self.labels = labels
        self.total = len(self.feats)

    def __getitem__(self, index):
        return self.feats[index], self.labels[index]

    def __len__(self):
        return self.total


class ImageDataset(Dataset):
    def __init__(self, filenames, labels, transform):
        '''
        :param names: image path, e.g. ["./data/1.png", "./data/2.png", ..., "./data/n.png"]
        :param labels: labels, e.g. single label: [[1], [0], [2]]; multi-labels: [[0, 1, 0], ..., [1,1,0]]
        :param img_transformer:
        :param args:
        '''

        self.filenames = filenames
        self.labels = labels
        self.total = len(self.filenames)
        self.transform = transform

    def get_image(self, index):
        filenames = self.filenames[index]
        images = [self.transform(Image.open(filename).convert('RGB')) for filename in filenames]
        return torch.stack(images, dim=0)

    def __getitem__(self, index):
        return self.get_image(index), self.labels[index]

    def __len__(self):
        return self.total


class FinetuneAlignVisionGraphDatasetFactory():
    def __init__(self, dataroot, dataset, img_root="image", split="scaffold", img_teacher=None, use_teacher_feat=False,
                 task_type="classification", graph_feat="all", cache_feat_prefix="", ret_index=False, args=None,
                 logger=None, cache_index=False, device="cpu", cache_root="caches"):
        self.dataroot = dataroot
        self.dataset = dataset
        self.img_root = img_root
        self.logger = logger
        self.log = print if logger is None else logger.info
        self.data_dict = load_dual_align_data(dataroot, dataset, img_root, task_type=task_type, graph_feat=graph_feat, verbose=False)
        self.total = len(self.data_dict["index"])
        self.img_teacher = img_teacher
        self.use_teacher_feat = use_teacher_feat
        self.cache_feat_prefix = cache_feat_prefix
        self.device = device
        self.cache_root = cache_root
        self.split = split
        self.ret_index = ret_index
        assert split in ["scaffold"]

        if cache_index and not os.path.exists(f"{cache_root}/cache_index"):
            os.makedirs(f"{cache_root}/cache_index")
        cache_index_path = f"{cache_root}/cache_index/{dataset}_{img_root}_{split}.npz"

        if os.path.exists(cache_index_path):
            self.log(f"read split index from cache: {cache_index_path}")
            train_idx, val_idx, test_idx = split_train_val_test_idx_split_file(split_path=cache_index_path, sort=False)
        else:
            if self.split == "scaffold":
                train_idx, val_idx, test_idx = scaffold_split_train_val_test(list(range(0, self.total)),
                                                                             self.data_dict["smiles"], frac_train=0.8,
                                                                             frac_valid=0.1, frac_test=0.1)
            else:
                raise Exception(f"{args.split} is not support.")

        self.data_dict["split"] = {"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx}

        if cache_index:
            np.savez(cache_index_path, idx_train=train_idx, idx_val=val_idx, idx_test=test_idx)

        # load data
        index, smiles, image_dataset_dict, graph_dataset_dict, image_graph_dataset_dict, label = self.load_data()

        self.args = args
        self.indexs = index
        self.smiles = smiles
        self.image_dataset_dict = image_dataset_dict
        self.graph_dataset_dict = graph_dataset_dict
        self.image_graph_dataset_dict = image_graph_dataset_dict
        self.label = label

    def load_data(self):
        train_idx, val_idx, test_idx = self.data_dict["split"]["train_idx"], self.data_dict["split"]["val_idx"], self.data_dict["split"]["test_idx"]
        if isinstance(train_idx, np.ndarray):
            train_idx, val_idx, test_idx = train_idx.tolist(), val_idx.tolist(), test_idx.tolist()
        index, smiles, label, image_path = self.data_dict["index"], self.data_dict["smiles"], self.data_dict["label"], self.data_dict["image_path"]
        image_path = np.array(image_path)

        # load image dataset
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor(), normalize])
        if self.img_teacher is not None and self.use_teacher_feat:
            cache_root = f"{self.cache_root}/cache_{self.img_root}_teacher_feat/"
            if not os.path.exists(cache_root):
                os.makedirs(cache_root)
            cache_file = f"{cache_root}/{self.cache_feat_prefix}{self.dataset}.npy"
            if os.path.exists(cache_file):
                feats = np.load(cache_file)
            else:
                dataset = ImageDataset(filenames=image_path, labels=[-1] * len(image_path), transform=transform)
                dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
                # dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
                feats = []
                self.img_teacher.eval()
                for x, y in tqdm(dataloader, desc=f"[{self.dataset}] extract features from teacher"):
                    bs, v, c, h, w = x.shape
                    x = x.reshape(bs*v, c, h, w)
                    x = x.to(self.device)
                    feat = self.img_teacher(x).reshape(bs, v, -1).mean(1)
                    feats.append(feat.detach().cpu().numpy())
                feats = np.concatenate(feats, axis=0)
                np.save(cache_file, feats)
            image_train_dataset = FeatDataset(feats=feats[train_idx], labels=[-1] * len(feats[train_idx]))
            image_valid_dataset = FeatDataset(feats=feats[val_idx], labels=[-1] * len(feats[val_idx]))
            image_test_dataset = FeatDataset(feats=feats[test_idx], labels=[-1] * len(feats[test_idx]))
        else:
            image_train_dataset = ImageDataset(filenames=image_path[train_idx], labels=[-1] * len(image_path[train_idx]), transform=transform)
            image_valid_dataset = ImageDataset(filenames=image_path[val_idx], labels=[-1] * len(image_path[val_idx]), transform=transform)
            image_test_dataset = ImageDataset(filenames=image_path[test_idx], labels=[-1] * len(image_path[test_idx]), transform=transform)
        image_dataset_dict = {
            "train": image_train_dataset,
            "valid": image_valid_dataset,
            "test": image_test_dataset,
        }

        # load graph dataset
        graph_dataset = GraphDataset(data=self.data_dict['graph_data'], slices=self.data_dict['graph_slices'])
        graph_train_dataset, graph_valid_dataset, graph_test_dataset = graph_dataset[train_idx], graph_dataset[val_idx], graph_dataset[test_idx]
        graph_dataset_dict = {
            "train": graph_train_dataset,
            "valid": graph_valid_dataset,
            "test": graph_test_dataset
        }

        # combine_image_graph_dataset
        image_graph_dataset_dict = {
            "train": CombineImageGraphDataset(image_train_dataset, graph_train_dataset, labels=label[train_idx], ret_index=self.ret_index),
            "valid": CombineImageGraphDataset(image_valid_dataset, graph_valid_dataset, labels=label[val_idx], ret_index=self.ret_index),
            "test": CombineImageGraphDataset(image_test_dataset, graph_test_dataset, labels=label[test_idx], ret_index=self.ret_index)
        }

        return index, smiles, image_dataset_dict, graph_dataset_dict, image_graph_dataset_dict, label

    def __len__(self):
        return self.total


class CombineImageGraphDataset(Dataset):
    def __init__(self, image_dataset, graph_dataset, labels, max_points=60, ret_index=False):
        self.image_dataset = image_dataset
        self.graph_dataset = graph_dataset
        self.labels = labels
        self.max_points = max_points
        self.total = len(self.labels)
        self.ret_index = ret_index

    def __getitem__(self, index):
        label_points_mask = np.zeros(self.max_points, dtype=np.int64)
        label_points_mask[:self.graph_dataset[index].x.shape[0]] = 1
        if self.ret_index:
            return index, self.image_dataset[index][0], self.graph_dataset[index], self.labels[index], label_points_mask
        else:
            return self.image_dataset[index][0], self.graph_dataset[index], self.labels[index], label_points_mask

    def __len__(self):
        return self.total

