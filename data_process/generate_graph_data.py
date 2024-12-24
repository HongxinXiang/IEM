import os
from argparse import ArgumentParser
from shutil import copyfile

import pandas as pd
import torch
from rdkit.Chem import MolToSmiles, MolFromSmiles
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

from utils.mol2graph import create_graph_from_smiles


if __name__ == '__main__':
    # Parse arguments
    parser = ArgumentParser(description='PyTorch Implementation of GraphGenerator')
    parser.add_argument('--dataroot', type=str, default="datasets/reg", help='path to exp folder')
    parser.add_argument('--datasets', type=str, default="freesolv", help='path to exp folder')
    parser.add_argument('--use_processed_csv', action='store_true', default=False, help='')
    args = parser.parse_args()

    datasets = args.datasets.split(",")
    for dataset in datasets:
        if args.use_processed_csv:
            csv_path = f"{args.dataroot}/{dataset}/processed/{dataset}_processed_ac.csv"
        else:
            csv_path = f"{args.dataroot}/{dataset}/raw/{dataset}.csv"

        df = pd.read_csv(csv_path)

        processed_saveroot = f"{args.dataroot}/{dataset}/processed"
        if not os.path.exists(processed_saveroot):
            os.makedirs(processed_saveroot)

        graph_list = []
        for index, smiles, label in tqdm(zip(df["index"].tolist(), df.smiles.tolist(), df["label"].tolist()), total=len(df["index"].tolist())):
            if isinstance(label, str):
                label = [float(item) for item in label.split(" ")]

            mol = MolFromSmiles(smiles)
            assert mol is not None
            canonical_smiles = MolToSmiles(mol, canonical=True)
            # generate graph
            graph = create_graph_from_smiles(smiles, label, index, pre_filter=None, pre_transform=None,
                                             task_type="regression", graph_feat_extractor="default")
            if graph is None:
                raise Exception
            graph_list.append(graph)
        data, slices = InMemoryDataset.collate(graph_list)
        torch.save((data, slices), f"{processed_saveroot}/geometric_data_processed.pt")

        df["smiles"].to_csv(f"{processed_saveroot}/smiles.csv", index=False, header=None)

        ################ using simple features
        simple_processed_saveroot = f"{args.dataroot}/{dataset}_simple/processed"
        if not os.path.exists(simple_processed_saveroot):
            os.makedirs(simple_processed_saveroot)

        data.x = data.x[:, :2]
        data.edge_attr = data.edge_attr[:, :2]

        torch.save((data, slices), f"{simple_processed_saveroot}/geometric_data_processed.pt")
        df["smiles"].to_csv(f"{simple_processed_saveroot}/smiles.csv", index=False, header=None)
        copyfile(csv_path, f"{simple_processed_saveroot}/{dataset}_processed_ac.csv")

