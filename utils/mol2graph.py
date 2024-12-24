import torch
from ogb.utils.mol import smiles2graph
from torch_geometric.data import Data


def mol_to_graph_data_obj_default(smiles_str):
    """
    using ogb to extract graph features

    :param smiles_str:
    :return:
    """
    graph_dict = smiles2graph(smiles_str)  # introduction of features: https://blog.csdn.net/qq_38862852/article/details/106312171
    edge_attr = torch.tensor(graph_dict["edge_feat"], dtype=torch.long)
    edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long)
    x = torch.tensor(graph_dict["node_feat"], dtype=torch.long)
    return x, edge_index, edge_attr


def create_graph_from_smiles(smiles, label, index=None, pre_filter=None, pre_transform=None, task_type="classification", graph_feat_extractor="default"):
    assert task_type in ["classification", "regression"]
    assert graph_feat_extractor in ["default"]

    try:
        if graph_feat_extractor == "default":
            x, edge_index, edge_attr = mol_to_graph_data_obj_default(smiles)
        else:
            raise Exception("graph_feat_extractor {} is undefined".format(graph_feat_extractor))
        if task_type == "classification":
            y = torch.tensor(label, dtype=torch.long).view(1, -1)
        else:
            y = torch.tensor(label, dtype=torch.float).view(1, -1)
        graph = Data(edge_attr=edge_attr, edge_index=edge_index, x=x, y=y, index=index)
        if pre_filter is not None and pre_filter(graph):
            return None
        if pre_transform is not None:
            graph = pre_transform(graph)
        return graph
    except:
        return None
