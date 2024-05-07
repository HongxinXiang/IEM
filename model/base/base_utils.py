from collections import OrderedDict

from torch import nn


def get_predictor(arch, in_features, num_tasks, inner_dim=None, dropout=0.2, activation_fn=None):
    if inner_dim is None:
        inner_dim = in_features // 2
    if activation_fn is None:
        activation_fn = "gelu"
    if arch == "arch1":
        return nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(in_features, inner_dim)),
            ("Softplus", nn.Softplus()),
            ("linear2", nn.Linear(inner_dim, num_tasks))
        ]))
    elif arch == "arch2":
        return nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(in_features, 128)),
            ('leakyreLU', nn.LeakyReLU()),
            ('dropout', nn.Dropout(dropout)),
            ('linear2', nn.Linear(128, num_tasks))
        ]))
    elif arch == "arch3":
        return nn.Sequential(OrderedDict([
            ('linear', nn.Linear(in_features, num_tasks))
        ]))
    elif arch == "none":
        return nn.Identity()

