import os
from collections import OrderedDict

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_classifier(arch, in_features, num_tasks):
    if arch == "arch1":
        return nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(in_features, in_features // 2)),
            ("Softplus", nn.Softplus()),
            ("linear2", nn.Linear(in_features // 2, num_tasks))
        ]))
    elif arch == "arch2":
        return nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(in_features, 128)),
            ('leakyreLU', nn.LeakyReLU()),
            ('dropout', nn.Dropout(0.2)),
            ('linear2', nn.Linear(128, num_tasks))
        ]))
    elif arch == "arch3":
        return nn.Sequential(OrderedDict([
            ('linear', nn.Linear(in_features, num_tasks))
        ]))


def save_checkpoint(model_dict, optimizer_dict, lr_scheduler_dict, desc, epoch, save_path, name_pre, name_post='_best', logger=None):
    log = print if logger is None else logger.info

    state = {
        'epoch': epoch,
        'desc': desc
    }

    if model_dict is not None:
        for key in model_dict.keys():
            model = model_dict[key]
            state[key] = {k: v.cpu() for k, v in model.state_dict().items()}
    if optimizer_dict is not None:
        for key in optimizer_dict.keys():
            optimizer = optimizer_dict[key]
            state[key] = optimizer.state_dict()
    if lr_scheduler_dict is not None:
        for key in lr_scheduler_dict.keys():
            lr_scheduler = lr_scheduler_dict[key]
            state[key] = lr_scheduler.state_dict()

    try:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            log("Directory {} is created.".format(save_path))
    except:
        pass

    filename = '{}/{}{}.pth'.format(save_path, name_pre, name_post)
    torch.save(state, filename)
    log('model has been saved as {}'.format(filename))


def load_checkpoint(pretrained_pth, image2d_teacher, image3d_teacher, propertiesPredictor, atomDistPredictor,
                    boundDistPredictor, geometryPredictor, optimizer=None, lr_scheduler=None, logger=None):
    log = logger.info if logger is not None else print
    flag = False
    resume_desc = None
    if os.path.isfile(pretrained_pth):
        pretrained_model = torch.load(pretrained_pth)
        resume_desc = pretrained_model["desc"]
        model_list = [("image2d_teacher", image2d_teacher), ("image3d_teacher", image3d_teacher),
                      ("propertiesPredictor", propertiesPredictor),
                      ("atomDistPredictor", atomDistPredictor), ("boundDistPredictor", boundDistPredictor),
                      ("geometryPredictor", geometryPredictor)]
        if optimizer is not None:
            model_list.append(("optimizer", optimizer, "optimizer"))
        if lr_scheduler is not None:
            model_list.append(("lr_scheduler", lr_scheduler, "lr_scheduler"))
        for model_key, model in model_list:
            try:
                model.load_state_dict(pretrained_model[model_key])
            except:
                ckp_keys = list(pretrained_model[model_key])
                cur_keys = list(model.state_dict())
                model_sd = model.state_dict()
                for ckp_key, cur_key in zip(ckp_keys, cur_keys):
                    model_sd[cur_key] = pretrained_model[model_key][ckp_key]
                model.load_state_dict(model_sd)
            log("[resume info] resume {} completed.".format(model_key))
        flag = True
    else:
        log("===> No checkpoint found at '{}'".format(pretrained_pth))

    return flag, resume_desc


def write_result_dict_to_tb(tb_writer: SummaryWriter, result_dict: dict, optimizer_dict: dict, show_epoch=True):
    loop = result_dict["epoch"] if show_epoch else result_dict["step"]
    for key in result_dict.keys():
        if key == "epoch" or key == "step":
            continue
        tb_writer.add_scalar(key, result_dict[key], loop)
    for key in optimizer_dict.keys():
        optimizer = optimizer_dict[key]
        tb_writer.add_scalar(key, optimizer.param_groups[0]["lr"], loop)


