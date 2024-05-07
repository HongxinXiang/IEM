import glob
import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataloader.data_utils import transforms_for_train_aug, transforms_for_eval
from dataloader.dataset import PretrainDataset, get_supervised_singal_dim
from loss.losses import SupConLoss
from model.base.predictor import Predictor
from model.base.predictor import VisionPredictor
from model.model_utils import save_checkpoint, load_checkpoint
from pretrain.pretrain_teacher_utils import train_one_epoch, evaluate
from utils.public_utils import fix_train_random_seed


def setup_DDP_mp(init_method, local_rank, rank, world_size, backend="nccl", verbose=False):
    if sys.platform == "win32":
        backend = "gloo"
    # If the OS is Windows or macOS, use gloo instead of nccl
    dist.init_process_group(backend=backend, init_method=init_method, world_size=world_size, rank=rank)
    # set distributed device
    device = torch.device("cuda:{}".format(local_rank))
    if verbose:
        print("Using device: {}".format(device))
        print(f"local rank: {local_rank}, global rank: {rank}, world size: {world_size}")
    return device


def parse_args():
    # Parse arguments
    parser = ArgumentParser(description='PyTorch Implementation of IEM')

    # basic
    parser.add_argument('--dataroot', type=str, help='data root')
    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument('--workers', default=2, type=int, help='number of data loading workers (default: 2)')

    # ddp
    parser.add_argument("--nodes", default=1, type=int, help="number of nodes for distributed training")
    parser.add_argument("--ngpus_per_node", default=2, type=int, help="number of GPUs per node for distributed training")
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:12355", type=str, help="url used to set up distributed training")
    parser.add_argument("--node_rank", default=0, type=int, help="node rank for distributed training")

    # model params
    parser.add_argument('--model_name', type=str, default="resnet18", help='model name')

    # optimizer
    parser.add_argument("--warmup_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument('--weighted_loss', action='store_true', help='add regularization for multi-task loss')

    # train
    parser.add_argument('--runseed', type=int, default=2021, help='random seed to run model (default: 2021)')
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
    parser.add_argument('--temperature', type=float, default=0.1, help="temperature required by contrastive loss")
    parser.add_argument('--base_temperature', type=float, default=0.1, help="temperature required by contrastive loss")
    parser.add_argument("--resume", type=str, default=None, help='Resume training from a path of checkpoint')

    # log
    parser.add_argument('--log_dir', default='./logs/pretrain_teacher_model/', help='path to log')

    # Parse arguments
    return parser.parse_args()


def print_only_rank0(text, logger=None):
    log = print if logger is None else logger.info
    if dist.get_rank() == 0:
        log(text)


def is_rank0():
    return dist.get_rank() == 0


def main(local_rank, ngpus_per_node, args):

    args.local_rank = local_rank
    args.rank = args.node_rank * ngpus_per_node + local_rank

    device = setup_DDP_mp(init_method=args.dist_url, local_rank=args.local_rank, rank=args.rank,
                          world_size=args.world_size, verbose=True)

    # fix random seeds
    fix_train_random_seed(args.runseed)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    print_only_rank0("run command: " + " ".join(sys.argv))
    print_only_rank0("log_dir: {}".format(args.log_dir))

    ########################## load dataset
    # transforms
    train_transforms = transforms_for_train_aug(resize=args.imageSize, mean_std=(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    valid_transforms = transforms_for_eval(resize=(args.imageSize, args.imageSize), img_size=(args.imageSize, args.imageSize), mean_std=(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))

    # Load dataset
    print_only_rank0("load dataset")
    train_dataset = PretrainDataset(args.dataroot, args.dataset, split="train", transforms=train_transforms, ret_index=True)
    valid_dataset = PretrainDataset(args.dataroot, args.dataset, split="valid", transforms=valid_transforms, ret_index=True)

    # initialize data loader
    batch_size = args.batch // args.world_size
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=args.workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=args.workers, pin_memory=True)

    ########################## load model
    # Load model
    image2d_teacher = VisionPredictor(model_name=args.model_name, head_arch="none", num_tasks=None, pretrained=True)
    image3d_teacher = VisionPredictor(model_name=args.model_name, head_arch="none", num_tasks=None, pretrained=True)
    num_features = image2d_teacher.in_features

    propertiesPredictor = Predictor(in_features=num_features, out_features=get_supervised_singal_dim()["basic_properties_info"])
    atomDistPredictor = Predictor(in_features=num_features, out_features=get_supervised_singal_dim()["atom_count_dict"])
    boundDistPredictor = Predictor(in_features=num_features, out_features=get_supervised_singal_dim()["bound_count_dict"])
    geometryPredictor = Predictor(in_features=num_features, out_features=get_supervised_singal_dim()["normalized_points_flatten"])

    # Loss and optimizer
    optim_params = [{"params": image2d_teacher.parameters()},
                    {"params": image3d_teacher.parameters()},
                    {"params": propertiesPredictor.parameters()},
                    {"params": atomDistPredictor.parameters()},
                    {"params": boundDistPredictor.parameters()},
                    {"params": geometryPredictor.parameters()}]
    optimizer = SGD(optim_params, momentum=args.momentum, lr=args.lr, weight_decay=args.weight_decay)
    criterionCL = SupConLoss(temperature=args.temperature, base_temperature=args.base_temperature, contrast_mode='all')
    criterionL1 = nn.L1Loss()
    criterionL1_none = nn.L1Loss(reduction="none")

    # lr scheduler
    lr_scheduler = None

    # Resume weights
    if args.resume is not None:
        flag, resume_desc = load_checkpoint(args.resume, image2d_teacher, image3d_teacher, propertiesPredictor, atomDistPredictor, boundDistPredictor,
                                            geometryPredictor, optimizer=None, lr_scheduler=lr_scheduler)
        args.start_epoch = int(resume_desc['epoch'])
        assert flag, "error in loading pretrained model {}.".format(args.resume)
        print_only_rank0("[resume description] {}".format(resume_desc))

    # model with DDP
    print_only_rank0("starting DDP.")
    # using DistributedDataParallel
    image2d_teacher = DDP(image2d_teacher.to(device), device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    image3d_teacher = DDP(image3d_teacher.to(device), device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    propertiesPredictor = DDP(propertiesPredictor.to(device), device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    atomDistPredictor = DDP(atomDistPredictor.to(device), device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    boundDistPredictor = DDP(boundDistPredictor.to(device), device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    geometryPredictor = DDP(geometryPredictor.to(device), device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    ########################## train
    best_loss = np.Inf
    for epoch in range(args.start_epoch, args.epochs):
        # set sampler
        train_loader.sampler.set_epoch(epoch)
        valid_loader.sampler.set_epoch(epoch)

        train_dict = train_one_epoch(image2d_teacher, image3d_teacher, propertiesPredictor=propertiesPredictor,
                                     atomDistPredictor=atomDistPredictor, boundDistPredictor=boundDistPredictor,
                                     geometryPredictor=geometryPredictor, optimizer=optimizer, data_loader=train_loader,
                                     criterionCL=criterionCL, criterionReg=(criterionL1, criterionL1_none),
                                     device=device, epoch=epoch, lr_scheduler=lr_scheduler,
                                     weighted_loss=args.weighted_loss, is_ddp=True)

        print_only_rank0(str(train_dict))

        evaluate_valid_results = evaluate(image2d_teacher, image3d_teacher, propertiesPredictor, atomDistPredictor,
                                          boundDistPredictor, geometryPredictor, valid_loader, criterionCL,
                                          (criterionL1, criterionL1_none), device, epoch, is_ddp=True)
        print_only_rank0("[valid evaluation] epoch: {} | {}".format(epoch, evaluate_valid_results))

        # save model
        model_dict = {"image2d_teacher": image2d_teacher, "image3d_teacher": image3d_teacher,
                      "propertiesPredictor": propertiesPredictor, "atomDistPredictor": atomDistPredictor,
                      "boundDistPredictor": boundDistPredictor, "geometryPredictor": geometryPredictor}
        optimizer_dict = {"optimizer": optimizer}
        lr_scheduler_dict = {"lr_scheduler": lr_scheduler} if lr_scheduler is not None else None

        cur_loss = train_dict["total_loss"]

        # save best model
        if is_rank0() and best_loss > cur_loss:
            files2remove = glob.glob(os.path.join(args.log_dir, "ckpts", "best_epoch*"))
            for _i in files2remove:
                os.remove(_i)
            best_loss = cur_loss
            best_pre = "best_epoch={}_loss={:.2f}".format(epoch, best_loss)
            save_checkpoint(model_dict, optimizer_dict, lr_scheduler_dict,
                            train_dict, epoch, save_path=os.path.join(args.log_dir, "ckpts"),
                            name_pre=best_pre, name_post="")

    print("training teacher model ends.")


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    # initialize some arguments
    args = parse_args()
    args.world_size = args.ngpus_per_node * args.nodes
    # run with torch.multiprocessing
    mp.spawn(main, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
