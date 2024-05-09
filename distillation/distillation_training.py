import os
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataloader.dataset import FinetuneAlignVisionGraphDatasetFactory, DualCollater
from distillation.train_utils import train_one_epoch, save_finetune_ckpt, evaluate
from model.model_utils import get_classifier
from model.teacher import ResumePredictors, ResumeSingleTeacher
from utils.public_utils import fix_train_random_seed, setup_device, get_tqdm_desc, is_left_better_right
from utils.splitter import *


def parse_args():
    # Parse arguments
    parser = ArgumentParser(description='PyTorch Implementation of IEM')

    # basic
    parser.add_argument('--dataroot', type=str, help='data root')
    parser.add_argument('--dataset', type=str, help='dataset name, e.g. bbbp, clintox, ...')
    parser.add_argument('--graph_feat', type=str, default="all", choices=['min', 'all'], help='')
    parser.add_argument('--label_column_name', type=str, default="label", help='column name of label')
    parser.add_argument('--image_dir_name', type=str, default="image3d", help='directory name of 3d image')
    parser.add_argument('--gpu', type=str, default="0", help='GPUs of CUDA_VISIBLE_DEVICES, e.g. 0,1,2,3')
    parser.add_argument('--ngpu', type=int, default=8, help='number of GPUs to use')
    parser.add_argument('--workers', default=2, type=int, help='number of data loading workers (default: 2)')

    # gnn
    parser.add_argument('--num_layers', type=int, default=5, help='the num_layers of deepergcn')
    parser.add_argument('--feat_dim', type=int, default=512, help='the dimension of topological space')
    parser.add_argument('--JK', type=str, default="last", choices=['concat', 'last', 'max', 'sum'], help='')
    parser.add_argument('--t_dropout', type=float, default=0.5, help='the dropout of gnn')
    parser.add_argument('--gnn_type', type=str, default="gin", choices=['gin', 'gcn', 'gat', 'graphsage'], help='')

    # teacher
    parser.add_argument("--resume_teacher", type=str, help='Resume training from a path of checkpoint')
    parser.add_argument("--resume_teacher_name", type=str, help='Resume training from a path of checkpoint')

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    # loss
    parser.add_argument('--weight_t', default=1, type=float, help='cross entropy of graph between pred and gt')
    parser.add_argument('--weight_te', default=0.01, type=float, help='task enhancement loss')
    parser.add_argument('--weight_ke', default=0.01, type=float, help='knowledge enhancement loss')

    # train
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42) to split dataset')
    parser.add_argument('--runseed', type=int, default=2021, help='random seed to run model (default: 2021)')
    parser.add_argument('--split', default="scaffold", type=str, choices=['scaffold'],
                        help='regularization of classification loss')
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--resume", type=str, default=None, help='Resume training from a path of checkpoint')
    parser.add_argument("--pretrain_gnn_path", type=str, default=None, help='Resume training from a path of checkpoint')
    parser.add_argument('--model_name', type=str, default="resnet18", help='model name')
    parser.add_argument('--task_type', type=str, default="classification", choices=["classification", "regression"],
                        help='task type')
    parser.add_argument('--save_finetune_ckpt', type=int, default=1, choices=[0, 1],
                        help='1 represents saving best ckpt, 0 represents no saving best ckpt')

    # log
    parser.add_argument('--log_dir', default='./logs/distillation_training/', help='path to log')

    # Parse arguments
    return parser.parse_args()


def main(args):
    ########################## basic
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    args.csv_file = f"{args.dataroot}/{args.dataset}/processed/{args.dataset}_processed_ac.csv"

    # gpus
    device, device_ids = setup_device(args.ngpu)

    # fix random seeds
    fix_train_random_seed(args.runseed)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    ########################## load teacher model
    teacher = ResumeSingleTeacher(args.model_name, args.resume_teacher, args.resume_teacher_name).eval()
    predictors = ResumePredictors(args.resume_teacher).eval()
    num_features = teacher.num_features

    if len(device_ids) > 1:
        teacher = torch.nn.DataParallel(teacher, device_ids=device_ids).cuda()
        predictors = torch.nn.DataParallel(predictors, device_ids=device_ids).cuda()
    else:
        teacher = teacher.to(device)
        predictors = predictors.to(device)

    ######################### basic information of data
    if args.task_type == "classification":
        eval_metric = "rocauc"
        valid_select = "max"
        min_value = -np.inf
        criterion = nn.BCEWithLogitsLoss(reduction="none")
    elif args.task_type == "regression":
        if args.dataset == "qm7" or args.dataset == "qm8" or args.dataset == "qm9":
            eval_metric = "mae"
            criterion = nn.L1Loss()
        else:
            eval_metric = "rmse"
            criterion = nn.MSELoss()
        valid_select = "min"
        min_value = np.inf
    else:
        raise Exception("param {} is not supported".format(args.task_type))

    print("eval_metric: {}".format(eval_metric))

    ########################## load dataset
    dataset_factory = FinetuneAlignVisionGraphDatasetFactory(args.dataroot, args.dataset, img_root=args.image_dir_name,
                                                             split=args.split, img_teacher=teacher,
                                                             use_teacher_feat=True, task_type=args.task_type,
                                                             graph_feat=args.graph_feat, ret_index=False, args=args,
                                                             cache_index=False, device=device,
                                                             cache_root="caches/" +
                                                                        [item for item in args.dataroot.split("/") if
                                                                         item != ''][-1])
    num_tasks = dataset_factory.label.shape[1]
    train_dataset, valid_dataset, test_dataset = dataset_factory.image_graph_dataset_dict["train"], \
        dataset_factory.image_graph_dataset_dict["valid"], dataset_factory.image_graph_dataset_dict["test"]
    del dataset_factory, teacher

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                                  pin_memory=True, collate_fn=DualCollater(follow_batch=[], multigpu=False))
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                                  pin_memory=True, collate_fn=DualCollater(follow_batch=[], multigpu=False))
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                                 pin_memory=True, collate_fn=DualCollater(follow_batch=[], multigpu=False))

    ########################## load student model
    if args.pretrain_gnn_path is None:
        from model.graph_base.gnns import GNN
        student_backbone = GNN(num_layer=args.num_layers, emb_dim=num_features, JK=args.JK, drop_ratio=args.t_dropout,
                               gnn_type=args.gnn_type)
    else:
        from model.graph_base.gnns_molebert import GNN_graphpred
        print("load pretrained student backbone from {}".format(args.pretrain_gnn_path))
        student_backbone = GNN_graphpred(num_layer=args.num_layers, emb_dim=300, JK=args.JK, drop_ratio=args.t_dropout,
                                         gnn_type=args.gnn_type, num_tasks=num_features)
        student_backbone.from_pretrained(args.pretrain_gnn_path)

    student_predictor = get_classifier("arch3", num_features, num_tasks)
    teacher_predictor = get_classifier("arch3", num_features, num_tasks)

    if len(device_ids) > 1:
        student_backbone = torch.nn.DataParallel(student_backbone, device_ids=device_ids).cuda()
        student_predictor = torch.nn.DataParallel(student_predictor, device_ids=device_ids).cuda()
        teacher_predictor = torch.nn.DataParallel(teacher_predictor, device_ids=device_ids).cuda()
    else:
        student_backbone = student_backbone.to(device)
        student_predictor = student_predictor.to(device)
        teacher_predictor = teacher_predictor.to(device)

    # Loss and optimizer
    optim_params = [{"params": student_backbone.parameters()},
                    {"params": student_predictor.parameters()},
                    {"params": teacher_predictor.parameters()}]
    optimizer = Adam(optim_params, lr=args.lr, weight_decay=0)

    ########################## train
    results = {
        'highest_valid': min_value,
        'final_test': min_value,
    }
    for epoch in range(args.start_epoch, args.epochs):
        tqdm_train_desc, _, tqdm_eval_val_desc, tqdm_eval_test_desc = get_tqdm_desc(args.dataset, epoch)

        train_loss = train_one_epoch(student_backbone=student_backbone, student_predictor=student_predictor,
                                     predictors=predictors, weight_t=args.weight_t,
                                     weight_te=args.weight_te, weight_ke=args.weight_ke, optimizer=optimizer,
                                     data_loader=train_dataloader, criterion=criterion, device=device, epoch=epoch,
                                     task_type=args.task_type, tqdm_desc=tqdm_train_desc)

        val_loss, val_results = evaluate(student_backbone=student_backbone, student_predictor=student_predictor,
                                         data_loader=valid_dataloader, criterion=criterion, device=device,
                                         task_type=args.task_type, tqdm_desc=tqdm_eval_val_desc)
        test_loss, test_results = evaluate(student_backbone=student_backbone, student_predictor=student_predictor,
                                           data_loader=test_dataloader, criterion=criterion, device=device,
                                           task_type=args.task_type, tqdm_desc=tqdm_eval_test_desc)

        valid_result = val_results[eval_metric.upper()]
        test_result = test_results[eval_metric.upper()]

        print({"dataset": args.dataset, "epoch": epoch, "Train Loss": train_loss, 'Validation': valid_result,
               'Test': test_result})

        if is_left_better_right(valid_result, results['highest_valid'], standard=valid_select):
            results['highest_valid'] = valid_result
            results['final_test'] = test_result

            if args.save_finetune_ckpt == 1:
                save_finetune_ckpt(student_backbone, student_predictor, optimizer, round(train_loss, 4), epoch,
                                   args.log_dir, "valid_best", lr_scheduler=None, result_dict=results)

    print("final results: {}".format(results))


if __name__ == '__main__':
    args = parse_args()
    main(args)
