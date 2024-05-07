import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

import sys
import os
import torch
from tqdm import tqdm
from utils.evaluate import metric
from utils.evaluate import metric_multitask
from utils.evaluate import metric_reg
from utils.evaluate import metric_reg_multitask
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F


# save checkpoint
def save_finetune_ckpt(backbone, predictor, optimizer, loss, epoch, save_path, filename_pre,
                       lr_scheduler=None, result_dict=None, logger=None):
    log = logger.info if logger is not None else print
    backbone_cpu = {k: v.cpu() for k, v in backbone.state_dict().items()}
    predictor_cpu = {k: v.cpu() for k, v in predictor.state_dict().items()}
    lr_scheduler = None if lr_scheduler is None else lr_scheduler.state_dict()
    state = {
            'epoch': epoch,
            'backbone': backbone_cpu,
            'predictor': predictor_cpu,
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler,
            'loss': loss,
            'result_dict': result_dict
        }
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        log("Directory {} is created.".format(save_path))

    filename = '{}/{}.pth'.format(save_path, filename_pre)
    torch.save(state, filename)
    log('model has been saved as {}'.format(filename))


def cal_downstream_loss(y_logit, labels, criterion, task_type):
    if task_type == "classification":
        is_valid = labels != -1
        loss_mat = criterion(y_logit.double(), labels)
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
        loss = torch.sum(loss_mat) / torch.sum(is_valid)
    elif task_type == "regression":
        loss = criterion(y_logit.double(), labels)
    return loss


def train_one_epoch(student_backbone, student_predictor, predictors, weight_t, weight_kd, weight_ke,
                    optimizer, data_loader, criterion, device, epoch, task_type, tqdm_desc="", teacher_clip=-1):
    if teacher_clip != -1 and epoch >= teacher_clip:
        weight_kd, weight_ke = 0, 0

    assert task_type in ["classification", "regression"]

    student_backbone.train()
    student_predictor.train()
    predictors.eval()
    accu_loss = torch.zeros(1).to(device)
    accu_g_loss = torch.zeros(1).to(device)
    accu_kd_loss = torch.zeros(1).to(device)
    accu_fe_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, desc=tqdm_desc)
    for step, data in enumerate(data_loader):
        images, graphs, labels, mask = data

        images, graphs, labels, mask = images.to(device), graphs.to(device), labels.to(device), mask.to(device)
        sample_num += images.shape[0]

        if "GNN_graphpred" in str(type(student_backbone)):
            graph_representation = student_backbone(graphs)
        else:
            node_representation = student_backbone(graphs)
            graph_representation = global_mean_pool(node_representation, graphs.batch)
        y_logit_student = student_predictor(graph_representation)

        if weight_kd != 0:
            y_logit_teacher = student_predictor(images)

        if weight_ke != 0:
            y_ke_teacher = predictors(images)
            y_ke_student = predictors(graph_representation)

        labels = labels.view(y_logit_student.shape).to(torch.float64)

        ################### calculate loss
        g_loss = cal_downstream_loss(y_logit_student, labels, criterion, task_type)

        if weight_kd != 0:
            kd_loss = F.smooth_l1_loss(y_logit_student, y_logit_teacher, reduction='mean')
        else:
            kd_loss = torch.zeros(1).to(device)

        if weight_ke != 0:
            y_ke_student_concat = torch.concat(y_ke_student, -1)
            y_ke_teacher_concat = torch.concat(y_ke_teacher, -1)

            full_mask = torch.ones(y_ke_student_concat.shape).long().to(device)
            start = y_ke_student_concat.shape[1] - y_ke_student[-1].shape[-1]
            full_mask[:, start:] = full_mask[:, start:] * mask
            fe_loss = (F.smooth_l1_loss(y_ke_student_concat, y_ke_teacher_concat, reduction='none') * full_mask).mean()
        else:
            fe_loss = torch.zeros(1).to(device)

        loss = g_loss * weight_t + kd_loss * weight_kd + fe_loss * weight_ke
        loss.backward()
        accu_loss += loss.detach()
        accu_g_loss += g_loss.detach()
        accu_kd_loss += kd_loss.detach()
        accu_fe_loss += fe_loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}; g_loss: {:.3f}; kd_loss: {:.3f}; fe_loss: {:.3f}; ".\
            format(epoch, accu_loss.item() / (step + 1), accu_g_loss.item() / (step + 1),
                   accu_kd_loss.item() / (step + 1), accu_fe_loss.item() / (step + 1))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1)


@torch.no_grad()
def evaluate(student_backbone, student_predictor, data_loader, criterion, device, task_type="classification", tqdm_desc="", ret_index=False):
    assert task_type in ["classification", "regression"]

    student_backbone.eval()
    student_predictor.eval()

    accu_loss = torch.zeros(1).to(device)

    y_scores, y_true, y_pred, y_prob = [], [], [], []
    sample_num = 0
    data_loader = tqdm(data_loader, desc=tqdm_desc)
    for step, data in enumerate(data_loader):
        if ret_index:
            _, images, graphs, labels, mask = data
        else:
            images, graphs, labels, mask = data
        images, graphs, labels, mask = images.to(device), graphs.to(device), labels.to(device), mask.to(device)
        sample_num += images.shape[0]

        with torch.no_grad():
            if "GNN_graphpred" in str(type(student_backbone)):
                graph_representation = student_backbone(graphs)
            else:
                node_representation = student_backbone(graphs)
                graph_representation = global_mean_pool(node_representation, graphs.batch)
            y_logit_student = student_predictor(graph_representation)

            labels = labels.view(y_logit_student.shape).to(torch.float64)
            if task_type == "classification":
                is_valid = labels != -1
                loss_mat = criterion(y_logit_student.double(), labels)
                loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
                loss = torch.sum(loss_mat) / torch.sum(is_valid)
            elif task_type == "regression":
                loss = criterion(y_logit_student.double(), labels)
            accu_loss += loss.detach()
            data_loader.desc = "{}; loss: {:.3f}".format(tqdm_desc, accu_loss.item() / (step + 1))

        y_true.append(labels.view(y_logit_student.shape))
        y_scores.append(y_logit_student)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    if y_true.shape[1] == 1:
        if task_type == "classification":
            y_pro = torch.sigmoid(torch.Tensor(y_scores))
            y_pred = torch.where(y_pro > 0.5, torch.Tensor([1]), torch.Tensor([0])).numpy()
            return accu_loss.item() / (step + 1), metric(y_true, y_pred, y_pro, empty=-1)
        elif task_type == "regression":
            return accu_loss.item() / (step + 1), metric_reg(y_true, y_scores)
    elif y_true.shape[1] > 1:  # multi-task
        if task_type == "classification":
            y_pro = torch.sigmoid(torch.Tensor(y_scores))
            y_pred = torch.where(y_pro > 0.5, torch.Tensor([1]), torch.Tensor([0])).numpy()
            return accu_loss.item() / (step + 1), metric_multitask(y_true, y_pred, y_pro, num_tasks=y_true.shape[1], empty=-1)
        elif task_type == "regression":
            return accu_loss.item() / (step + 1), metric_reg_multitask(y_true, y_scores, num_tasks=y_true.shape[1])
    else:
        raise Exception("error in the number of task.")

