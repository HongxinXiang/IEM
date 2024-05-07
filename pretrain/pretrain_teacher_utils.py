import sys

import torch
from torch.distributed.nn import all_gather
from tqdm import tqdm


def train_one_epoch(image2d_teacher, image3d_teacher, propertiesPredictor, atomDistPredictor, boundDistPredictor,
                    geometryPredictor, optimizer, data_loader, criterionCL, criterionReg, device, epoch,
                    weighted_loss=False, lr_scheduler=None, is_ddp=False):
    criterionReg, criterionReg_none = criterionReg

    image2d_teacher.train()
    image3d_teacher.train()
    propertiesPredictor.train()
    atomDistPredictor.train()
    boundDistPredictor.train()
    geometryPredictor.train()

    accu_loss = torch.zeros(1).to(device)
    accu_CL_loss = torch.zeros(1).to(device)
    accu_img_loss = torch.zeros(1).to(device)
    accu_image3d_loss = torch.zeros(1).to(device)

    accu_props_img_loss = torch.zeros(1).to(device)
    accu_atom_img_loss = torch.zeros(1).to(device)
    accu_bound_img_loss = torch.zeros(1).to(device)
    accu_geom_img_loss = torch.zeros(1).to(device)

    accu_props_image3d_loss = torch.zeros(1).to(device)
    accu_atom_image3d_loss = torch.zeros(1).to(device)
    accu_bound_image3d_loss = torch.zeros(1).to(device)
    accu_geom_image3d_loss = torch.zeros(1).to(device)

    optimizer.zero_grad()

    train_dict = {}
    data_loader = tqdm(data_loader, total=len(data_loader))
    for step, (
            image, image3d, label_properties_dist, label_atom_dist, label_bound_dist, label_points_dist,
            label_points_mask,
            indexs) in enumerate(data_loader):
        n_samples, n_views, n_chanel, h, w = image3d.shape
        if n_samples <= 1:
            continue
        image, image3d, label_properties_dist, label_atom_dist, label_bound_dist, label_points_dist, label_points_mask, indexs = image.to(
            device), image3d.to(device), label_properties_dist.to(device), label_atom_dist.to(
            device), label_bound_dist.to(device), label_points_dist.to(device), label_points_mask.to(
            device), indexs.to(device)

        # forward
        feat_img, feat_image3d = image2d_teacher(image), image3d_teacher(
            image3d.reshape(n_samples * n_views, n_chanel, h, w)).reshape(n_samples, n_views, -1)
        feat_image3d_mean = feat_image3d.mean(1)

        # contrastive learning between regular images and multi-view 3D images
        if is_ddp:
            all_feat1 = torch.cat(all_gather(feat_img), dim=0)
            all_feat2 = torch.cat(all_gather(feat_image3d_mean), dim=0)
            indexs = torch.cat(all_gather(indexs), dim=0)
            L_CL = criterionCL(all_feat1, all_feat2, labels=indexs)
        else:
            L_CL = criterionCL(feat_img, feat_image3d_mean, labels=indexs)

        # feature enhance
        bar_props_img, bar_atom_dist_img, bar_bound_dist_img, bar_geom_img = propertiesPredictor(
            feat_img), atomDistPredictor(feat_img), boundDistPredictor(feat_img), geometryPredictor(feat_img)
        bar_props_image3d, bar_atom_dist_image3d, bar_bound_dist_image3d, bar_geom_image3d = propertiesPredictor(
            feat_image3d_mean), atomDistPredictor(feat_image3d_mean), boundDistPredictor(
            feat_image3d_mean), geometryPredictor(feat_image3d_mean)

        # 2D image loss
        L_props_img = criterionReg(bar_props_img, label_properties_dist)
        L_atom_dist_img = criterionReg(bar_atom_dist_img, label_atom_dist)
        L_bound_dist_img = criterionReg(bar_bound_dist_img, label_bound_dist)
        L_geom_img = (criterionReg_none(bar_geom_img,
                                        label_points_dist) * label_points_mask).sum() / label_points_mask.sum()
        L_img = L_props_img + L_atom_dist_img + L_bound_dist_img + L_geom_img

        # 3D image loss
        L_props_image3d = criterionReg(bar_props_image3d, label_properties_dist)
        L_atom_dist_image3d = criterionReg(bar_atom_dist_image3d, label_atom_dist)
        L_bound_dist_image3d = criterionReg(bar_bound_dist_image3d, label_bound_dist)
        L_geom_image3d = (criterionReg_none(bar_geom_image3d,
                                            label_points_dist) * label_points_mask).sum() / label_points_mask.sum()
        L_image3d = L_props_image3d + L_atom_dist_image3d + L_bound_dist_image3d + L_geom_image3d

        # backward
        if weighted_loss:
            loss = L_CL + L_img + L_image3d

            weight_L_CL = (L_CL / loss * 9).detach()

            weight_L_props_img = (L_props_img / loss * 9).detach()
            weight_L_atom_dist_img = (L_atom_dist_img / loss * 9).detach()
            weight_L_bound_dist_img = (L_bound_dist_img / loss * 9).detach()
            weight_L_geom_img = (L_geom_img / loss * 9).detach()

            weight_L_props_image3d = (L_props_image3d / loss * 9).detach()
            weight_L_atom_dist_image3d = (L_atom_dist_image3d / loss * 9).detach()
            weight_L_bound_dist_image3d = (L_bound_dist_image3d / loss * 9).detach()
            weight_L_geom_image3d = (L_geom_image3d / loss * 9).detach()

            weighted_L_CL = weight_L_CL * L_CL
            weighted_L_img = weight_L_props_img * L_props_img + weight_L_atom_dist_img * L_atom_dist_img + weight_L_bound_dist_img * L_bound_dist_img + weight_L_geom_img * L_geom_img
            weighted_L_image3d = weight_L_props_image3d * L_props_image3d + weight_L_atom_dist_image3d * L_atom_dist_image3d + weight_L_bound_dist_image3d * L_bound_dist_image3d + weight_L_geom_image3d * L_geom_image3d

            weighted_loss = weighted_L_CL + weighted_L_img + weighted_L_image3d
            weighted_loss.backward()
        else:
            loss = L_CL + L_img + L_image3d
            loss.backward()

        # logger
        accu_loss += loss.detach()
        accu_CL_loss += L_CL.detach()
        accu_img_loss += L_img.detach()
        accu_image3d_loss += L_image3d.detach()

        accu_props_img_loss += L_props_img
        accu_atom_img_loss += L_atom_dist_img
        accu_bound_img_loss += L_bound_dist_img
        accu_geom_img_loss += L_geom_img

        accu_props_image3d_loss += L_props_image3d
        accu_atom_image3d_loss += L_atom_dist_image3d
        accu_bound_image3d_loss += L_bound_dist_image3d
        accu_geom_image3d_loss += L_geom_image3d

        data_loader.desc = "[train epoch {}] total loss: {:.3f}; CL loss: {:.3f}; " \
                           "image loss: {:.3f} [props: {:.3f}; atom: {:.3f}; bound: {:.3f}; geom: {:.3f}]; " \
                           "image3d loss: {:.3f}; [props: {:.3f}; atom: {:.3f}; bound: {:.3f}; geom: {:.3f}]".format(
            epoch, accu_loss.item() / (step + 1), accu_CL_loss.item() / (step + 1), accu_img_loss.item() / (step + 1),
                   accu_props_img_loss.item() / (step + 1), accu_atom_img_loss.item() / (step + 1),
                   accu_bound_img_loss.item() / (step + 1), accu_geom_img_loss.item() / (step + 1),
                   accu_image3d_loss.item() / (step + 1), accu_props_image3d_loss.item() / (step + 1),
                   accu_atom_image3d_loss.item() / (step + 1), accu_bound_image3d_loss.item() / (step + 1),
                   accu_geom_image3d_loss.item() / (step + 1))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

        train_dict = {
            "step": (step + 1) + len(data_loader) * epoch,
            "epoch": epoch + (step + 1) / len(data_loader),
            "total_loss": accu_loss.item() / (step + 1),
            "CL_loss": accu_CL_loss.item() / (step + 1),

            "image2d loss": accu_img_loss.item() / (step + 1),
            "image2d props loss": accu_props_img_loss.item() / (step + 1),
            "image2d atom loss": accu_atom_img_loss.item() / (step + 1),
            "image2d bound loss": accu_bound_img_loss.item() / (step + 1),
            "image2d geom loss": accu_geom_img_loss.item() / (step + 1),

            "image3d loss": accu_image3d_loss.item() / (step + 1),
            "image3d props loss": accu_props_image3d_loss.item() / (step + 1),
            "image3d atom loss": accu_atom_image3d_loss.item() / (step + 1),
            "image3d bound loss": accu_bound_image3d_loss.item() / (step + 1),
            "image3d geom loss": accu_geom_image3d_loss.item() / (step + 1)
        }

    # Update learning rates
    if lr_scheduler is not None:
        lr_scheduler.step()

    return train_dict


def evaluate(image2d_teacher, image3d_teacher, propertiesPredictor, atomDistPredictor, boundDistPredictor,
             geometryPredictor,
             data_loader, criterionCL, criterionReg, device, epoch, is_ddp=False):
    criterionReg, criterionReg_none = criterionReg

    image2d_teacher.eval()
    image3d_teacher.eval()
    propertiesPredictor.eval()
    atomDistPredictor.eval()
    boundDistPredictor.eval()
    geometryPredictor.eval()

    accu_loss = 0
    accu_CL_loss = 0
    accu_img_loss = 0
    accu_image3d_loss = 0

    accu_props_img_loss = 0
    accu_atom_img_loss = 0
    accu_bound_img_loss = 0
    accu_geom_img_loss = 0

    accu_props_image3d_loss = 0
    accu_atom_image3d_loss = 0
    accu_bound_image3d_loss = 0
    accu_geom_image3d_loss = 0

    train_dict = {}
    data_loader = tqdm(data_loader, total=len(data_loader))
    for step, (
            image, image3d, label_properties_dist, label_atom_dist, label_bound_dist, label_points_dist,
            label_points_mask,
            indexs) in enumerate(data_loader):
        n_samples, n_views, n_chanel, h, w = image3d.shape
        if n_samples <= 1:
            continue
        image, image3d, label_properties_dist, label_atom_dist, label_bound_dist, label_points_dist, label_points_mask, indexs = image.to(
            device), image3d.to(device), label_properties_dist.to(device), label_atom_dist.to(
            device), label_bound_dist.to(device), label_points_dist.to(device), label_points_mask.to(
            device), indexs.to(device)

        # forward
        feat_img, feat_image3d = image2d_teacher(image), image3d_teacher(
            image3d.reshape(n_samples * n_views, n_chanel, h, w)).reshape(n_samples, n_views, -1)
        feat_image3d_mean = feat_image3d.mean(1)

        # contrastive learning between regular images generated by rdkit and PyMol 3D images
        if is_ddp:
            all_feat1 = torch.cat(all_gather(feat_img), dim=0)
            all_feat2 = torch.cat(all_gather(feat_image3d_mean), dim=0)
            indexs = torch.cat(all_gather(indexs), dim=0)
            L_CL = criterionCL(all_feat1, all_feat2, labels=indexs)
        else:
            L_CL = criterionCL(feat_img, feat_image3d_mean, labels=indexs)

        # knowledge enhance
        bar_props_img, bar_atom_dist_img, bar_bound_dist_img, bar_geom_img = propertiesPredictor(
            feat_img), atomDistPredictor(feat_img), boundDistPredictor(feat_img), geometryPredictor(feat_img)
        bar_props_image3d, bar_atom_dist_image3d, bar_bound_dist_image3d, bar_geom_image3d = propertiesPredictor(
            feat_image3d_mean), atomDistPredictor(feat_image3d_mean), boundDistPredictor(
            feat_image3d_mean), geometryPredictor(feat_image3d_mean)

        # image loss
        L_props_img = criterionReg(bar_props_img, label_properties_dist)
        L_atom_dist_img = criterionReg(bar_atom_dist_img, label_atom_dist)
        L_bound_dist_img = criterionReg(bar_bound_dist_img, label_bound_dist)
        L_geom_img = (criterionReg_none(bar_geom_img,
                                        label_points_dist) * label_points_mask).sum() / label_points_mask.sum()
        L_img = L_props_img + L_atom_dist_img + L_bound_dist_img + L_geom_img

        # image3d loss
        L_props_image3d = criterionReg(bar_props_image3d, label_properties_dist)
        L_atom_dist_image3d = criterionReg(bar_atom_dist_image3d, label_atom_dist)
        L_bound_dist_image3d = criterionReg(bar_bound_dist_image3d, label_bound_dist)
        L_geom_image3d = (criterionReg_none(bar_geom_image3d,
                                            label_points_dist) * label_points_mask).sum() / label_points_mask.sum()
        L_image3d = L_props_image3d + L_atom_dist_image3d + L_bound_dist_image3d + L_geom_image3d

        loss = L_CL + L_img + L_image3d

        # logger
        accu_loss += loss.item()
        accu_CL_loss += L_CL.item()
        accu_img_loss += L_img.item()
        accu_image3d_loss += L_image3d.item()

        accu_props_img_loss += L_props_img.item()
        accu_atom_img_loss += L_atom_dist_img.item()
        accu_bound_img_loss += L_bound_dist_img.item()
        accu_geom_img_loss += L_geom_img.item()

        accu_props_image3d_loss += L_props_image3d.item()
        accu_atom_image3d_loss += L_atom_dist_image3d.item()
        accu_bound_image3d_loss += L_bound_dist_image3d.item()
        accu_geom_image3d_loss += L_geom_image3d.item()

        data_loader.desc = "[evaluate epoch {}] total loss: {:.3f}; CL loss: {:.3f}; " \
                           "image loss: {:.3f} [props: {:.3f}; atom: {:.3f}; bound: {:.3f}; geom: {:.3f}]; " \
                           "image3d loss: {:.3f}; [props: {:.3f}; atom: {:.3f}; bound: {:.3f}; geom: {:.3f}]".format(
            epoch, accu_loss / (step + 1), accu_CL_loss / (step + 1), accu_img_loss / (step + 1),
                   accu_props_img_loss / (step + 1), accu_atom_img_loss / (step + 1),
                   accu_bound_img_loss / (step + 1), accu_geom_img_loss / (step + 1),
                   accu_image3d_loss / (step + 1), accu_props_image3d_loss / (step + 1),
                   accu_atom_image3d_loss / (step + 1), accu_bound_image3d_loss / (step + 1),
                   accu_geom_image3d_loss / (step + 1))

        train_dict = {
            "step": (step + 1) + len(data_loader) * epoch,
            "epoch": epoch + (step + 1) / len(data_loader),
            "total_loss": accu_loss / (step + 1),
            "CL_loss": accu_CL_loss / (step + 1),

            "image2d loss": accu_img_loss / (step + 1),
            "image2d props loss": accu_props_img_loss / (step + 1),
            "image2d atom loss": accu_atom_img_loss / (step + 1),
            "image2d bound loss": accu_bound_img_loss / (step + 1),
            "image2d geom loss": accu_geom_img_loss / (step + 1),

            "image3d loss": accu_image3d_loss / (step + 1),
            "image3d props loss": accu_props_image3d_loss / (step + 1),
            "image3d atom loss": accu_atom_image3d_loss / (step + 1),
            "image3d bound loss": accu_bound_image3d_loss / (step + 1),
            "image3d geom loss": accu_geom_image3d_loss / (step + 1)
        }

    return train_dict
