# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils
import torch.nn.functional as F
from geomloss import SamplesLoss

# import wandb
import torch.distributed as dist


def train_one_epoch(
    model: torch.nn.Module,
    criterion: DistillationLoss,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    model_ema: Optional[ModelEma] = None,
    mixup_fn: Optional[Mixup] = None,
    set_training_mode=True,
    args=None,
):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10
    # sinkhorn_loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=0.01)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True, dtype=torch.int64)

        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

        # if args.bce_loss:
        #     targets = targets.gt(0.0).type(targets.dtype)

        with torch.cuda.amp.autocast():
            outputs = model(images, targets)
            pred_jigsaw = outputs.pred_jigsaw
            gt_jigsaw = outputs.gt_jigsaw
            # loss = 0 * criterion(images, outputs.sup, targets)
            if args.rec is True:
                loss_rec = outputs.rec_loss * args.lambda_rec
                loss_jigsaw = F.cross_entropy(pred_jigsaw, gt_jigsaw)
                loss = loss_jigsaw + loss_rec
            else:
                pred_jigsaw_prob = F.softmax(outputs.pred_jigsaw, dim=-1)
                gt_jigsaw_one_hot = F.one_hot(
                    outputs.gt_jigsaw, num_classes=pred_jigsaw_prob.size(-1)
                ).float()
                # BCE LOSS
                loss_jigsaw = criterion(pred_jigsaw_prob, gt_jigsaw_one_hot)

                # SINKHORN LOSS
                # loss_jigsaw = sinkhorn_loss_fn(
                #     pred_jigsaw_prob, gt_jigsaw_one_hot
                # ).mean()

                # CE LOSS
                # loss_jigsaw = criterion(outputs.pred_jigsaw, outputs.gt_jigsaw)
                loss = loss_jigsaw

        loss_value = loss.item()
        if args.rec is True:
            loss_rec_value = loss_rec.item()
        loss_jigsaw_value = loss_jigsaw.item()

        predicted_patch_indices = torch.argmax(pred_jigsaw_prob, dim=-1)
        gt_patch_indices = torch.argmax(gt_jigsaw_one_hot, dim=-1)
        correct_preds = (predicted_patch_indices == gt_patch_indices).all(dim=-1)
        num_corrects = correct_preds.sum().item()
        num_total = images.size(0)
        running_accuracy = num_corrects / num_total

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = (
            hasattr(optimizer, "is_second_order") and optimizer.is_second_order
        )
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=is_second_order,
        )

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss_total=loss_value)
        if args.rec is True:
            metric_logger.update(loss_rec=loss_rec_value)
        metric_logger.update(loss_jigsaw=loss_jigsaw_value)
        metric_logger.update(jigsaw_acc=running_accuracy)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # if dist.get_rank() == 0:
        #     wandb.log(
        #         {
        #             "total_loss": loss_value,
        #             "jigsaw_loss": loss_jigsaw_value,
        #             "jigsaw_acc": running_accuracy,
        #         }
        #     )
        #     if args.rec is True:
        #         wandb.log({"rec_loss": loss_rec_value})
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True, dtype=torch.int64)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images, target)
            pred_jigsaw_prob = F.softmax(output.pred_jigsaw, dim=-1)
            gt_jigsaw_one_hot = F.one_hot(
                output.gt_jigsaw, num_classes=pred_jigsaw_prob.size(-1)
            ).float()
            # BCE LOSS
            loss_jigsaw = criterion(pred_jigsaw_prob, gt_jigsaw_one_hot)

            # SINKHORN LOSS
            # loss_jigsaw = sinkhorn_loss_fn(
            #     pred_jigsaw_prob, gt_jigsaw_one_hot
            # ).mean()

            # CE LOSS
            # loss_jigsaw = criterion(outputs.pred_jigsaw, outputs.gt_jigsaw)
            loss = loss_jigsaw

        # acc1, acc5 = accuracy(output.sup, target, topk=(1, 5))
        predicted_patch_indices = torch.argmax(pred_jigsaw_prob, dim=-1)
        gt_patch_indices = torch.argmax(gt_jigsaw_one_hot, dim=-1)
        correct_preds = (predicted_patch_indices == gt_patch_indices).all(dim=-1)
        num_corrects = correct_preds.sum().item()
        batch_size = images.shape[0]
        batch_acc = num_corrects / batch_size

        metric_logger.update(loss=loss.item())
        # metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        # metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["acc"].update(batch_acc, n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print(
    #     "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}".format(
    #         top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss
    #     )
    # )
    print(
        "* Acc {acc.global_avg:.3f} loss {losses.global_avg:.3f}".format(
            acc=metric_logger.acc, losses=metric_logger.loss
        )
    )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def infer_perm(data_loader, model, device):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()

    all_pred_labels = []
    all_targets = []

    patch_num_per_img = 36

    for images, targets in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            pred_jigsaw = output.pred_jigsaw

        # convert logits to hard labels
        pred_labels = torch.argmax(pred_jigsaw, dim=1)

        # reshape and convert to list of lists
        pred_labels_list = pred_labels.view(-1, patch_num_per_img).tolist()
        all_pred_labels.extend(pred_labels_list)

        targets_list = targets.tolist()
        all_targets.extend(targets_list)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    return all_pred_labels, all_targets
