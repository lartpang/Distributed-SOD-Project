# @Time    : 2020/7/4
# @Author  : Lart Pang
# @Email   : lartpang@163.com
# @File    : pipeline_ops.py
# @Project : utils/pipeline_ops.py
# @GitHub  : https://github.com/lartpang
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as sche
import torch.optim.optimizer as optim
from torch.optim import Adam, SGD

from utils.misc import construct_print


def get_total_loss(
    train_preds: torch.Tensor, train_masks: torch.Tensor, loss_funcs: list
) -> (float, list):
    """
    return the sum of the list of loss functions with train_preds and train_masks

    Args:
        train_preds (torch.Tensor): predictions
        train_masks (torch.Tensor): masks
        loss_funcs (list): the list of loss functions

    Returns: the sum of all losses and the list of result strings

    """
    loss_list = []
    loss_item_list = []

    assert len(loss_funcs) != 0, "请指定损失函数`loss_funcs`"
    for loss in loss_funcs:
        loss_out = loss(train_preds, train_masks)
        loss_list.append(loss_out)
        loss_item_list.append(f"{loss_out.item():.5f}")

    train_loss = sum(loss_list)
    return train_loss, loss_item_list


def save_checkpoint(
    model: nn.Module = None,
    optimizer: optim.Optimizer = None,
    amp=None,
    exp_name: str = "",
    current_epoch: int = 1,
    full_net_path: str = "",
    state_net_path: str = "",
):
    """
    保存完整参数模型（大）和状态参数模型（小）

    Args:
        model (nn.Module): model object
        optimizer (optim.Optimizer): optimizer object
        scheduler (sche._LRScheduler): scheduler object
        amp (): apex.amp
        exp_name (str): exp_name
        current_epoch (int): in the epoch, model **will** be trained
        full_net_path (str): the path for saving the full model parameters
        state_net_path (str): the path for saving the state dict.
    """
    net_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    state_dict = {
        "arch": exp_name,
        "epoch": current_epoch,
        "net_state": net_state,
        "opti_state": optimizer.state_dict(),
        "amp_state": amp.state_dict() if amp else None,
    }
    torch.save(state_dict, full_net_path)
    torch.save(net_state, state_net_path)


def resume_checkpoint(
    model: nn.Module = None,
    optimizer: optim.Optimizer = None,
    amp=None,
    exp_name: str = "",
    load_path: str = "",
    mode: str = "all",
    local_rank: int = 0,
):
    """
    从保存节点恢复模型

    Args:
        model (nn.Module): model object
        optimizer (optim.Optimizer): optimizer object
        scheduler (sche._LRScheduler): scheduler object
        amp (): apex.amp
        exp_name (str): exp_name
        load_path (str): 模型存放路径
        mode (str): 选择哪种模型恢复模式:
            - 'all': 回复完整模型，包括训练中的的参数；
            - 'onlynet': 仅恢复模型权重参数
        local_rank (int): 指定权重加载的目标GPU

    Returns mode: 'all' start_epoch; 'onlynet' None
    """
    if os.path.exists(load_path) and os.path.isfile(load_path):
        construct_print(f"Loading checkpoint '{load_path}'")
        checkpoint = torch.load(load_path, map_location={"cuda:0": f"cuda:{local_rank}"})
        if mode == "all":
            if exp_name == checkpoint["arch"]:
                start_epoch = checkpoint["epoch"]
                # "net_state": model.module.state_dict()
                # if hasattr(model, "module") else model.state_dict(),
                if hasattr(model, "module"):
                    model.module.load_state_dict(checkpoint["net_state"])
                else:
                    model.load_state_dict(checkpoint["net_state"])
                optimizer.load_state_dict(checkpoint["opti_state"])
                if checkpoint["amp_state"]:
                    if amp:
                        amp.load_state_dict(checkpoint["amp_state"])
                    else:
                        construct_print("You are not using amp.")
                else:
                    construct_print("The state_dict of amp is None.")
                construct_print(
                    f"Loaded '{load_path}' " f"(will train at epoch" f" {checkpoint['epoch']})"
                )
                return start_epoch
            else:
                raise Exception(f"{load_path} does not match.")
        elif mode == "onlynet":
            if hasattr(model, "module"):
                model.module.load_state_dict(checkpoint["net_state"])
            else:
                model.load_state_dict(checkpoint["net_state"])
            construct_print(
                f"Loaded checkpoint '{load_path}' " f"(only has the model's weight params)"
            )
        else:
            raise NotImplementedError
    else:
        raise Exception(f"{load_path}路径不正常，请检查")


def make_scheduler(
    optimizer: optim.Optimizer, total_num: int, scheduler_type: str, scheduler_info: dict
) -> sche._LRScheduler:
    def get_lr_coefficient(curr_epoch):
        nonlocal total_num
        # curr_epoch start from 0
        # total_num = iter_num if args["sche_usebatch"] else end_epoch
        if scheduler_type == "poly":
            coefficient = pow((1 - float(curr_epoch) / total_num), scheduler_info["lr_decay"])
        elif scheduler_type == "poly_warmup":
            turning_epoch = scheduler_info["warmup_epoch"]
            if curr_epoch < turning_epoch:
                # 0,1,2,...,turning_epoch-1
                coefficient = 1 / turning_epoch * (1 + curr_epoch)
            else:
                # turning_epoch,...,end_epoch
                curr_epoch -= turning_epoch - 1
                total_num -= turning_epoch - 1
                coefficient = pow((1 - float(curr_epoch) / total_num), scheduler_info["lr_decay"])
        elif scheduler_type == "cosine_warmup":
            turning_epoch = scheduler_info["warmup_epoch"]
            if curr_epoch < turning_epoch:
                # 0,1,2,...,turning_epoch-1
                coefficient = 1 / turning_epoch * (1 + curr_epoch)
            else:
                # turning_epoch,...,end_epoch
                curr_epoch -= turning_epoch - 1
                total_num -= turning_epoch - 1
                coefficient = (1 + np.cos(np.pi * curr_epoch / total_num)) / 2
        elif scheduler_type == "f3_sche":
            coefficient = 1 - abs((curr_epoch + 1) / (total_num + 1) * 2 - 1)
        else:
            raise NotImplementedError
        return coefficient

    scheduler = sche.LambdaLR(optimizer, lr_lambda=get_lr_coefficient)
    return scheduler


class CustomScheduler(object):
    def __init__(
        self, optimizer: optim.Optimizer, total_num: int, scheduler_type: str, scheduler_info: dict
    ):
        self.lr_group = [param["lr"] for param in optimizer.param_groups]
        self.total_num = total_num
        self.type = scheduler_type
        self.info = scheduler_info

    def __get_lr_coefficient(self, curr_epoch):
        # curr_epoch start from 0
        # total_num = iter_num if args["sche_usebatch"] else end_epoch
        if self.type == "poly":
            coefficient = pow((1 - float(curr_epoch) / self.total_num), self.info["lr_decay"])
        elif self.type == "poly_warmup":
            turning_epoch = self.info["warmup_epoch"]
            if curr_epoch < turning_epoch:
                # 0,1,2,...,turning_epoch-1
                coefficient = 1 / turning_epoch * (1 + curr_epoch)
            else:
                # turning_epoch,...,end_epoch
                curr_epoch -= turning_epoch - 1
                self.total_num -= turning_epoch - 1
                coefficient = pow((1 - float(curr_epoch) / self.total_num), self.info["lr_decay"])
        elif self.type == "cosine_warmup":
            turning_epoch = self.info["warmup_epoch"]
            if curr_epoch < turning_epoch:
                # 0,1,2,...,turning_epoch-1
                coefficient = 1 / turning_epoch * (1 + curr_epoch)
            else:
                # turning_epoch,...,end_epoch
                curr_epoch -= turning_epoch - 1
                self.total_num -= turning_epoch - 1
                coefficient = (1 + np.cos(np.pi * curr_epoch / self.total_num)) / 2
        elif self.type == "f3_sche":
            coefficient = 1 - abs((curr_epoch + 1) / (self.total_num + 1) * 2 - 1)
        else:
            raise NotImplementedError
        return coefficient

    def step(self, optimizer, curr_epoch):
        for idx, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = self.lr_group[idx] * self.__get_lr_coefficient(
                curr_epoch=curr_epoch
            )

    def __str__(self):
        return f"Scheduler:\n\tType: {self.type}\n\tInfo: {self.info}\n"


def make_optimizer(model: nn.Module, optimizer_type: str, optimizer_info: dict) -> optim.Optimizer:
    if optimizer_type == "sgd_trick":
        # https://github.com/implus/PytorchInsight/blob/master/classification/imagenet_tricks.py
        params = [
            {
                "params": [
                    p for name, p in model.named_parameters() if ("bias" in name or "bn" in name)
                ],
                "weight_decay": 0,
            },
            {
                "params": [
                    p
                    for name, p in model.named_parameters()
                    if ("bias" not in name and "bn" not in name)
                ]
            },
        ]
        optimizer = SGD(
            params,
            lr=optimizer_info["lr"],
            momentum=optimizer_info["momentum"],
            weight_decay=optimizer_info["weight_decay"],
            nesterov=optimizer_info["nesterov"],
        )
    elif optimizer_type == "sgd_r3":
        params = [
            # 不对bias参数执行weight decay操作，weight decay主要的作用就是通过对网络
            # 层的参数（包括weight和bias）做约束（L2正则化会使得网络层的参数更加平滑）达
            # 到减少模型过拟合的效果。
            {
                "params": [
                    param for name, param in model.named_parameters() if name[-4:] == "bias"
                ],
                "lr": 2 * optimizer_info["lr"],
            },
            {
                "params": [
                    param for name, param in model.named_parameters() if name[-4:] != "bias"
                ],
                "lr": optimizer_info["lr"],
                "weight_decay": optimizer_info["weight_decay"],
            },
        ]
        optimizer = SGD(params, momentum=optimizer_info["momentum"])
    elif optimizer_type == "sgd_all":
        optimizer = SGD(
            model.parameters(),
            lr=optimizer_info["lr"],
            weight_decay=optimizer_info["weight_decay"],
            momentum=optimizer_info["momentum"],
        )
    elif optimizer_type == "adam":
        optimizer = Adam(
            model.parameters(),
            lr=optimizer_info["lr"],
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=optimizer_info["weight_decay"],
        )
    elif optimizer_type == "f3_trick":
        backbone, head = [], []
        for name, params_tensor in model.named_parameters():
            if name.startswith("div_2"):
                pass
            elif name.startswith("div"):
                backbone.append(params_tensor)
            else:
                head.append(params_tensor)
        params = [
            {"params": backbone, "lr": 0.1 * optimizer_info["lr"]},
            {"params": head, "lr": optimizer_info["lr"]},
        ]
        optimizer = SGD(
            params=params,
            momentum=optimizer_info["momentum"],
            weight_decay=optimizer_info["weight_decay"],
            nesterov=optimizer_info["nesterov"],
        )
    else:
        raise NotImplementedError
    return optimizer


if __name__ == "__main__":
    a = torch.rand((3, 3)).bool()
    print(isinstance(a, torch.FloatTensor), a.type())
