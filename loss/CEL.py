# -*- coding: utf-8 -*-
# @Time    : 2020/3/28
# @Author  : Lart Pang
# @FileName: CEL.py
# @GitHub  : https://github.com/lartpang/MINet/blob/master/code/loss/CEL.py
import torch
from torch import nn


class CEL(nn.Module):
    def __init__(self):
        super(CEL, self).__init__()
        self.eps = 1e-6

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pred = pred.sigmoid()
        intersection = pred * target
        numerator = (pred - intersection).sum() + (target - intersection).sum()
        denominator = pred.sum() + target.sum()
        return numerator / (denominator + self.eps)

    def __str__(self):
        return "You are using `CEL`!"
