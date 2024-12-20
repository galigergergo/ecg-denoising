# -*- coding: utf-8 -*-
"""
Created on June 17, 2020

ISTANet(shared network with 4 conv + ReLU) + regularized hyperparameters softplus(w*x + b). 
The Intention is to make gradient step mu and thresholding value theta positive and monotonically decrease.

@author: XIANG
"""

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from os.path import join as pjoin


def initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, 0, 0.01)
            init.constant_(m.bias, 0)


# define basic block of FISTA-Net
class BasicBlock(nn.Module):
    """docstring for BasicBlock"""

    def __init__(self, features=32):
        super(BasicBlock, self).__init__()
        self.Sp = nn.Softplus()

    def forward(self, x, PhiTPhi, PhiTb, lambda_step, soft_thr):
        # CIKK: iteráció kék oldala - gradient descent module --\/
        # naive gradient descent update
        x = x.squeeze(1) - self.Sp(lambda_step) * (torch.bmm(PhiTPhi.squeeze(1), x.squeeze(1)) - PhiTb.squeeze(1))
        x = torch.unsqueeze(x, 1)

        # CIKK: minden, ami ez alatt van a narancssárga rész - proximal mapping module --\/
        # soft-thresholding block
        x_st = torch.mul(torch.sign(x), F.relu(torch.abs(x) - self.Sp(soft_thr)))

        # prediction output (skip connection); non-negative output
        x_pred = x + x_st

        # compute symmetry loss
        symloss = torch.Tensor([0])

        return x_pred, symloss, x_st

    
class FISTANet(nn.Module):
    def __init__(self, LayerNo, featureNo):
        super(FISTANet, self).__init__()
        self.LayerNo = LayerNo
        onelayer = []

        self.bb = BasicBlock(features=featureNo)
        for i in range(LayerNo):
            onelayer.append(self.bb)

        self.fcs = nn.ModuleList(onelayer)
        self.fcs.apply(initialize_weights)

        # thresholding value
        self.w_theta = nn.Parameter(torch.Tensor([-0.5]))
        self.b_theta = nn.Parameter(torch.Tensor([-2]))
        # gradient step
        self.w_mu = nn.Parameter(torch.Tensor([-0.2]))
        self.b_mu = nn.Parameter(torch.Tensor([0.1]))
        # two-step update weight
        self.w_rho = nn.Parameter(torch.Tensor([0.5]))
        self.b_rho = nn.Parameter(torch.Tensor([0]))

        self.Sp = nn.Softplus()

    def forward(self, x0, b, Phi):
        """
        Phi   : system matrix; default dim 2500 x 100;
        b     : measured signal vector; default dim 2500 x 1;
        x0    : initialized x; default dim 100 x 1.
        """
        # convert data format from (batch_size, channel, vector_row, vector_col) to (vector_row, batch_size)
        # NEW: in our case it is (batch_size, row, col)
        # b = torch.squeeze(b, 1)
        PhiTPhi = torch.bmm(Phi.permute(0, 2, 1), Phi)
        PhiTb = torch.bmm(Phi.permute(0, 2, 1), b)
        PhiTPhi = torch.unsqueeze(PhiTPhi, 1)
        PhiTb = torch.unsqueeze(PhiTb, 1)
        x0 = torch.unsqueeze(x0, 1)

        # initialize the result
        xold = x0
        y = xold
        layers_sym = []     # for computing symmetric loss
        layers_st = []      # for computing sparsity constraint
        xnews = []          # iteration result
        xnews.append(xold)

        for i in range(self.LayerNo):
            # CIKK: (15) --\/
            theta_ = self.w_theta * i + self.b_theta
            mu_ = self.w_mu * i + self.b_mu
            # CIKK: (8a) + (8b) - nagy iteráció rész --\/
            [xnew, layer_sym, layer_st] = self.fcs[i](y, PhiTPhi, PhiTb, mu_, theta_)            
            # CIKK: - rho update rész --\/
            rho_ = (self.Sp(self.w_rho * i + self.b_rho) - self.Sp(self.b_rho)) / self.Sp(self.w_rho * i + self.b_rho)
            # CIKK: (8c) - következő réteg bemenetének számolása --\/
            y = xnew + rho_ * (xnew - xold)  # two-step update
            xnews.append(xnew)   # iteration result
            layers_st.append(layer_st)
            layers_sym.append(layer_sym)

        xnew = xnew.squeeze(1)
  
        return xnew, layers_sym, layers_st
