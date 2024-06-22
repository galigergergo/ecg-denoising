# -*- coding: utf-8 -*-
"""
Created on June 17, 2020

ISTANet(shared network with 4 conv + ReLU) + regularized hyperparameters softplus(w*x + b). 
The Intention is to make gradient step \mu and thresholding value \theta positive and monotonically decrease.

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

        self.conv_D = nn.Conv2d(1, features, (3, 3), stride=1, padding=1)
        self.conv1_forward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv2_forward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv3_forward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv4_forward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)

        self.conv1_backward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv2_backward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv3_backward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv4_backward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv_G = nn.Conv2d(features, 1, (3, 3), stride=1, padding=1)

    def forward(self, x, PhiTPhi, PhiTb, lambda_step, soft_thr, epoch):
        # CIKK: iteráció kék oldala - gradient descent module --\/
        # naive gradient descent update
        
        x = x.squeeze(1) - self.Sp(lambda_step) * (torch.bmm(PhiTPhi.squeeze(1), x.squeeze(1)) - PhiTb.squeeze(1))
        x = torch.unsqueeze(x, 1)

        # CIKK: minden, ami ez alatt van a narancssárga rész - proximal mapping module --\/
        x_input = x

        x_D = self.conv_D(x_input.float())

        x = self.conv1_forward(x_D)
        x = F.relu(x)
        x = self.conv2_forward(x)
        x = F.relu(x)
        x = self.conv3_forward(x)
        x = F.relu(x)
        x_forward = self.conv4_forward(x)

        # soft-thresholding block
        x_st = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.Sp(soft_thr)))

        x = self.conv1_backward(x_st)
        x = F.relu(x)
        x = self.conv2_backward(x)
        x = F.relu(x)
        x = self.conv3_backward(x)
        x = F.relu(x)
        x_backward = self.conv4_backward(x)

        x_G = self.conv_G(x_backward)

        # prediction output (skip connection); non-negative output
        # x_pred = F.relu(x_input + x_G)
        x_pred = x_input + x_G

        # compute symmetry loss
        x = self.conv1_backward(x_forward)
        x = F.relu(x)
        x = self.conv2_backward(x)
        x = F.relu(x)
        x = self.conv3_backward(x)
        x = F.relu(x)
        x_D_est = self.conv4_backward(x)
        symloss = x_D_est - x_D

        return [x_pred, symloss, x_st]
    
# def test_plot_alpha(x_0, save_path, file_name):
#     fig, axs = plt.subplots(3, 1)
#     fig.set_figheight(1800/plt.rcParams['figure.dpi'])
#     fig.set_figwidth(1000/plt.rcParams['figure.dpi'])
#     axs[0].plot(np.linspace(0, 1, x_0[0, :, :].cpu().squeeze().detach().shape[0]), x_0[0, :, :].cpu().squeeze().detach())
#     axs[1].plot(np.linspace(0, 1, x_0[0, :, :].cpu().squeeze().detach().shape[0]), x_0[500, :, :].cpu().squeeze().detach())
#     axs[2].plot(np.linspace(0, 1, x_0[0, :, :].cpu().squeeze().detach().shape[0]), x_0[950, :, :].cpu().squeeze().detach())
#     if not os.path.exists(pjoin(save_path, 'plots', 'alpha')):
#         os.makedirs(pjoin(save_path, 'plots', 'alpha'))
#     plt.savefig(pjoin(save_path, 'plots', 'alpha', file_name))
#     plt.clf()
#     plt.close()
#     #pass

def test_plot_alpha(x_0, xnew, save_path, file_name, target):
    fig, axs = plt.subplots(3, 1, num=1, clear=True)
    fig.set_figheight(1500/plt.rcParams['figure.dpi'])
    fig.set_figwidth(1000/plt.rcParams['figure.dpi'])
    for ai, i in enumerate([0, 500, 950]):
        axs[ai].plot(np.linspace(0, 1, x_0[i, :, :].cpu().squeeze().detach().shape[0]), x_0[i, :, :].cpu().squeeze().detach(), label=f'INITIAL (ZEROS: {torch.sum(x_0[i, :, :].abs().cpu().squeeze().detach()>1e-3)} / {x_0[i, :, :].cpu().squeeze().detach().shape[0]} | L1: {torch.mean(torch.abs(x_0[i, :, :]))})', linewidth=0.5)
        axs[ai].plot(np.linspace(0, 1, target[i, :, :].cpu().squeeze().detach().shape[0]), target[i, :, :].cpu().squeeze().detach(), label=f'BPDN (ZEROS: {torch.sum(target[i, :, :].abs().cpu().squeeze().detach()>1e-3)} / {target[i, :, :].cpu().squeeze().detach().shape[0]} | L1: {torch.mean(torch.abs(target[i, :, :]))})', linewidth=0.5)
        axs[ai].plot(np.linspace(0, 1, xnew[i, :, :].cpu().squeeze().detach().shape[0]), xnew[i, :, :].cpu().squeeze().detach(), 'r', label=f'FISTA-Net (ZEROS: {torch.sum(xnew[i, :, :].abs().cpu().squeeze().detach()>1e-3)} / {xnew[i, :, :].cpu().squeeze().detach().shape[0]} | L1: {torch.mean(torch.abs(xnew[i, :, :]))})', linewidth=0.5)
        axs[ai].legend()
    if not os.path.exists(pjoin(save_path, 'plots', 'alpha')):
        os.makedirs(pjoin(save_path, 'plots', 'alpha'))
    plt.savefig(pjoin(save_path, 'plots', 'alpha', file_name))
    # plt.clf()
    # plt.close()
    #pass
    
    
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

    def forward(self, x0, b, Phi, epoch, save_path=None, file_name=None, target=None):
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
            [xnew, layer_sym, layer_st] = self.fcs[i](y, PhiTPhi, PhiTb, mu_, theta_, epoch)            
            # CIKK: - rho update rész --\/
            rho_ = (self.Sp(self.w_rho * i + self.b_rho) - self.Sp(self.b_rho)) / self.Sp(self.w_rho * i + self.b_rho)
            # CIKK: (8c) - következő réteg bemenetének számolása --\/
            y = xnew + rho_ * (xnew - xold)  # two-step update
            xnews.append(xnew)   # iteration result
            layers_st.append(layer_st)
            layers_sym.append(layer_sym)

        xnew = xnew.squeeze(1)
        # xnew[np.abs(xnew) < 1e-6] = 0
        # xnew = torch.div(xnew, 10)
        # pred = b - torch.bmm(Phi, xnew)
        pred = xnew
        
        if file_name != None and not epoch%10:   # validation only
           test_plot_alpha(xold, xnew, save_path, file_name, target)
        
        return [pred, layers_sym, layers_st]
