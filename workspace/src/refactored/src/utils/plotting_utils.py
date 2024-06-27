import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from os.path import join as pjoin
import mlflow


def plot_est_comp(x_in, x0_pred, pred, target, bpdn_est, dictionary, context, epoch, batch_idx):
    fig, axs = plt.subplots(3, 1, num=1, clear=True)
    fig.set_figheight(900/plt.rcParams['figure.dpi'])  
    fig.set_figwidth(600/plt.rcParams['figure.dpi'])
    loss = nn.MSELoss()
    for ai, i in enumerate([0, 500, 950]):
        axs[ai].plot(x_in[i, :, :].cpu().squeeze().detach(),
                    label=f'INPUT (MSE: {loss(x_in[i, :, :].cpu().squeeze().detach(), target[i, :, :].cpu().squeeze().detach())})', linewidth=0.5)
        axs[ai].plot(target[i, :, :].cpu().squeeze().detach(),
                    label=f'TARGET (MSE: {loss(target[i, :, :].cpu().squeeze().detach(), target[i, :, :].cpu().squeeze().detach())})', linewidth=0.5)
        axs[ai].plot(x_in[i, :, :].cpu().squeeze().detach()-dictionary@bpdn_est[8000+i, :], '--',
                    label=f'BPDN (MSE: {loss(x_in[i, :, :].cpu().squeeze().detach()-dictionary@bpdn_est[8000+i, :], target[i, :, :].cpu().squeeze().detach())})', linewidth=0.5)
        axs[ai].plot(x0_pred[i, :, :].cpu().squeeze().detach(), color='cyan', #linestyle=(0,(1,10)),
                    label=f'INITIAL (MSE: {loss(x0_pred[i, :, :].cpu().squeeze().detach(), target[i, :, :].cpu().squeeze().detach())})', linewidth=0.5)#linewidth=2.5)
        axs[ai].plot(pred[i, :, :].cpu().squeeze().detach(), #linestyle=(0,(1,10)),
                    label=f'FISTA-Net (MSE: {loss(pred[i, :, :].cpu().squeeze().detach(), target[i, :, :].cpu().squeeze().detach())})', linewidth=0.5)#linewidth=2.5)
        axs[ai].legend(prop={'size': 6})
    mlflow.log_figure(fig, f'plots/est-comp/{context}_ep-{epoch}_batch-{batch_idx}.png')


def plot_alpha_comp(x_0, xnew, x_bpdn, context, epoch, batch_idx):
    fig, axs = plt.subplots(3, 1, num=1, clear=True)
    fig.set_figheight(900/plt.rcParams['figure.dpi'])
    fig.set_figwidth(600/plt.rcParams['figure.dpi'])
    for ai, i in enumerate([0, 500, 950]):
        axs[ai].plot(np.linspace(0, 1, x_0[i, :, :].cpu().squeeze().detach().shape[0]), x_0[i, :, :].cpu().squeeze().detach(), label=f'INITIAL (ZEROS: {torch.sum(x_0[i, :, :].abs().cpu().squeeze().detach()>1e-3)} / {x_0[i, :, :].cpu().squeeze().detach().shape[0]} | L1: {torch.mean(torch.abs(x_0[i, :, :]))})', linewidth=0.5)
        axs[ai].plot(np.linspace(0, 1, x_bpdn[i, :, :].cpu().squeeze().detach().shape[0]), x_bpdn[i, :, :].cpu().squeeze().detach(), label=f'BPDN (ZEROS: {torch.sum(x_bpdn[i, :, :].abs().cpu().squeeze().detach()>1e-3)} / {x_bpdn[i, :, :].cpu().squeeze().detach().shape[0]} | L1: {torch.mean(torch.abs(x_bpdn[i, :, :]))})', linewidth=0.5)
        axs[ai].plot(np.linspace(0, 1, xnew[i, :, :].cpu().squeeze().detach().shape[0]), xnew[i, :, :].cpu().squeeze().detach(), 'r', label=f'FISTA-Net (ZEROS: {torch.sum(xnew[i, :, :].abs().cpu().squeeze().detach()>1e-3)} / {xnew[i, :, :].cpu().squeeze().detach().shape[0]} | L1: {torch.mean(torch.abs(xnew[i, :, :]))})', linewidth=0.5)
        axs[ai].legend(prop={'size': 6})
    mlflow.log_figure(fig, f'plots/alpha-comp/{context}_ep-{epoch}_batch-{batch_idx}.png')
