from loader import DataSplit
import numpy as np
from helpers import show_image_matrix
import torch
from M5FISTANet import FISTANet
import argparse
from solver import Solver
import os
from os.path import dirname, join as pjoin
from metric import compute_measure
from torchsummary import summary
from time import sleep

# =============================================================================
# Load dataset
# some parameters of the dataset
nr_diff_imgs = 10000
test_split = 0.1
val_split = 0.1

fista_net_mode = 1      # 0, test mode; 1, train mode.

batch_size = 10   # 10
LayerNo = 4
FeatureNo = 16

num_epochs = 50
test_epoch = 10
start_epoch = 25

lr_dec_after = 10
lr_dec_every = 5

# load input and target dataset; test_loader, train_loader, val_loader
root_dir = "../../zenodo_data_interface/data/figureSdata"
data_dir = "./data"

padding = 21
Nx = 64

# print('main run now...')

# =============================================================================
#
# Get 100 batch of test data and validate data
if __name__ == '__main__':
    train_loader, val_loader, test_loader, snr_nr = DataSplit(root_dir, nr_diff_imgs,
                                                              batch_size=batch_size,
                                                              validation_split=val_split,
                                                              test_split=test_split)

    for i, (y_v, images_v) in enumerate(test_loader):
        if i == 0:
            test_images = images_v
            test_data_1 = y_v[0]
            test_data_2 = y_v[1]
        elif i == 2:
            break
        else:
            test_images = torch.cat((test_images, images_v), axis=0)
            test_data_1 = torch.cat((test_data_1, y_v[0]), axis=0)
            test_data_2 = torch.cat((test_data_2, y_v[1]), axis=0)

    # add channel axis; torch tensor format (batch_size, channel, width, height
    test_images = torch.unsqueeze(test_images, 1)  # torch.Size([128, 1, 64, 64])
    test_data_1 = torch.unsqueeze(test_data_1, 1)  # torch.Size([128, 1, 64, 256])
    # test_data_1 = torch.unsqueeze(test_data_1, 3)  # torch.Size([128, 1, 104, 1])
    test_data_2 = torch.unsqueeze(test_data_2, 1)  # torch.Size([128, 1, 64, 256])
    # test_data_2 = torch.unsqueeze(test_data_2, 3)  # torch.Size([128, 1, 104, 1])

    print("Shape of test dataset: {}".format(test_images.shape))

    # =============================================================================
    # Model 5
    # FISTA-Net: Proposed method

    print('===========================================')
    print('FISTA-Net...')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(device)
    device = 'cuda'

    # load sensitivity matrix of EIT; load mask matrix, dim from 3228 to 4096
    # K matrix beolvasasa
    # CIKK: A_y matrix from (1), later just A (16) --\/ (K kernel matrix nálunk)
    # np.loadtxt(pjoin(data_dir, "Jmat.csv"), delimiter=",", dtype=float)
    J_ndt = np.loadtxt(pjoin(data_dir, "kernel_abel_matrix.csv"), delimiter=",", dtype=float)
    J_ndt = torch.from_numpy(J_ndt[:64,:])
    J_ndt = J_ndt.clone().detach().to(device=device)
    #J_ndt = torch.tensor(J_ndt, dtype=torch.float32, device=device)

    # row and column difference matrix for TV
    # TODO: [51] cikkből Laplacian difference matrix - egységmátrixra (később esetleg curvelet)
    # CIKK: L Laplacian matrix (16) --\/
    DMts = [] #np.eye(64)  # np.loadtxt(pjoin(data_dir, "DM.csv"), delimiter=",", dtype=float)
    #DMts = torch.from_numpy(DM)
    #DMts = DMts.clone().detach().to(device=device) #
    #DMts = torch.tensor(DMts, dtype=torch.float32, device=device)

    # CIKK: kör maszkolása a négyzet mátrixból --\/
    mask = MatMask(64)
    mask = torch.from_numpy(mask)
    mask = mask.clone().detach()  # torch.tensor(mask, dtype=torch.float32, device=device)

    fista_net = FISTANet(LayerNo, FeatureNo, J_ndt, DMts, mask)
    fista_net = fista_net.to(device)
    
    summary(fista_net, input_size=(1, 64, 298), device=str(device))

    sleep(1000000)

    print('Total number of parameters fista net:',
          sum(p.numel() for p in fista_net.parameters()))

    # define arguments of fista_net
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='FISTANet')
    parser.add_argument('--num_epochs', type=int, default=num_epochs)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--data_dir', type=str, default=data_dir)
    parser.add_argument('--save_path', type=str, default='./models/FISTANet/')
    parser.add_argument('--start_epoch', type=int, default=start_epoch)
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--device', default=device)
    parser.add_argument('--log_interval', type=int, default=20)  # 20
    parser.add_argument('--test_epoch', type=int, default=test_epoch)

    parser.add_argument('--lr_dec_after', type=int, default=lr_dec_after)
    parser.add_argument('--lr_dec_every', type=int, default=lr_dec_every)
    # NDT arguments
    parser.add_argument('--Nx', type=int, default=Nx)
    parser.add_argument('--padding', type=int, default=padding)
    args = parser.parse_args()

    solver = Solver(fista_net, train_loader, val_loader, snr_nr, batch_size, args, (test_data_1, test_data_2))
    
    if args.start_epoch > 0:
        solver.load_model(start_epoch)

    if fista_net_mode == 1:
        solver.train()
        fista_net_test = solver.test()
    else:
        fista_net_test = solver.test()

    dir_name = "./figures"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print('Create path : {}'.format(dir_name))

    fig_name = dir_name + '/fista_net_' + str(args.test_epoch) + 'epoch.png'
    fista_net_test = fista_net_test.cpu().double()[:, :, :args.Nx, args.padding:-args.padding]
    test_images = test_images[:, :, :args.Nx, args.padding:-args.padding]
    results = [test_images, fista_net_test]
    # Evalute reconstructed images with PSNR, SSIM, RMSE.
    p_fista, s_fista, m_fista = compute_measure(test_images, fista_net_test, 1)
    print('PSNR: {:.5f}\t SSIM: {:.5f} \t RMSE: {:.5f}'.format(p_fista,
                                                               s_fista,
                                                               m_fista))
    titles = ['truth', 'fista_net']
    show_image_matrix(fig_name, results, titles=titles, indices=slice(0, 15))
