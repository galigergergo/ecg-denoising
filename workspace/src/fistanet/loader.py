# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 13:51:58 2024

DataLoader of ECG data.

@author: Galiger Gergo

"""
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from os.path import join as pjoin
import numpy as np

class ECGDataset(Dataset):
    def _calc_split_params(self, mode, tot_smpl_no, tvt_split):
        smpl_start_i = 0
        split_norm = 0
        res_dct = dict()
        for m in ['train', 'valid', 'test']:
            split_norm += tvt_split[m] / sum(tvt_split.values())
            last_smpl_i = int(np.floor(tot_smpl_no * split_norm))
            if m == mode:
                return smpl_start_i, last_smpl_i - smpl_start_i
            smpl_start_i = last_smpl_i
    
    def __init__(self, mode, data_dir, file_name, tvt_split, transform=None):
        assert mode in ['train', 'valid', 'test']
        self.mode = mode
        
        self.inp_dir = data_dir
        self.file_name = file_name
        self.transform = transform

        # load data file, but only keep necessary part
        data_temp_ = np.load(pjoin(self.inp_dir, self.file_name))
        tot_smpl_no = data_temp_.shape[2]
        smpl_start_i, self.smpl_no = self._calc_split_params(mode, tot_smpl_no, tvt_split)
        self.data_ = data_temp_[:, :, smpl_start_i:smpl_start_i+self.smpl_no]
        
        # transform the input tensor into required formats
        if self.transform:
            input_m = self.transform(input_m)
     
    def __len__(self):
        return self.smpl_no

    def __getitem__(self, idx):
        return self.data_[0, :, idx], self.data_[1, :, idx]


def DataSplit(data_dir, file_name, tvt_split, batch_size=128, transform=None):
    ds_trn = ECGDataset('train', data_dir, file_name, tvt_split, transform)
    ds_val = ECGDataset('valid', data_dir, file_name, tvt_split, transform)
    ds_tst = ECGDataset('test', data_dir, file_name, tvt_split, transform)

    trn_smplr = SubsetRandomSampler(list(range(len(ds_trn))))

    trn_ldr = DataLoader(ds_trn, batch_size=batch_size, sampler=trn_smplr)
    val_ldr = DataLoader(ds_val, batch_size=batch_size)
    tst_ldr = DataLoader(ds_tst, batch_size=batch_size)

    return trn_ldr, val_ldr, tst_ldr
