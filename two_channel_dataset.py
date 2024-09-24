import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Sequence, Tuple, Union
import math
import time
from torch.utils.data.dataset import Dataset
from torch.nn import init
import math
import h5py
import sys
import os
import torch
from PIL import Image
import random
#import scipy.io as sp
from util.util import generate_mask_alpha, generate_mask_beta
import scipy.ndimage
from util.util import fft2, ifft2, cplx_to_tensor, complex_conj, complex_matmul, absolute
import h5py
import glob
from models2 import networks

def convert_2chan_into_abs_2(img):
    img_real = img[0][0]
    img_imag = img[0][1]
    img_complex = torch.complex(img_real, img_imag)
    return img_complex

def make_data_list(file_path,file_array):
    file_data = []
    for i in range(len(file_array)):
        data_file = file_array[i]
        data_from_file = np.load(os.path.join(file_path,data_file),'r')
        file_data.append(data_from_file)
    return file_data
# first half of the kspace image dataset which has 1400 images
Kspace_data_name3 = '/mnt/DataA/NEW_KSPACE'
kspace_data = []
kspace_array = os.listdir(Kspace_data_name3)
kspace_array = sorted(kspace_array)

kspace_data = []
clean_data1 = []
vali_data1 = []
def make_vdrs_mask(N1,N2,nlines,init_lines,seed=0):
    mask_vdrs=np.zeros((N1,N2),dtype='bool')
    low1=(N2-init_lines)//2
    low2=(N2+init_lines)//2
    mask_vdrs[:,low1:low2]=True
    nlinesout=(nlines-init_lines)//2
    rng = np.random.default_rng(seed)
    t1 = rng.choice(low1-1, size=nlinesout, replace=False)
    t2 = rng.choice(np.arange(low2+1, N2), size=nlinesout, replace=False)
    mask_vdrs[:,t1]=True; mask_vdrs[:,t2]=True
    return mask_vdrs
    

for i in range(415,1000): 
    kspace_file = kspace_array[h]
    kspace_data_from_file = np.load(os.path.join(Kspace_data_name3,kspace_file),'r')
    clean_data1.append(kspace_data_from_file)


for h in range(336,337):
    kspace_file_vali =  kspace_array[h]
    kspace_vali_data_from_file = np.load(os.path.join(Kspace_data_name3,kspace_file_vali),'r')
    if kspace_vali_data_from_file['k_r'].shape[2]<373:
        vali_data1.append(kspace_vali_data_from_file)

mask_data_selet = []
mask_vali = []

mask_data_name = '/mnt/DataA/MRI_sampling/4_accerlation_mask'
mask_array = os.listdir(mask_data_name)
mask_array = sorted(mask_array)
mask_file = mask_array[1081]
mask_from_file = np.load(os.path.join(mask_data_name,mask_file),'r')
for e in range(len(clean_data1)):
    if clean_data1[e]['k_r'].shape[2]> 368:
        mask_data_selet.append(np.pad(mask_from_file, ((0, 0), (2, 2)), 'constant'))
    else:
        mask_data_selet.append(mask_from_file)
for k in range(len(vali_data1)):
    if vali_data1[k]['k_r'].shape[2]> 368:
        mask_vali.append(np.pad(mask_from_file, ((0, 0), (2, 2)), 'constant'))
    else:
        mask_vali.append(mask_from_file)
        


class nyumultidataset(Dataset): # model data loader
    def  __init__(self ,kspace_data,mask_data):
        self.A_paths = kspace_data
        #self.A_paths = sorted(self.A_paths)
        self.A_size = len(self.A_paths)
        self.mask_path = mask_data
        #self.mask_path =sorted(self.mask_path)
        self.nx = 640
        self.ny = 368

    def __getitem__(self, index):
        A_temp = self.A_paths[index]
        s_r = A_temp['s_r']/ 32767.0 
        s_i = A_temp['s_i']/ 32767.0 
        k_r = A_temp['k_r']/ 32767.0
        k_i = A_temp['k_i']/ 32767.0 
        ncoil, nx, ny = s_r.shape
        #mask = self.mask_path[index]
        mask = make_vdrs_mask(nx,ny,np.int32(ny*0.25),np.int32(ny*0.08))
        k_np = np.stack((k_r, k_i), axis=0)
        s_np = np.stack((s_r[:, nx // 2 - 160:nx // 2 + 160, ny // 2 - 160:ny // 2 + 160],
                         s_i[:, nx // 2 - 160:nx // 2 + 160, ny // 2 - 160:ny // 2 + 160]), axis=0)
        mask_2 = torch.tensor(np.repeat(mask[np.newaxis, nx // 2 - 160:nx // 2 + 160, ny // 2 - 160:ny // 2 + 160], 15, axis=0), dtype=torch.float32)
        mask = torch.tensor(np.repeat(mask[np.newaxis, nx // 2 - 160:nx // 2 + 160, ny // 2 - 160:ny // 2 + 160], 2, axis=0), dtype=torch.float32)
        A_k = torch.tensor(k_np, dtype=torch.float32).permute(1, 0, 2, 3)
        #mask_two_channel =mask_two_channel.numpy()
        ##A_I is the ifft of the kspace 
        A_I = ifft2(A_k.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        A_I = A_I[:, :, nx // 2 - 160:nx // 2 + 160, ny // 2 - 160:ny // 2 + 160]
        ##A_s is the sensitive map 
        A_s = torch.tensor(s_np, dtype=torch.float32).permute(1, 0, 2, 3)
        SOS = torch.sum(complex_matmul(A_I, complex_conj(A_s)),dim=0)
        A_I = A_I/torch.max(torch.abs(SOS)[:])
        print(A_I.shape)
        A_k = fft2(A_I.permute(0,2,3,1)).permute(0,3,1,2)
        kreal = A_k
        print(kreal.shape)
        print(mask.shape)
        print('A_s',A_s.shape)
        AT = networks.OPAT2(A_s)
        Iunder = AT(kreal, mask)
        Ireal = AT(kreal, torch.ones_like(mask))
        #return  Iunder, Ireal, A_s, mask
        return  Iunder, Ireal, A_s, mask,mask_2,A_I, A_k
     
       
    def __len__(self):
        return len(self.A_paths)
train_size = 0.9
num_train = 1000
train_clean_paths = clean_data1[:num_train]
mask_data_paths =mask_data_selet[:num_train]
train_dataset = nyumultidataset(train_clean_paths,mask_data_paths)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1,shuffle=False)
test_clean_paths = vali_data1 #1090
mask_test_paths = mask_vali
test_dataset = nyumultidataset(test_clean_paths,mask_test_paths)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,shuffle=False)


    

    

    
    

    

    
    
    
    
