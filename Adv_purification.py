from pathlib import Path
from models import utils as mutils
from sde_lib import VESDE
from sampling import (ReverseDiffusionPredictor,
                      LangevinCorrector,
                      get_pc_fouriercs_RI_coil_SENSE)
from models import ncsnpp
from models import networks
from models.didn import DIDN
import time
from utils import fft2_m, ifft2_m, get_mask, get_data_scaler, get_data_inverse_scaler, restore_checkpoint, \
    normalize_complex, root_sum_of_squares, lambda_schedule_const, lambda_schedule_linear
import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np
from models.ema import ExponentialMovingAverage
import matplotlib.pyplot as plt
import importlib
import argparse
import sigpy.mri as mr
import os
import global_network_dataset2

device = torch.device("cuda:0")

def PGD(pgd_steps, cg_iter, smap, mask, input, label, crition, eps, alpha, norm='linfty'):
    clamp_fn = l2_clamp if norm == 'l2' else linfty_clamp

    orig_input = input.clone().detach()
    input = input.clone().detach()

    input = clamp_fn(input + torch.normal(0, eps, input.shape).to(device), input, eps)
    input = torch.clamp(input, min=-1, max=1)

    for i in range(pgd_steps):
        input.requires_grad = True
        output = Recon(cg_iter, smap, mask, input)
        loss = crition(output, label)
        loss.backward()
        adv_images = input + alpha * input.grad.sign()
        input = clamp_fn(adv_images, orig_input, eps)
        input = torch.clamp(input, min=-1, max=1).detach()
    return input
    
def Recon(cg_iter, smap, mask, input):
    output_CG = input
    for i in range(cg_iter):
        output_NN = netG(output_CG)
        output_CG = CG(output_NN, tol=0.0001, L=1, smap=smap, mask=mask, alised_image=input)
    return output_CG
    
def linfty_clamp(input, center, epsilon):
    input = torch.clamp(input, min=center-epsilon, max=center+epsilon)
    return input

def l2_clamp(input, center, epsilon):
    delta = (input - center).flatten(1)
    delta_len = torch.linalg.vector_norm(delta, ord=2, dim=1)
    delta_len = delta_len.repeat(delta.shape[1], 1).T
    delta[delta_len > epsilon] = delta[delta_len > epsilon] / delta_len[delta_len > epsilon] * epsilon
    input = center + delta.reshape(input.shape)
    return input
    
def loss_fn(outputs, labels):
    loss = mse_loss(outputs, labels)
    return loss
    
mse_loss = nn.MSELoss().to(device)
N = 150
m = 1
fname = 1
path = '/home/shijunliang/SDF_data/score-MRI-main/samples/'

print('initaializing...')
configs = importlib.import_module(f"configs.ve.fastmri_knee_320_ncsnpp_continuous")
config = configs.get_config()
img_size = config.data.image_size
batch_size = 1

schedule = 'linear'
start_lamb = 1.0
end_lamb = 0.2
m_steps = 50

if schedule == 'const':
    lamb_schedule = lambda_schedule_const(lamb=start_lamb)
elif schedule == 'linear':
    lamb_schedule = lambda_schedule_linear(start_lamb=start_lamb, end_lamb=end_lamb)
else:
    NotImplementedError(f"Given schedule {schedule} not implemented yet!")


ckpt_filename = f"/home/shijunliang/SDF_data/score-MRI-main/weights/checkpoint_95.pth"
sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=N)

config.training.batch_size = batch_size
predictor = ReverseDiffusionPredictor#.to(device)
corrector = LangevinCorrector#.to(device)
probability_flow = False
snr = 0.16



scaler = get_data_scaler(config)
inverse_scaler = get_data_inverse_scaler(config)

# create model and load checkpoint
score_model = mutils.create_model(config).to(device)
ema = ExponentialMovingAverage(score_model.parameters(),
                               decay=config.model.ema_rate)
state = dict(step=0, model=score_model, ema=ema)
state = restore_checkpoint(ckpt_filename, state, device, skip_sigma=True)
ema.copy_to(score_model.parameters())


for train_direct,train_target,train_smap,train_mask,mask_2,A_I,train_kspace in global_network_dataset2.train_loader:
    ksp = (train_kspace[:,:, 0, :, :] + 1j * train_kspace[:,:, 1, :, :]).to(device)
    mps2 = (train_smap[:,:, 0, :, :] + 1j *train_smap[:,:, 1, :, :]).to(device)
    train_input = train_direct.to(device).float()
    train_smap = train_smap.to(device).float()
    train_mask = train_mask.to(device).float()
    train_label = train_target.to(device).float()
    undersample_kspace = ksp*mask_2.to(device)
    mask2 = mask_2.to(device)
    under_img2= ifft2_m(undersample_kspace).to(device)
    adv_test_input = test_input.clone()
    adv_test_input = PGD(pgd_steps, blockIter, train_smap, train_mask, adv_test_input, train_label, loss_fn, epsilon, epsilon / 3)
    with torch.no_grad():
         output = Recon(6, test_smap, test_mask,adv_test_input)
    output = output[:, 0, :, :] + 1j * output[:, 1, :, :].to(device)
    output = mps2 * output
    pc_fouriercs = get_pc_fouriercs_RI_coil_SENSE(sde,
                                              predictor, corrector,
                                              inverse_scaler,
                                              snr=snr,
                                              n_steps=m,
                                              m_steps=50,
                                              mask=mask2,
                                              sens=mps2,
                                              lamb_schedule=lamb_schedule,
                                              probability_flow=probability_flow,
                                              continuous=config.training.continuous,
                                              denoise=True)
    x = pc_fouriercs(score_model.to(device), scaler(output).to(device), y=undersample_kspace .to(device))
    

