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

def attack_single_run(x, y, cg_iter,smap,mask, x_init=None):
    # lfinite norm case
    
    t = 2 * torch.rand(x.shape).to(device).detach() - 1
    x_adv = x + eps * torch.ones_like(x).detach() * normalize(t)
    # projection or clamping

    x_adv = torch.clamp(x_adv, min=-1, max=1)
    x_best = x_adv.clone()
    x_best_adv = x_adv.clone()
    loss_steps = torch.zeros([n_iter, x.shape[0]]).to(device)
    loss_best_steps = torch.zeros([n_iter + 1, x.shape[0]]).to(device)
    acc_steps = torch.zeros_like(loss_best_steps)
    #  define the loss for the reconstruction
    criterion_indiv = mse_loss


    x_adv.requires_grad_()
    grad = torch.zeros_like(x)
    for _ in range(eot_iter):
        #if not self.is_tf_model:
        with torch.enable_grad():
            logits = Recon(cg_iter, smap, mask, x_adv)
            loss_indiv = criterion_indiv(logits, y)
            loss = loss_indiv.sum()

        grad += torch.autograd.grad(loss, [x_adv])[0].detach()

    grad /= float(eot_iter)
    grad_best = grad.clone()

    loss_best = loss_indiv.detach().clone()

    alpha = 2.# if self.norm in ['Linf', 'L2'] else 1. if self.norm in ['L1'] else 2e-2
    step_size = alpha * eps * torch.ones([x.shape[0], *([1] * ndims)]).to(device).detach()
    x_adv_old = x_adv.clone()
    counter = 0
    k = n_iter_2 + 0
    n_fts = math.prod(orig_dim)
    
    counter3 = 0

    loss_best_last_check = loss_best.clone()
    reduced_last_check = torch.ones_like(loss_best)
    n_reduced = 0

    u = torch.arange(x.shape[0], device=device)
    for i in range(n_iter):
        ### gradient step
        with torch.no_grad():
            x_adv = x_adv.detach()
            grad2 = x_adv - x_adv_old
            x_adv_old = x_adv.clone()

            a = 0.75 if i > 0 else 1.0

            # Linf norm
            x_adv_1 = x_adv + step_size * torch.sign(grad)
            x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1,x - eps), x + eps), 0.0, 1.0)
            x_adv_1 = torch.clamp(torch.min(torch.max(x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a),
                    x - eps), x + eps), 0.0, 1.0)

            x_adv = x_adv_1 + 0.

        ### get gradient
        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(eot_iter):
            #if not self.is_tf_model:
                with torch.enable_grad():
                    logits = Recon(cg_iter, smap, mask, x_adv)
                    loss_indiv = criterion_indiv(logits, y)
                    loss = loss_indiv.sum()

                grad += torch.autograd.grad(loss, [x_adv])[0].detach()

        grad /= float(eot_iter)

        pred = logits.detach().max(1)[1] == y
        #acc = torch.min(acc, pred)
        #acc_steps[i + 1] = acc + 0
        #ind_pred = (pred == 0).nonzero().squeeze()
        #x_best_adv[ind_pred] = x_adv[ind_pred] + 0.


        ### check step size
        with torch.no_grad():
            y1 = loss_indiv.detach().clone()
            #print(y1)
            loss_steps[i] = y1 + 0
            #ind = (y1 > loss_best).nonzero().squeeze()
            x_best = x_adv.clone()
            grad_best = grad.clone()
            #print(ind)
            #print(y1.shape)
            loss_best = y1 + 0
            loss_best_steps[i + 1] = loss_best + 0

            counter3 += 1

            if counter3 == k:
                fl_oscillation = check_oscillation(loss_steps, i, k,
                      loss_best, k3=thr_decr)
                fl_reduce_no_impr = (1. - reduced_last_check) * (
                      loss_best_last_check >= loss_best).float()
                fl_oscillation = torch.max(fl_oscillation,
                      fl_reduce_no_impr)
                reduced_last_check = fl_oscillation.clone()
                loss_best_last_check = loss_best.clone()

                if fl_oscillation.sum() > 0:
                    ind_fl_osc = (fl_oscillation > 0).nonzero().squeeze()
                    step_size[ind_fl_osc] /= 2.0
                    n_reduced = fl_oscillation.sum()

                    x_adv[ind_fl_osc] = x_best[ind_fl_osc].clone()
                    grad[ind_fl_osc] = grad_best[ind_fl_osc].clone()

                    k = max(k - size_decr, n_iter_min)
            #print(type(k))
            #print(type(size_decr))
            #print(type(n_iter_min))

    return (x_best, x_best_adv)
    
def Recon(cg_iter, smap, mask, input):
    output_CG = input
    for i in range(cg_iter):
        output_NN = netG(output_CG)
        output_CG = CG(output_NN, tol=0.0001, L=1, smap=smap, mask=mask, alised_image=input)
    return output_CG
    
def normalize(x):
    #if self.norm == 'Linf':
    t = x.abs().view(x.shape[0], -1).max(1)[0]

    return x / (t.view(-1, *([1] * ndims)) + 1e-12)
    
def loss_fn(outputs, labels):
    loss = mse_loss(outputs, labels)
    return loss
    
def check_oscillation(x, j, k, y5, k3=0.75):
    t = torch.zeros(x.shape[1]).to(device)
    for counter5 in range(k):
        t += (x[j - counter5] > x[j - counter5 - 1]).float()

    return (t <= k * k3 * torch.ones_like(t)).float()

n_iter=100
eot_iter=1
n_iter_2 = max(int(0.22 * n_iter), 1)
n_iter_min = max(int(0.06 * n_iter), 1)
size_decr = max(int(0.03 * n_iter), 1)
thr_decr = 0.75
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
    orig_dim = list(test_input.shape[1:])
    ndims = len(orig_dim)
    adv_train_input, x_best_adv = attack_single_run(train_input, train_label, 6,train_smap ,train_mask , x_init=None)
    with torch.no_grad():
         output = Recon(6, test_smap, test_mask,adv_train_input)
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
    

