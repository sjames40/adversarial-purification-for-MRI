from pathlib import Path
from models import utils as mutils
from sde_lib import VESDE
from sampling import (ReverseDiffusionPredictor,
                      LangevinCorrector,
                      get_pc_fouriercs_RI_coil_SENSE)
from models import ncsnpp
import time
from utils import fft2_m, ifft2_m, get_mask, get_data_scaler, get_data_inverse_scaler, restore_checkpoint, \
    normalize_complex, root_sum_of_squares, lambda_schedule_const, lambda_schedule_linear
import torch
import torch.nn as nn
from utils import fft2, ifft2, clear, fft2_m, ifft2_m, root_sum_of_squares
import torch.fft as fft
import numpy as np
from models.ema import ExponentialMovingAverage
import matplotlib.pyplot as plt
import importlib
import argparse
#import sigpy.mri as mr
import os
import two_channel_dataset


device = torch.device("cuda:0")


for test_direct,test_target,test_smap,test_mask,mask_2,A_I,test_kspace in two_channel_dataset.test_loader:
    k_np =test_kspace
    A_k_ref = k_np[:,:, 0, :, :] + 1j * k_np[:,:, 1, :, :]
    sense_maps_ref = test_smap[:,:, 0, :, :] + 1j *test_smap[:,:, 1, :, :]




def fft_with_shifts(img):
    return fft.fftshift(fft.fft2(fft.ifftshift(img)))

def ifft_with_shifts(ksp):
    return fft.fftshift(fft.ifft2(fft.ifftshift(ksp)))

def ksp_and_mps_to_gt(ksp, mps):
    gt = mps.conj() * ifft_with_shifts(ksp)
    gt = torch.sum(gt, axis=0)
    return gt

def mps_and_gt_to_ksp(mps, gt):
    ksp = fft_with_shifts(mps * gt)
    return ksp




ksp1 = A_k_ref[0]
mps1 = sense_maps_ref[0]
A_I_image = A_I[:,:, 0, :, :] + 1j * A_I[:,:, 1, :, :]
kps2 = ksp1.unsqueeze(0).to(device)
mps2 =mps1.unsqueeze(0).to(device)
mask_2 = mask_2.to(device)
undersample_kspace = kps2*mask_2
under_img2= ifft2_m(undersample_kspace)
under_img3 = mps2.conj() * under_img2
under_img3 = torch.sum(under_img3[0], axis=0)
gt1 = ksp_and_mps_to_gt(ksp1, mps1)
gt1 = gt1/torch.max(torch.abs(gt1))


def data_fidelity(mask, x, Fy):
    x = ifft2(fft2(x) * (1. - mask) + Fy)
    #x_mean = ifft2(fft2(x_mean) * (1. - mask) + Fy)
    return x


N = 200
m = 1
fname = 1

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



from math import log10, sqrt
def PSNR2(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 1.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

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



from models import networks



def Recon(cg_iter, smap, mask, input):
    output_CG = input
    for i in range(cg_iter):
        output_NN = netG(output_CG)
        output_CG = CG(output_NN, tol=0.0001, L=1, smap=smap, mask=mask, alised_image=input)
    return output_CG


blockIter = 6
pgd_steps = 30
pgd_epsilon = 0.0039
pgd_epsilon2 = 0.0013
LossLambda = 1



from models.didn import DIDN
from models import networks



aim = '/mnt/DataB/'




netG = torch.load(os.path.join(aim,'Newproject','DIDN_global_model_142iteration_3res_6iter.pt')).to(device)



epsilons = [2/255]




device = torch.device("cuda:0")



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


test_loader = two_channel_dataset.test_loader

test_rmse_total = 0.
test_psnr_total = 0.
test_ssim_total = 0.
adv_test_rmse_total = [0.] * len(epsilons)
adv_test_psnr_total = [0.] * len(epsilons)
adv_test_ssim_total = [0.] * len(epsilons)
def CG(output, tol, L, smap, mask, alised_image):
    return networks.CG.apply(output, tol, L, smap, mask, alised_image)
adv_img_out = []
for i, (test_direct, test_target, test_smap, test_mask,test_mask_2,test_A_I,test_kspace2) in enumerate(test_loader):
    test_input = test_direct.to(device).float()
    test_smap = test_smap.to(device).float()
    test_mask = test_mask.to(device).float()
    test_label = test_target.to(device).float()
    clean_test_input = test_input
    
    for ii, epsilon in enumerate(epsilons):
        adv_test_input = test_input.clone()
        adv_test_input = PGD(pgd_steps, blockIter, test_smap, test_mask, adv_test_input, test_label, loss_fn, epsilon, epsilon / 3)
        with torch.no_grad():
            output = Recon(6, test_smap, test_mask,adv_test_input)



adv_test_input2 = adv_test_input[:, 0, :, :] + 1j * adv_test_input[:, 1, :, :]


output2 = output[:, 0, :, :] + 1j * output[:, 1, :, :]




print(PSNR2(np.abs(output2.cpu().detach().numpy()),np.abs(gt1.detach().numpy())))




x = output2
x2 = mps2 * x




adv_kspace = fft2_m(x2)*mask_2



pc_fouriercs = get_pc_fouriercs_RI_coil_SENSE(sde,
                                              predictor, corrector,
                                              inverse_scaler,
                                              snr=snr,
                                              n_steps=m,
                                              m_steps=50,
                                              mask=mask_2,
                                              sens=mps2,
                                              lamb_schedule=lamb_schedule,
                                              probability_flow=probability_flow,
                                              continuous=config.training.continuous,
                                              denoise=True)

print(f'Beginning inference')
tic = time.time()
#x = pc_fouriercs(score_model.to(device), scaler(under_img).to(device), y=under_kspace.to(device))
x = pc_fouriercs(score_model.to(device), scaler(x2).to(device), y=adv_kspace.to(device))
toc = time.time() - tic
print(f'Time took for recon: {toc} secs.')


x1 = mps2.conj() * x
recon_image = torch.sum(x1[0], axis=0)


print(PSNR2(np.abs(recon_image.cpu().detach().numpy()),np.abs(gt1.detach().numpy())))



recon_real = np.real(recon_image.cpu().detach().numpy())
recon_complex =np.imag(recon_image.cpu().detach().numpy())
recon_two_ch = np.stack((recon_real, recon_complex), axis=0)


recon_two_ch = torch.tensor(recon_two_ch).to(device)



recon_two_ch2 = recon_two_ch.unsqueeze(0) 


def Recon2(cg_iter, smap, mask, input):
    output_CG = input
    for i in range(cg_iter):
        output_NN = netG(output_CG)
        output_CG = CG(output_NN, tol=0.0001, L=3, smap=smap, mask=mask, alised_image=input)
    return output_CG



test_mask2 = torch.ones_like(test_mask)



with torch.no_grad():
    output_after = Recon2(6, test_smap, test_mask2,recon_two_ch2 )



output_after2 = output_after[:, 0, :, :] + 1j * output_after[:, 1, :, :]






