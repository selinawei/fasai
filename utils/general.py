import torch
from torch import Tensor
import torch.nn as nn
from math import log10
from pytorch_msssim import ssim as base_ssim

def eval_bn(m:nn.Module):
    if type(m) == torch.nn.BatchNorm2d:
        m.eval()
        
def train_bn(m:nn.Module):
    if type(m) == torch.nn.BatchNorm2d:
        m.train()

def summarize_loss(weights,losses)->Tensor:
    assert len(weights) == len(losses)
    loss_all = 0.
    for weight,loss in zip(weights,losses):
        loss_all += weight*loss
    return loss_all

def psnr(img1:Tensor,img2:Tensor):
    assert img1.shape == img2.shape
    return -10*log10(torch.mean((img1-img2)**2))

def ssim(img1:Tensor,img2:Tensor):
    assert img1.shape == img2.shape
    return base_ssim(img1,img2,data_range=1.)
