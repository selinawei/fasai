import os
import sys
import cv2
import yaml
import imageio
import random
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from easydict import EasyDict as edict
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from argparse import ArgumentParser,Namespace
from torchvision.transforms.functional import to_pil_image

from arguments import Config
from datareader import PackedData
from datareader.dataset import DatasetFactory
from utils.system import safe_state,set_seed
from utils.tensor import uniform_norm,dcn_warp,minmax_norm
from model import GESAI

@torch.no_grad()
def compare(net:GESAI,dataset,results_dir:str,bs=1,eps=1e-3,depths=[0.6,0.8,1.0]):
    net = net.eval()
    dataloader = DataLoader(dataset,batch_size=bs)
    with tqdm(total=len(dataloader)) as pbar:
        pbar.set_description_str(f'[Val]')
        for i,var in enumerate(dataloader):
            var:PackedData = var.cuda()
            # unpack var
            voxelgrid:Tensor = var.voxelgrid
            bs,ts = voxelgrid.shape[:2]
            time:Tensor = var.time.reshape(bs,ts,1,1)
            fx:Tensor = var.fx.reshape(bs,1,1,1)
            v:Tensor = var.v.reshape(bs,1,1,1)
            flows = []
            # depth
            depth = net.depthnet.forward(uniform_norm(voxelgrid)) # [1,ts,H,W]
            # flow & shift
            flow = fx*v/(depth.clone()+eps)
            # get all flow
            flows.append(flow)
            for d in depths: flows.append(fx*v/d*torch.ones_like(flow))
            # ref_t
            each_time = time.squeeze() # ts
            ref_t = ((each_time[0]+each_time[-1])/2.).float().cuda().reshape(1,1,1,1)
            for idx,each_flow in enumerate(flows):
                shift_ref = each_flow*(time-ref_t)
                refocus_voxelgrid_ref = dcn_warp(voxelgrid,-shift_ref) # [1,ts,H,W]
                acc_frame_ref = minmax_norm(torch.mean(refocus_voxelgrid_ref,dim=1,keepdim=True)) # [1,1,H,W]
                pred_frame_ref = net.framenet(refocus_voxelgrid_ref.unsqueeze(dim=2)) # [1,1,H,W]
                acc_save,pred_save = to_pil_image(acc_frame_ref[0]),to_pil_image(pred_frame_ref[0])
                suffix = 'all_in_focus' if idx == 0 else str(round(depths[idx-1],2))
                os.makedirs(f"{results_dir}/compare",exist_ok=True)
                acc_save.save(f"{results_dir}/compare/{i:04d}_acc_{suffix}.png")
                pred_save.save(f"{results_dir}/compare/{i:04d}_pred_{suffix}.png")
                pbar.update(1)

if __name__ == '__main__':
    parser = ArgumentParser(description="Training script parameters")
    config = Config(parser)
    args = parser.parse_args(sys.argv[1:])
    config.extract(args)
    safe_state(config.quiet)
    set_seed(config.seed)
    torch.autograd.set_detect_anomaly(config.detect_anomaly)
    dataset = DatasetFactory().get(config.dataset,config,"Test")
    # save dir
    results_dir = os.path.abspath(f'./results/{config.exp_name}')
    assert os.path.exists(f"{results_dir}")
    # net
    rank = 0
    torch.cuda.set_device(rank) # default: cuda:0
    net = GESAI(config).cuda()
    net = net.train()
    model_dir = f"{results_dir}/model"
    ddp_checkpoint = torch.load(f"{model_dir}/{sorted(os.listdir(model_dir))[-1]}",map_location=f"cuda:{rank}")['net']
    checkpoint = OrderedDict()
    for k,v in ddp_checkpoint.items(): checkpoint[k[7:]] = v # remove 'module.'
    net.load_state_dict(checkpoint)
    # eval
    compare(net,dataset,results_dir)