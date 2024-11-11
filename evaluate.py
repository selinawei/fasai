import os
import sys
import cv2
import yaml
import imageio
import random
import numpy as np
from argparse import ArgumentParser
from collections import OrderedDict
from easydict import EasyDict as edict
import torch
import lpips

from arguments import Config
from utils.system import safe_state,set_seed
from utils.tensor import dcn_warp,minmax_norm
import utils.general
from model import FEASAI

class Eval:
    def __init__(self,opt:Config,content):
        assert content in ['simple_align','complex_align','all']
        self.opt = opt
        self.content = content
        self.w,self.h = 346,260
        self.roi_x,self.roi_y = 256,256
        # data
        content = ['simple_align','complex_align'] if content == "all" else [content]
        self.data_list = []
        for each in content:
            data_path = os.path.abspath(f'{opt.data_path}/{each}/Test')
            f_names = sorted(os.listdir(f'{data_path}'))
            for f_name in f_names:
                abs_f_name = f"{data_path}/{f_name}"
                self.data_list.append(np.load(abs_f_name,allow_pickle=True))
        # save_dir
        self.results_dir = os.path.abspath(f'./results/{opt.exp_name}')
        assert os.path.exists(f"{self.results_dir}")
        # net
        rank = 0
        torch.cuda.set_device(rank) # default: cuda:0
        self.net = FEASAI(opt).cuda()
        model_dir = f"{self.results_dir}/model"
        ddp_checkpoint = torch.load(f"{model_dir}/{sorted(os.listdir(model_dir))[-1]}",map_location=f"cuda:{rank}")['net']
        checkpoint = OrderedDict()
        for k,v in ddp_checkpoint.items(): checkpoint[k[7:]] = v # remove 'module.'
        self.net.load_state_dict(checkpoint)
        
    @torch.no_grad()
    def eval_imaging(self,eps=1e-3):
        self.net.eval()
        self.lpips_fn = lpips.LPIPS(net='alex').cuda()
        total_acc_recoder = edict(psnr=0.,ssim=0.,lpips=0.)
        total_pred_recoder = edict(psnr=0.,ssim=0.,lpips=0.)
        total_counter = 0
        for i,data in enumerate(self.data_list):
            acc_recoder = edict(psnr=0.,ssim=0.,lpips=0.)
            pred_recoder = edict(psnr=0.,ssim=0.,lpips=0.)
            save_path = f"{self.results_dir}/eval/imaging/{self.content}/{i:04d}/"
            contents = ['acc','pred','gt']
            for content in contents: os.makedirs(f"{save_path}/{content}",exist_ok=True)
            clip_x,clip_y = (self.w-self.roi_x)//2,(self.h-self.roi_y)//2
            voxelgrid = torch.FloatTensor(data['voxelgrid'])[:,clip_y:clip_y+self.roi_y,clip_x:clip_x+self.roi_x].unsqueeze(dim=0).cuda() # [1,ts,H,W]
            _,ts,h,w = voxelgrid.shape
            time = torch.FloatTensor(data['time']).reshape(1,ts,1,1).cuda() # [ts]
            fx,v = torch.tensor(data['fx']).float().reshape(1,1,1,1).cuda(),torch.tensor(data['v']).float().reshape(1,1,1,1).cuda()
            ref_ts = torch.FloatTensor(data['ref_t']).cuda()
            gt_frames = (torch.FloatTensor(data['gt'])/255.0).cuda()
            depth = self.net.depthnet.forward(voxelgrid) # [bs,ts,H,W]
            flow = fx*v/(depth.clone()+eps)
            for idx in range(len(ref_ts)):   
                gt = (gt_frames[idx])[clip_y:clip_y+self.roi_y,clip_x:clip_x+self.roi_x].reshape(1,1,h,w)
                ref_t = (ref_ts[idx]).reshape(1,1,1,1)
                shift_ref = flow*(time-ref_t)
                refocus_voxelgrid_ref_t = dcn_warp(voxelgrid,-shift_ref) # [1,ts,H,W]
                acc_frame_ref_t = 0.6*minmax_norm(torch.mean(refocus_voxelgrid_ref_t,dim=1,keepdim=True)) # [1,1,H,W]
                pred_frame_ref_t = self.net.framenet(refocus_voxelgrid_ref_t) # [1,1,H,W]
                acc_recoder.psnr += utils.general.psnr(acc_frame_ref_t,gt)
                acc_recoder.ssim += utils.general.ssim(acc_frame_ref_t,gt)
                acc_recoder.lpips += self.lpips_fn(acc_frame_ref_t,gt).item()
                pred_recoder.psnr += utils.general.psnr(pred_frame_ref_t,gt)
                pred_recoder.ssim += utils.general.ssim(pred_frame_ref_t,gt)
                pred_recoder.lpips += self.lpips_fn(pred_frame_ref_t,gt).item()
                cv2.imwrite(f'{save_path}/acc/{idx:04d}.png',(acc_frame_ref_t*255/acc_frame_ref_t.max()).cpu().squeeze().numpy())
                cv2.imwrite(f'{save_path}/pred/{idx:04d}.png',255*pred_frame_ref_t.cpu().squeeze().numpy())
                cv2.imwrite(f'{save_path}/gt/{idx:04d}.png',255*gt.cpu().squeeze().numpy())
            print(f'[val] seq_{i:04d}: acc  | psnr:{acc_recoder.psnr/len(ref_ts):.4f} ssim:{acc_recoder.ssim/len(ref_ts):.4f} lpips:{acc_recoder.lpips/len(ref_ts):.4f}')
            print(f'[val] seq_{i:04d}: pred | psnr:{pred_recoder.psnr/len(ref_ts):.4f} ssim:{pred_recoder.ssim/len(ref_ts):.4f} lpips:{pred_recoder.lpips/len(ref_ts):.4f}')
            for key in acc_recoder.keys(): total_acc_recoder[key] += acc_recoder[key]
            for key in pred_recoder.keys(): total_pred_recoder[key] += pred_recoder[key]
            total_counter += len(ref_ts)
        print(f'[val] seq_total: acc  | psnr:{total_acc_recoder.psnr/total_counter:.4f} '+
                f'ssim:{total_acc_recoder.ssim/total_counter:.4f} lpips:{total_acc_recoder.lpips/total_counter:.4f}')
        print(f'[val] seq_total: pred | psnr:{total_pred_recoder.psnr/total_counter:.4f} '+
                f'ssim:{total_pred_recoder.ssim/total_counter:.4f} lpips:{total_pred_recoder.lpips/total_counter:.4f}')

    @torch.no_grad()
    def eval_depth(self,bounds=2.5):
        self.net.eval()
        for i,data in enumerate(self.data_list):
            save_path = f"{self.results_dir}/eval/depth/{self.content}/{i:04d}/"
            os.makedirs(f"{save_path}",exist_ok=True)
            clip_x,clip_y = (self.w-self.roi_x)//2,(self.h-self.roi_y)//2
            voxelgrid = torch.FloatTensor(data['voxelgrid'])[:,clip_y:clip_y+self.roi_y,clip_x:clip_x+self.roi_x].unsqueeze(dim=0).cuda() # [1,ts,H,W]
            depth = self.net.depthnet.forward(voxelgrid).squeeze() # [ts,H,W]
            for idx,each in enumerate(depth):
                each = torch.clamp(255*(each/bounds),0,255).cpu().squeeze().numpy()
                cv2.imwrite(f'{save_path}/{idx:04d}.png',each)
                
if __name__ == '__main__': 
    parser = ArgumentParser(description="Training script parameters")
    config = Config(parser)
    args = parser.parse_args(sys.argv[1:])
    config.extract(args)
    safe_state(config.quiet)
    set_seed(config.seed)
    torch.autograd.set_detect_anomaly(config.detect_anomaly)
    evalor = Eval(config,content="all")
    evalor.eval_imaging()
    evalor.eval_depth()

# @torch.no_grad()
# def gener_video(net:GESAI,dataset:GESAIDataset,results_dir:str,fps=180,bs=1,eps=1e-3,bounds=2.):
#     net = net.eval()
#     dataloader = DataLoader(dataset,batch_size=bs)
#     with tqdm(total=len(dataloader)) as pbar:
#         pbar.set_description_str(f'[Val]')
#         for i,var in enumerate(dataloader):
#             var = edict(utils.move_to_device(var))
#             # unpack var
#             voxelgrid:Tensor = var.voxelgrid
#             bs,ts,h,w = voxelgrid.shape
#             time:Tensor = var.time.reshape(bs,ts,1,1)
#             fx:Tensor = var.fx.reshape(bs,1,1,1)
#             v:Tensor = var.v.reshape(bs,1,1,1)
#             # depth
#             depth = net.depthnet.forward(voxelgrid) # [1,ts,H,W]
#             # flow & shift
#             flow = fx*v/(depth.clone()+eps)
#             each_time = time.squeeze() # ts
#             ref_times = torch.linspace(start=each_time[0],end=each_time[-1],steps=fps).float().cuda() # [fps]
#             acc_list,pred_list,depth_list = [],[],[]
#             for ref_t in ref_times:
#                 ref_t = ref_t.reshape(1,1,1,1)
#                 shift_ref = flow*(time-ref_t)
#                 refocus_depth_ref = utils.dcn_warp(depth,-shift_ref) # [1,ts,H,W]
#                 depth_ref = utils.nonzero_mean(refocus_depth_ref,dim=1,keepdim=True) # [1,1,H,W]
#                 refocus_voxelgrid_ref = utils.dcn_warp(voxelgrid,-shift_ref) # [1,ts,H,W]
#                 acc_frame_ref = utils.minmax_norm(torch.mean(refocus_voxelgrid_ref,dim=1,keepdim=True)) # [1,1,H,W]
#                 pred_frame_ref = net.framenet(refocus_voxelgrid_ref) # [1,1,H,W]
#                 acc_save,pred_save = to_pil_image(acc_frame_ref[0]),to_pil_image(pred_frame_ref[0])
#                 depth_save = to_pil_image(depth_ref[0][0]/bounds)
#                 acc_list.append(acc_save)
#                 pred_list.append(pred_save)
#                 depth_list.append(depth_save)
#             os.makedirs(f"{results_dir}/video",exist_ok=True)
#             imageio.mimsave(f"{results_dir}/video/{i:04d}_acc.mp4",acc_list,"mp4",fps=30)
#             imageio.mimsave(f"{results_dir}/video/{i:04d}_pred.mp4",pred_list,"mp4",fps=30)
#             imageio.mimsave(f"{results_dir}/video/{i:04d}_depth.mp4",depth_list,"mp4",fps=30)
#             pbar.update(1)




