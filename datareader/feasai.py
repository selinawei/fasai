import os

import random
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from arguments import Config
from arguments.prepare_data import DatasetConfig
from datareader import PackedData
from utils.system import listdir
from utils.tensor import flip
import torch.nn.functional as F
import torchvision.transforms as transforms

class FEASAIDataset(Dataset):
    def __init__(self,opt:Config,mode:str):
        super().__init__()
        if mode is not None:
            self.opt = opt
            self.base_path = opt.data_path
            self.mode = mode
            self.w,self.h = 346,260
            self.roi_x,self.roi_y = 256,256
            self.data_path = []
            for content in opt.contents:
                if isinstance(content,list): content = "".join(content)
                self.data_path.append(f"{self.base_path}/{content}/{mode}")
            self.prefetch_data()

    def normalize_to_255(self, tensor):
        # 确保 tensor 是浮点数类型，以避免整数除法
        tensor = tensor.float()
        min_val = tensor.min()
        max_val = tensor.max()

        # 防止除以 0，进行归一化并缩放到 [0, 255]
        normalized_tensor = 255 * (tensor - min_val) / (max_val - min_val + 1e-8)

        # 返回为 uint8 类型
        return normalized_tensor.to(torch.uint8)  # 转换为 uint8 类型

    def prefetch_data(self):
        print("Prefetching numpy data ...")
        self.data_list = []
        for data_path in self.data_path:
            f_names = listdir(data_path)
            for f_name in f_names:
                # with np.load(f_name, allow_pickle=True) as data:
                #     self.data_list.append({key: data[key] for key in data})
                self.data_list.append(np.load(f_name,allow_pickle=True))
                
    def __getitem__(self, index):
        
        data = self.data_list[index]
        var = PackedData()
        # unpack var
        var.voxelgrid = torch.FloatTensor(data['voxelgrid']) # [ts,H,W]
        var.time = torch.FloatTensor(data['time']) # [ts]
        var.fx,var.v = torch.tensor(data['fx']).float(),torch.tensor(data['v']).float()
        occ_t = torch.FloatTensor(data['occ_t'])
        occ_aps = torch.FloatTensor(data['occ_aps'])
        gt_t = torch.FloatTensor(data['gt_t'])
        gt = torch.FloatTensor(data['gt'])
        depth_gt = torch.FloatTensor(data['depth'])
        # mean = occ_aps.mean(dim=[1, 2])  # 对 (H, W) 维度求均值，得到每个通道的均值
        # std = occ_aps.std(dim=[1, 2])  # 对 (H, W) 维度求标准差，得到每个通道的标准差
        # normalize = transforms.Normalize(mean=mean, std=std)
        # occ_aps = normalize(occ_aps)
        # mean = gt.mean(dim=[1, 2])  # 对 (H, W) 维度求均值，得到每个通道的均值
        # std = gt.std(dim=[1, 2])  # 对 (H, W) 维度求标准差，得到每个通道的标准差
        # normalize = transforms.Normalize(mean=mean, std=std)
        # gt = normalize(gt)
        # for i in range(occ_aps.shape[0]):
        #     occ_aps[i] = self.normalize_to_255(occ_aps[i])
        # for i in range(gt.shape[0]):
        #     gt[i] = self.normalize_to_255(gt[i])

        # print(len(gt_t))
        # data enhancement
        # ref
        ref_len = len(gt)
        dist = round(ref_len*0.2)
        rand_ref = round(ref_len*0.4)
        var.gt_imgs = gt.clone()
        var.gt,var.other = gt[rand_ref],gt[rand_ref+dist]
        var.gt_t,var.other_t = gt_t[rand_ref],gt_t[rand_ref+dist]
        
        # clip
        clip_x,clip_y = (self.w-self.roi_x)//2,(self.h-self.roi_y)//2
        if self.mode == "Train" and self.opt.rand_clip is True:
            clip_x,clip_y = random.randint(0,self.w-self.roi_x-1),random.randint(0,self.h-self.roi_y-1)
        var.voxelgrid = var.voxelgrid[:,clip_y:clip_y+self.roi_y,clip_x:clip_x+self.roi_x]
        occ_aps = occ_aps[:,clip_y:clip_y+self.roi_y,clip_x:clip_x+self.roi_x]
        depth_gt = depth_gt[:,clip_y:clip_y+self.roi_y,clip_x:clip_x+self.roi_x]
        var.gt_imgs = var.gt_imgs[:,clip_y:clip_y+self.roi_y,clip_x:clip_x+self.roi_x]
        current_length_depth = depth_gt.shape[0]
        current_length = occ_aps.shape[0]
        current_length_gt = var.gt_imgs.shape[0]
        target_length = 27
        if current_length<target_length:
            pad_length = target_length - current_length
            var.occ_aps = F.pad(occ_aps, (0, 0, 0, 0, 0, pad_length), "constant", 0)
            var.occ_t = F.pad(occ_t, (0,  pad_length), value=0)
        else:
        # 如果长于目标长度，则裁剪
            var.occ_aps = occ_aps[:target_length]
            var.occ_t = occ_t[:target_length]

        if current_length_gt<target_length:
            pad_length = target_length - current_length_gt
            var.gt_imgs = F.pad(var.gt_imgs, (0, 0, 0, 0, 0, pad_length), "constant", 0)
        else:
        # 如果长于目标长度，则裁剪
            var.gt_imgs = var.gt_imgs[:target_length]

        if current_length_depth<target_length:
            pad_length = target_length - current_length_depth
            var.depth_gt = F.pad(depth_gt, (0, 0, 0,0, 0, pad_length), "constant", 0)
        else:
        # 如果长于目标长度，则裁剪
            var.depth_gt = depth_gt[:target_length]

        var.gt = var.gt[clip_y:clip_y+self.roi_y,clip_x:clip_x+self.roi_x]
        var.other = var.other[clip_y:clip_y+self.roi_y,clip_x:clip_x+self.roi_x]
        # flip
        flip_mode = random.randint(0,3) if self.mode == 'Train' and self.opt.rand_flip is True else 0
        var = flip(var,mode=flip_mode)
        var.gt,var.other =  var.gt[None,...]/255.0,var.other[None,...]/255.0 # [1,H,W]
        var.occ_aps = var.occ_aps/255.0
        var.gt_imgs = var.gt_imgs/255.0
        var.voxelgrid = (var.voxelgrid - var.voxelgrid.min()) / (var.voxelgrid.max() - var.voxelgrid.min() + 1e-3)
        return var.get()

    def __len__(self):
        return len(self.data_list)

    def preprocess(self,opt:DatasetConfig):
        # numpy load data
        print("Loading numpy files ...")
        train_data_list,test_data_list = [],[]
        for content in opt.contents:
            if isinstance(content,list): content = "".join(content)
            content_data_list = []
            f_names = listdir(f'{opt.data_path}/{content}')
            content_data_list = [np.load(f_name,allow_pickle=True) for f_name in f_names]
            for i,data in enumerate(content_data_list):
                each = int(1/opt.val_ratio)
                if i%each == each//2: test_data_list.append(data)
                else: train_data_list.append(data)
        modes,data_lists = ['Train','Test'],[train_data_list,test_data_list]
        for mode,data_list in zip(modes,data_lists):
            save_path = f"{opt.data_path}/{opt.data_name}/{mode}"
            os.makedirs(save_path,exist_ok=True)
            # preprocess
            with tqdm(total=len(data_list)) as pbar:
                pbar.set_description_str(f'Processing {mode} data')
                for i,data in enumerate(data_list):
                    # load data
                    # events
                    events:np.ndarray = data['events']
                    x = events[:,0].astype(np.float64)
                    y = events[:,1].astype(np.float64)
                    t = events[:,2].astype(np.float64)
                    # other data
                    k,p,v = data['k'],data['p'],data['v']
                    fx = k
                    k = np.array([[679,0,191.3],[0,675.6,120.35],[0,0,1]])
                    occ_free_aps = data['occ_free_aps']
                    occ_aps = data['occ_aps']
                    # for aps in occ_free_aps:
                    #     print(aps.shape)
                    occ_free_aps = np.stack([cv2.undistort(aps,k,p) for aps in occ_free_aps],axis=0)
                    occ_aps = np.stack([cv2.undistort(aps,k,p) for aps in occ_aps],axis=0)
                    # print(occ_free_aps.shape)
                    occ_t:np.ndarray = data['occ_aps_ts']
                    gt_t:np.ndarray = data['occ_free_aps_ts']
                    gt_t = gt_t-(gt_t[0]-occ_t[0])
                    
                    ## pack events to event frames
                    x,y = x.astype(int),y.astype(int)
                    # calculate t
                    event_time_mask = np.where((t>=gt_t.min())&(t<=gt_t.max()))
                    x,y,t = x[event_time_mask],y[event_time_mask],t[event_time_mask]
                    t_min,t_max = t.min(),t.max()
                    t -= t_min
                    interval = (t_max-t_min)/opt.time_step
                    # convert events to event frames
                    voxelgrid = np.zeros((opt.time_step,opt.h,opt.w),dtype=np.float32)
                    T,H,W = voxelgrid.shape
                    voxelgrid = voxelgrid.ravel()
                    ind = (t/interval).astype(int)
                    ind[ind == T] -= 1
                    np.add.at(voxelgrid,x+y*W+ind*W*H,1)
                    voxelgrid = np.reshape(voxelgrid,(T,H,W))
                    voxelgrid = np.stack([cv2.undistort(each,k,p) for each in voxelgrid],axis=0)
                    # ref_t & time
                    gt_t = (gt_t-t_min).astype(np.float32)
                    occ_t = (occ_t-t_min).astype(np.float32)
                    time = interval*(np.arange(opt.time_step)+0.5).astype(np.float32)
                    ## ------- save processed data -------    
                    np.savez(f"{save_path}/{i:04d}.npz",voxelgrid=voxelgrid,time=time,occ_aps=occ_aps,gt=occ_free_aps,occ_t=occ_t,gt_t=gt_t,fx=fx,v=v)
                    pbar.update(1)