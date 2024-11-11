import torch
from torch import Tensor
import torchvision
import torch.nn.functional as F
from datareader import PackedData


def print_gpu_usage():
    # 获取当前显存使用量
    allocated_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # 转换为 MB
    reserved_memory = torch.cuda.memory_reserved() / (1024 ** 2)    # 转换为 MB
    print(f"Allocated memory: {allocated_memory:.2f} MB")
    print(f"Reserved memory: {reserved_memory:.2f} MB")


def flip(var:PackedData,mode:int):
    assert mode in [0,1,2,3]
    h_flip = torchvision.transforms.RandomHorizontalFlip(p=1)
    v_flip = torchvision.transforms.RandomVerticalFlip(p=1)
    ts = var.voxelgrid.shape[0]
    occ_len = len(var.occ_aps)
    gt_imgs_len = len(var.gt_imgs)
    depth_len = len(var.depth_gt)
    if mode == 1:
        var.voxelgrid = torch.stack([v_flip(var.voxelgrid[idx]) for idx in range(ts)],dim=0)
        var.gt,var.other = v_flip(var.gt),v_flip(var.other)
        var.occ_aps = torch.stack([v_flip(var.occ_aps[idx]) for idx in range(occ_len)],dim=0)
        var.gt_imgs = torch.stack([v_flip(var.gt_imgs[idx]) for idx in range(gt_imgs_len)],dim=0)
        var.depth_gt = torch.stack([v_flip(var.depth_gt[idx]) for idx in range(depth_len)],dim=0)
    if mode == 2:
        var.voxelgrid = torch.stack([h_flip(var.voxelgrid[idx]) for idx in range(ts)],dim=0)
        var.gt,var.other = h_flip(var.gt),h_flip(var.other)
        var.occ_aps = torch.stack([h_flip(var.occ_aps[idx]) for idx in range(occ_len)], dim=0)
        var.gt_imgs = torch.stack([h_flip(var.gt_imgs[idx]) for idx in range(gt_imgs_len)],dim=0)
        var.depth_gt = torch.stack([h_flip(var.depth_gt[idx]) for idx in range(depth_len)], dim=0)
        var.v = -var.v
    if mode == 3:
        var.voxelgrid = torch.stack([v_flip(h_flip(var.voxelgrid[idx])) for idx in range(ts)],dim=0)
        var.gt,var.other = v_flip(h_flip(var.gt)),v_flip(h_flip(var.other))
        var.occ_aps = torch.stack([v_flip(h_flip(var.occ_aps[idx])) for idx in range(occ_len)],dim=0)
        var.gt_imgs = torch.stack([v_flip(h_flip(var.gt_imgs[idx])) for idx in range(gt_imgs_len)],dim=0)
        var.depth_gt = torch.stack([v_flip(h_flip(var.depth_gt[idx])) for idx in range(depth_len)],dim=0)
        var.v = -var.v
    return var

def warp(voxelgrid:Tensor,flow:Tensor):
    # voxelgrid: [bs,ts,H,W] | flow: [bs,ts,H,W]
    bs,ts,H,W = voxelgrid.shape
    voxelgrid = voxelgrid.reshape(bs*ts,1,H,W)# [bs*ts,1,H,W]
    flow = flow.reshape(bs*ts,1,H,W) # [bs*ts,1,H,W]
    # build grid
    xx,yy = torch.arange(0,W).view(1,-1).repeat(H,1),torch.arange(0,H).view(-1,1).repeat(1,W)
    xx,yy = xx.reshape(1,1,H,W),yy.reshape(1,1,H,W)
    grid = torch.cat((xx,yy),dim=1).repeat(bs*ts,1,1,1).float().to(flow.device) # [bs*ts,2,H,W]
    # flow
    each_flow_x = flow # [bs*ts,1,H,W]
    each_flow_y = torch.zeros_like(each_flow_x,dtype=each_flow_x.dtype,device=each_flow_x.device)
    each_flow = torch.cat([each_flow_x,each_flow_y],dim=1) # [bs*ts,2,H,W]
    # grid with flow
    grid += each_flow
    grid[:,0,:,:] = 2.0*grid[:,0,:,:].clone()/(W-1)-1.0
    grid[:,1,:,:] = 2.0*grid[:,1,:,:].clone()/(H-1)-1.0
    grid = grid.permute(0,2,3,1).contiguous() # [bs,H,W,2]
    result = F.grid_sample(voxelgrid,grid,padding_mode='zeros',align_corners=True) # [bs*ts,1,H,W]
    return result.reshape(bs,ts,H,W)

def dcn_warp(voxelgrid:Tensor,flow:Tensor):
    # voxelgrid: [bs,ts,H,W] | flow: [bs,ts,H,W]
    bs,ts,H,W = voxelgrid.shape
    zero_flow = torch.zeros_like(flow)
    flow = torch.stack([zero_flow,flow],dim=2) # [bs,ts,2,H,W]
    flow = flow.reshape(bs,ts*2,H,W) # [bs,ts*2,H,W]
    weight = torch.eye(ts,device=flow.device).float().reshape(ts,ts,1,1)
    return torchvision.ops.deform_conv2d(voxelgrid,flow,weight) # [bs,ts,H,W]

def nonzero_mean(tensor:Tensor,dim:int,keepdim:bool=False,eps=1e-3):
    numel = torch.sum((tensor>eps).float(),dim=dim,keepdim=keepdim)
    value = torch.sum(tensor,dim=dim,keepdim=keepdim)
    return value/(numel+eps)

def minmax_norm(t:Tensor,eps=1e-5):
    bs,ts,h,w = t.shape
    t = t.reshape(bs,ts*h*w)
    t_min,t_max = torch.min(t,dim=1,keepdim=True)[0],torch.max(t,dim=1,keepdim=True)[0]
    t_norm = (t-t_min)/(t_max-t_min+eps)
    return t_norm.reshape(bs,ts,h,w)

def self_cross_warp(depth:Tensor,flow:Tensor,time:Tensor,cross_time:Tensor=None)->Tensor:
    # with torch.no_grad():
    bs,ts = time.shape[:2]
    self_warped_depth = []
    if cross_time is None:
        for ch in range(ts):
            shift_each = flow*(time-time[:,ch:ch+1,...])
            recocus_each = nonzero_mean(dcn_warp(depth,-shift_each),dim=1,keepdim=True) # [bs,1,H,W]
            self_warped_depth.append(recocus_each)
    else:
        bs,ts = cross_time.shape[:2]
        for ch in range(ts):
            shift_each = flow*(time-cross_time[:,ch:ch+1,...])
            recocus_each = nonzero_mean(dcn_warp(depth,-shift_each),dim=1,keepdim=True) # [bs,1,H,W]
            self_warped_depth.append(recocus_each)
    return torch.cat(self_warped_depth,dim=1) # [bs,ts,H,W]

# def self_cross_warp(depth: Tensor, flow: Tensor, time: Tensor, cross_time: Tensor = None) -> Tensor:
#     bs, ts = time.shape[:2]
#     H, W = depth.shape[-2:]
#     if cross_time is None:
#         cross_time = time
#     t1 = cross_time.shape[1]
#     shift_all = flow.unsqueeze(1).expand(-1, t1, -1, -1, -1) * (cross_time.unsqueeze(2) - time.unsqueeze(1))
#     shift_all = shift_all.reshape(bs*t1,ts,H,W)
#     warped_depth = dcn_warp(depth.repeat(t1,1,1,1), shift_all)
#     recocus_all = nonzero_mean(warped_depth, dim=1, keepdim=True)  # [bs * t1, 1, H, W]
#     self_warped_depth = recocus_all.reshape(bs, t1, H, W)  # [bs, t1, H, W]
#     return self_warped_depth

def uniform_norm(tensor:Tensor)->Tensor:
    tensor_shape = tensor.shape
    bs = tensor_shape[0]
    tensor = tensor.reshape(bs,-1)
    mean,var = torch.mean(tensor,dim=1,keepdim=True),torch.var(tensor,dim=1,keepdim=True)
    tensor = (tensor-mean)/var
    tensor = tensor.reshape(*tensor_shape)
    return tensor  
