import sys
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
sys.path.append("..")
# import model.swin as swin
from model.inet import DeOccNet
from model.inet_simple import iNet
from model.dnet import Flownet
# from arguments import Config
from utils.tensor import uniform_norm,minmax_norm,dcn_warp,self_cross_warp
from datareader import PackedData
from model.config import GlobalConfig as Config

# class Encoder(nn.Module):
#
#     """
#     input: occ_aps ([bs,?,H,W]) and voxelgrid([bs,ts,H,W])
#     """
#     def __init__(self,opt:Config) -> None:
#         super().__init__()
#         img_size,n_scale = opt.img_size,opt.depthnet_n_scale
#         patch_resolution = img_size//(2**n_scale)
#         self.embd_trans = swin.embed()
#         self.mshstb = swin.MSSTB(opt.ts,opt.ts,patch_resolution,embed_dim=opt.depthnet_embed_dim,
#                                  Layer=swin.DeformableBasicLayer,n_Layers=opt.depthnet_n_layers,n_scale=n_scale)
#     def forward(self,occ:Tensor,voxelgrid:Tensor)->Tensor:
#         occ,voxelgrid = uniform_norm(occ),uniform_norm(voxelgrid)
#         occ,voxelgrid = self.embd_trans(occ,voxelgrid)
#         x = self.mshstb(x)
#
#
# class DepthNet(nn.Module):
#     def __init__(self,opt:Config):
#         super().__init__()
#         img_size,n_scale = opt.img_size,opt.depthnet_n_scale
#         patch_resolution = img_size//(2**n_scale)
#         self.FFE  = dnet.FFE(opt.ts+opt.occ_frame_num,opt.ts,patch_resolution,embed_dim=opt.depthnet_embed_dim,
#                 layer=dnet.BasicLayer,n_layers=opt.depthnet_n_layers,n_scale=n_scale)
#         self.mshstb = swin.MSSTB(opt.ts+opt.occ_frame_num,opt.ts,patch_resolution,embed_dim=opt.depthnet_embed_dim,
#                 layer=swin.DeformableBasicLayer,n_layers=opt.depthnet_n_layers,n_scale=n_scale)
#
#     def forward(self,x:Tensor,y:Tensor)->Tensor:
#         # input: [bs,ts,h,w]
#         x = uniform_norm(x)
#         y = uniform_norm(y)
#         outd = self.mshstb(x,y)
#         outd = F.relu6(x)+torch.sigmoid(x)
#         return outd
#
# class FrameNet(nn.Module):
#     def __init__(self,opt:Config):
#         super().__init__()
#         img_size,n_scale = opt.img_size,opt.framenet_n_scale
#         patch_resolution = img_size//(2**n_scale)
#         self.mshstb = swin.MSSTB(opt.ts+opt.occ_frame_num,1,patch_resolution,embed_dim=opt.framenet_embed_dim,
#                 layer=swin.HybridBasicLayer,n_layers=opt.framenet_n_layers,n_scale=n_scale)
#
#     def forward(self,x:Tensor)->Tensor:
#         # input: [bs,ts,h,w]
#         x = uniform_norm(x)
#         x = self.mshstb(x)
#         x = torch.sigmoid(x)
#         return x

def print_gpu_usage():
    # 获取当前显存使用量
    allocated_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # 转换为 MB
    reserved_memory = torch.cuda.memory_reserved() / (1024 ** 2)    # 转换为 MB

def interp(img,time_img,time_ev):
    bs = time_ev.shape[0]
    slices = [torch.min(torch.abs(time_img - time_ev[:,i:i+1,...]),dim=1, keepdim=True)[1].reshape(bs) for i in range(time_ev.shape[1])]
    img_refts = [torch.stack([img[bs, ts:ts + 1, ...] for bs, ts in enumerate(slice)], dim=0) for slice in slices]
    ev = torch.cat(img_refts, dim=1)  # [bs, 64, H, W]

    return ev # [bs,64,H,W]

def depth_proj(depth, v, ori_t, out_t):
    pass

class FEASAI(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.depthnet = TransfuserBackbone(Config())
        self.depthnet = Flownet()
        # self.framenet = DeOccNet(27)
        self.framenet = iNet()

    # def forward(self, eps=1e-3):
    def forward(self,var:PackedData,eps=1e-3)->PackedData:
        # voxelgrid: [bs,ts,H,W] # time: [bs,ts]
        # unpack var


        # occ_aps_ts:Tensor = var.occ_aps_ts

        # occ_t:Tensor = var.occ_t.reshape(bs,occ_t_len,1,1)
        # event
        voxelgrid:Tensor = var.voxelgrid
        bs, ts , H, W= voxelgrid.shape
        occ_t_len = 27
        time:Tensor = var.time.reshape(bs,ts,1,1)
        vox_t_len = var.time.shape[1]
        # frame
        occ_aps: Tensor = var.occ_aps.reshape(bs,occ_t_len,H,W)
        occ_t:Tensor = var.occ_t.reshape(bs,occ_t_len,1,1)
        #ref
        reft:Tensor = var.gt_t.reshape(bs,1,1,1)
        # param
        fx: Tensor = var.fx[:, 0, 0].reshape(bs, 1, 1, 1)
        v:Tensor = var.v.reshape(bs,1,1,1)

        # occ_aps = F.pad(occ_aps, (0, 0, 0, 0, 0, 30 - occ_aps.shape[1]), value=0)
        # occ_t = F.pad(occ_t, (0, 0, 0, 0, 0, 30 - occ_t.shape[1]), value=0)

        # # test
        # bs = 4
        # ts = 64
        # occ_t_len = 30
        # voxelgrid = torch.rand(bs,ts,260,346).cuda()
        # H,W = voxelgrid.shape[2],voxelgrid.shape[3]
        # time = torch.rand(bs,ts,1,1).cuda()
        # occ_aps = torch.rand(bs,occ_t_len,260,346).cuda()
        # occ_t = torch.rand(bs,occ_t_len,1,1).cuda()
        # reft = torch.rand(bs,2,1,1).cuda()
        # fx = torch.rand(bs,1,1,1).cuda()
        # v = torch.rand(bs,1,1,1).cuda()

        # aps_plus_voxel:Tensor = torch.cat((voxelgrid,occ_aps),dim=1)
        # print(occ_t.shape)
        # encode = self.encode.forward(occ_aps,voxelgrid)
        
        # # # depth
        # pose = occ_t * v.expand(occ_t.shape)
        # mask = (v <= 0).expand_as(pose)  # [B, T, 1, 1]
        # # min_pose, _ = pose.min(dim=1, keepdim=True)
        # # pose = torch.where(mask, pose - min_pose, pose)
        # pose = torch.where(mask, -1 * pose, pose)

        # std_dev = 0.02
        # noise = std_dev * torch.randn_like(pose)
        # pose = torch.relu(pose + noise)
        pose = occ_t * v.expand(occ_t.shape)

        Tx = v * (occ_t - occ_t[:, 0:1, 0:1, 0:1])  # [bs, 27,1,1]
        base_extr_m = torch.eye(4, device=occ_t.device).float()
        base_extr_m = base_extr_m.unsqueeze(0).unsqueeze(0).expand(bs, occ_t_len, -1, -1).clone()  # [bs, 27, 4, 4]
        base_extr_m[:, :, 0, 3] = Tx.squeeze(-1).squeeze(-1)  # [bs, 27, 4,4]

        flow_27 = self.depthnet.forward(x_i=occ_aps, x_e=voxelgrid, pose=pose, intr=var.fx.unsqueeze(1).repeat(1,occ_t_len,1,1), extr=base_extr_m) # [bs,ts,H,W] [bs,ts,H,W] [bs,ts,1,1]
        
        # depth = fx*v/(flow.clone()+eps)
        flow_64 = interp(flow_27, occ_t, time)
        flow_sign = v / torch.abs(v)
        flow_64 = flow_64+eps # [bs,ts,H,W]
        flow_27 = flow_27+eps # [bs,27,H,W]
        depth_64 = fx*v/(flow_sign*flow_64)
        depth_27 = fx*v/(flow_sign*flow_27)

        assert (depth_64 >= 0).all() and (depth_27 >= 0).all(), "Depth must me positive."

        # self_warped_depth_64 = self_cross_warp(depth_64, flow_64, time)
        # self_warped_depth_27 = self_cross_warp(depth_27, flow_27, occ_t)
        # cross_warped_depth_27 = self_cross_warp(depth_64, flow_64, time, cross_time=occ_t)
        # cross_warped_depth_64 = self_cross_warp(depth_27, flow_27, occ_t, cross_time=time)
        # self_warped_img_64 = self_cross_warp(voxelgrid, flow_64, time)
        # self_warped_img_27 = self_cross_warp(occ_aps, flow_27, occ_t)
        # cross_warped_img_27 = self_cross_warp(voxelgrid, flow_64, time, cross_time=occ_t)
        # cross_warped_img_64 = self_cross_warp(occ_aps, flow_27, occ_t, cross_time=time)
         
        # flow & shift
        reft_each = reft[:,0:1,:,:]
        shift_ref_ev = flow_64*(time-reft_each)
        shift_ref_img = flow_27*(occ_t-reft_each)
        refocus_voxelgrid_ev = dcn_warp(voxelgrid,-shift_ref_ev) # [bs,64,H,W]
        refocus_img = dcn_warp(occ_aps,-shift_ref_img) # [bs,27,H,W]
        refocus_depth = dcn_warp(depth_27,-shift_ref_img)

        # save into packed data
        # var.pred_img_depth = depth_27
        # var.pred_ev_depth = depth_64
        # var.self_warped_depth_img = self_warped_depth_27
        # var.self_warped_depth_ev = self_warped_depth_64
        # var.cross_warped_depth_img = cross_warped_depth_27
        # var.cross_warped_depth_ev = cross_warped_depth_64
        # var.self_warped_img = self_warped_img_27
        # var.self_warped_ev = self_warped_img_64
        # var.cross_warped_img = cross_warped_img_27
        # var.cross_warped_ev = cross_warped_img_64

        ev_idx = torch.argmin(torch.abs(var.time - var.gt_t.reshape(bs,1)), dim=1).reshape(bs,1,1,1)
        img_idx = torch.argmin(torch.abs(var.occ_t - var.gt_t.reshape(bs,1)), dim=1).reshape(bs,1,1,1)

        var.ev_ref_frame = torch.mean(refocus_voxelgrid_ev, dim=1, keepdim=True)
        var.img_ref_frame = torch.mean(refocus_img, dim=1, keepdim=True)
        var.depth_ref_frame = torch.mean(refocus_depth, dim=1, keepdim=True)
        var.ev_depth_frame = torch.gather(depth_64,1,ev_idx.expand(-1, 1, H, W))
        var.img_depth_frame = torch.gather(depth_27,1,img_idx.expand(-1, 1, H, W))
        var.gt_depth_frame = torch.gather(var.depth_gt,1,img_idx.expand(-1, 1, H, W))

        # all_corr = 0
        # for i in range(27):
        #     corr = F.conv2d(var.img_depth_frame, depth_27[:,i:i+1,...], padding=(H//2, W//2))
        #     all_corr -= torch.max(corr)
        # var.depth_consistent = all_corr

        if var.epoch is not None and var.epoch < 100:
            pred_frame = self.framenet(refocus_voxelgrid_ev, refocus_img, refocus_img) # [bs,occ_len,H,W]
        else:
            pred_frame = self.framenet(refocus_voxelgrid_ev, refocus_img, occ_aps)

        var.pred_frame = pred_frame[:,0:1,...]
        return var


def count_parameters_in_MB(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
if __name__ == "__main__":
    #pass
    # opt = Config()
    import torch
    from config import GlobalConfig as Config
    size = count_parameters_in_MB(FEASAI())
    model = FEASAI().cuda()
    pred_frame, depth_reft, self_warped_depth, acc_frame_reft, occ_frame_reft = model()
    # print("size",size)
    # print("pred_frame",pred_frame.shape)
    # print("depth_reft",depth_reft.shape)
    # print("self_warped_depth",self_warped_depth.shape)
    # print("acc_frame_reft",acc_frame_reft.shape)
    # print("occ_frame_reft",occ_frame_reft.shape)
