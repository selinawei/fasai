import torch
import torch.nn as nn
from torch import Tensor
from torchvision import models

from datareader import PackedData
from utils.tensor import dcn_warp
import utils

class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        model_vgg = models.vgg16(pretrained=False)
        state_dict_vgg = torch.load("/workspace/xak1wx/FEASAI/model/vgg16-397923af.pth")
        model_vgg.load_state_dict(state_dict_vgg)
        features = model_vgg.features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return out

class ImageLoss:
    def __init__(self,weights=[1e0,32,2e-4],percep_weights=[1e-1,1/21,10/21,10/21]):
        assert len(percep_weights) == 4
        self.percep_weights = percep_weights
        self.vgg = Vgg16().cuda()
        self.vgg.eval()
        self.MSE = nn.MSELoss() 
        self.L1 = nn.L1Loss()
        self.content_weights,self.pixel_weights,self.tv_weights = weights

    def __call__(self,pred:Tensor,gt:Tensor):
        assert pred.shape == gt.shape
        pred,gt = pred.cuda(),gt.cuda()
        b,c,h,w = pred.shape
        # assert c in [1,3]
        ## calculate pixel loss
        L_pixel = self.L1(pred,gt) / c
        # calculate total variation regularization (anisotropic version)
        diff_i = torch.sum(torch.abs(pred[:, :, :, 1:] - gt[:, :, :, :-1]))
        diff_j = torch.sum(torch.abs(pred[:, :, 1:, :] - gt[:, :, :-1, :]))
        L_tv = (diff_i+diff_j)/float(c*h*w)
        ## gray to 3-dim image to fit vgg16
        # if c == 1: pred,gt = pred.repeat(1,3,1,1),gt.repeat(1,3,1,1)    
        ## calculate perceptual loss
        L_perceptual = 0.
        pred_features = self.vgg(pred.reshape(b*c, 1, h, w).repeat(1,3,1,1))
        gt_features = self.vgg(gt.reshape(b*c, 1, h, w).repeat(1,3,1,1))
        L_perceptual_list = []
        for predf,gtf in zip(pred_features,gt_features):
            L_perceptual_list.append(self.MSE(predf,gtf))
        for weight,each_loss in zip(self.percep_weights,L_perceptual_list):
            L_perceptual += weight*each_loss
        ## total loss
        total_loss = self.content_weights*L_perceptual/c+self.pixel_weights*L_pixel+self.tv_weights*L_tv
        return total_loss

class TotalLoss:
    def __init__(
        self,
        image_weights=[1e0,32,2e-4],
        image_perweights=[1e-1,1/21,10/21,10/21]):
        # init
        self.L1 = nn.L1Loss()
        self.L2 = nn.MSELoss()
        self.imloss = ImageLoss(weights=image_weights,percep_weights=image_perweights)
    
    def __call__(self,var:PackedData,epoch=None,eps=1e-3):
        bs = len(var.fx)
        fx: Tensor = var.fx[0][0][0].reshape(1, 1, 1, 1).repeat(bs, 1, 1, 1)
        v:Tensor = var.v.reshape(bs,1,1,1)
        reft:Tensor = var.gt_t.reshape(bs,1,1,1) 
        othert:Tensor = var.other_t.reshape(bs,1,1,1)

        # # depth consistent loss
        # # consistent_loss_d = self.L1(var.pred_img_depth,var.self_warped_depth_img) + self.L1(var.pred_img_depth,var.cross_warped_depth_img) + self.L1(var.self_warped_depth_img,var.cross_warped_depth_img)
        # # consistent_loss_d += self.L1(var.pred_ev_depth,var.self_warped_depth_ev) + self.L1(var.pred_ev_depth,var.cross_warped_depth_ev) + self.L1(var.self_warped_depth_ev,var.cross_warped_depth_ev)
        # # consistent_loss_i = self.imloss(var.self_warped_img,var.cross_warped_img) + self.imloss(var.self_warped_img,var.gt_imgs) + self.imloss(var.cross_warped_img,var.gt_imgs) + self.imloss(var.self_warped_ev,var.cross_warped_ev)
        # # depth loss
        # # depth_loss = self.L2(var.pred_img_depth,var.depth_gt) + self.L2(var.self_warped_depth_img,var.depth_gt) + self.L2(var.cross_warped_depth_img,var.depth_gt)
        # depth_loss = self.L2(var.pred_img_depth,var.depth_gt) + self.L2(var.self_warped_depth_img,var.depth_gt) + self.L2(var.cross_warped_depth_img,var.depth_gt)

        # # pred gt loss
        # frame_loss = self.imloss(var.pred_frame,var.gt) + 0.2*self.imloss(var.ev_ref_frame,var.gt) + 0.2*self.imloss(var.img_ref_frame,var.gt)

        # consistent_loss_i = depth_loss
        # consistent_loss_d = frame_loss

        # if epoch is not None and epoch < 50:
        #     consistent_loss = consistent_loss_i
        # else:
        #     consistent_loss = consistent_loss_i + consistent_loss_d / var.pred_img_depth.shape[1]

        depth_loss = 2*(self.L1(var.gt_depth_frame,var.img_depth_frame) + self.L1(var.gt_depth_frame,var.ev_depth_frame) + self.L1(var.depth_ref_frame, var.gt_depth_frame))
            
        consistent_loss = 10*(self.imloss(var.ev_ref_frame,var.gt) + self.imloss(var.img_ref_frame,var.gt))
        frame_loss = self.imloss(var.pred_frame,var.gt)
        if epoch is not None and epoch >= 100:
            frame_loss *= 10


        return ["frame","depth","consistent"],[frame_loss,depth_loss,consistent_loss]


def find_last_non_zero_channel(depth_gt: torch.Tensor) -> torch.Tensor:
    ts, H, W = depth_gt.shape
    
    # 计算每个通道是否全零
    is_non_zero = torch.any(depth_gt.view(ts, -1) != 0, dim=1).int()  # [27]
    
    # 找到最后一个非全零的通道索引
    # 使用 torch.flip 将通道顺序反转，然后找到第一个非全零的通道索引
    is_non_zero = torch.flip(is_non_zero, dims=[0])
    last_non_zero_idx = ts - 1 - torch.argmax(is_non_zero, dim=0)
    
    return last_non_zero_idx