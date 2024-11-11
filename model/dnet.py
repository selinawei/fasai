import math
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
import timm

import torchvision
from timm.models.layers import trunc_normal_
from einops import rearrange
from model.MVDEM import MultiViewUniMatch


def count_parameters_in_MB(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


class DCNLayer(nn.Module):
    def __init__(self,in_ch,out_ch,offset_group,ksize=3,stride=1,padding=1):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.ksize = ksize
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding     
        # weight and bias
        self.weight = nn.Parameter(torch.Tensor(out_ch,in_ch,ksize,ksize))
        self.bias = nn.Parameter(torch.Tensor(out_ch))
        # offset conv
        self.conv_offset_mask = nn.Conv2d(in_ch,2*offset_group*ksize**2,ksize,stride,padding,bias=True)
        # init        
        stdv = 1. / math.sqrt(in_ch*(ksize**2))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()
        nn.init.constant_(self.conv_offset_mask.weight, 0.)
        nn.init.constant_(self.conv_offset_mask.bias, 0.)

    def forward(self,x:Tensor)->Tensor:
        # x:[bs,in_ch,h,w] offset:[bs,2*nc,h,w] | nc = offset_group*ksize**2
        out = self.conv_offset_mask(x) # [bs,nc,h,w]
        offset,mask = torch.chunk(out,chunks=2,dim=1) # [bs,nc,h,w]
        zero_offset = torch.zeros_like(offset) # [bs,nc,h,w]
        offset = torch.stack([zero_offset,offset],dim=2) # [bs,nc,2,h,w]
        bs,nc,_,h,w = offset.shape
        offset = offset.reshape(bs,nc*2,h,w) # # [bs,nc*2,h,w]
        mask = torch.sigmoid(mask) # [bs,nc,h,w]
        return torchvision.ops.deform_conv2d(x,offset,self.weight,self.bias,self.stride,self.padding,mask=mask) 


class DCNResBlock(nn.Module):

    def __init__(self,dim,norm_layer=nn.BatchNorm2d,use_dropout=False,use_bias=True):
        super().__init__()
        self.conv_block = self.build_block(dim,norm_layer,use_dropout,use_bias)

    def build_block(self,dim,norm_layer,use_dropout,use_bias):
        padding_type = 'reflect'
        conv_block = []
        # padding mode
        assert padding_type in ['zero','reflect']
        p = 1 if padding_type == 'zero' else 0
        if padding_type == 'reflect': conv_block += [nn.ReflectionPad2d(1)]
        # conv - 1
        conv_block += [DCNLayer(dim, dim, offset_group=16, padding=p),norm_layer(dim),nn.ReLU(True)]
        if use_dropout: conv_block += [nn.Dropout(0.5)]
        # conv - 2 
        if padding_type == 'reflect': conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]
        # # #
        return nn.Sequential(*conv_block)

    def forward(self,x:Tensor):
        inputs = x.clone()
        out = F.relu(inputs + self.conv_block(x))
        return out


class UNet(nn.Module):
    def __init__(self,in_chans,out_chans,embed_dim,n_layers,n_scale,layer):
        super().__init__()
        self.init_conv = nn.Sequential(*[
            nn.ReflectionPad2d(3),nn.Conv2d(in_chans,embed_dim,kernel_size=7),
            nn.BatchNorm2d(embed_dim),nn.ReLU(True)
        ])
        # init Conv
        self.down = []
        # add downsampling layers
        for i in range(n_scale):  
            ratio = 2**i
            self.down += nn.Sequential(*[
                nn.Conv2d(embed_dim*ratio,embed_dim*ratio*2,kernel_size=3,stride=2,padding=1,bias=False),
                nn.BatchNorm2d(embed_dim*ratio*2),nn.ReLU(True)
            ])
        self.down = nn.Sequential(*self.down)
        ratio = 2**n_scale
        # add core layers
        embed_dim = embed_dim + int(27*64/ratio)
        self.core = []
        for i in range(n_layers):
            self.core += [layer(embed_dim*ratio)]
        self.core = nn.Sequential(*self.core)
        # add upsampling layers
        self.up = []
        for i in range(n_scale):  
            ratio = 2**(n_scale-i)
            self.up += nn.Sequential(*[
                nn.ConvTranspose2d(embed_dim*ratio,embed_dim*ratio//2,kernel_size=3, stride=2,\
                                padding=1,output_padding=1,bias=False),
                nn.BatchNorm2d(embed_dim*ratio//2),nn.ReLU(True)
            ])
        self.up = nn.Sequential(*self.up)
        self.out_conv = nn.Sequential(*[
            nn.ReflectionPad2d(3),nn.Conv2d(embed_dim,out_chans,kernel_size=7),
        ])

    def forward(self, x:Tensor, mv_f):
        x = self.init_conv(x)
        x = self.down(x)
        x = torch.cat([x, mv_f], dim=1)
        x = self.core(x)
        x = self.up(x)
        x = self.out_conv(x)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d,nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data,0.0,0.02)
            if m.bias is not None: nn.init.constant_(m.bias.data,0.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data,1.0,0.02)
            nn.init.constant_(m.bias.data,0.0)


class ResnetBlock(nn.Module):

    def __init__(self,dim,norm_layer=nn.BatchNorm2d,out_dim=None,use_dropout=False,use_bias=True):
        super().__init__()
        self.conv_block = self.build_conv_block(dim,norm_layer,use_dropout,use_bias,out_dim=out_dim)
        if out_dim is not None:
            self.init_input = nn.Sequential(
                nn.Conv2d(dim, out_dim, kernel_size=3, padding=1, bias=use_bias), norm_layer(out_dim),nn.ReLU(True)
            )

    def build_conv_block(self,dim,norm_layer,use_dropout,use_bias,out_dim=None):
        padding_type = 'reflect'
        conv_block = []
        # padding mode
        assert padding_type in ['zero','reflect']
        p = 1 if padding_type == 'zero' else 0
        if padding_type == 'reflect': conv_block += [nn.ReflectionPad2d(1)]
        # conv - 1
        if out_dim is None:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),norm_layer(dim),nn.ReLU(True)]
        else:
            conv_block += [nn.Conv2d(dim, out_dim, kernel_size=3, padding=p, bias=use_bias),norm_layer(out_dim),nn.ReLU(True)]
            dim = out_dim
        if use_dropout: conv_block += [nn.Dropout(0.5)]
        # conv - 2 
        if padding_type == 'reflect': conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]
        # # #
        return nn.Sequential(*conv_block)

    def forward(self,x:Tensor):
        inputs = x.clone()
        if hasattr(self, "init_input"):
            inputs = self.init_input(inputs)
        out = F.relu(inputs + self.conv_block(x))
        return out


class MutualAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_fused = ResnetBlock(3*dim,nn.BatchNorm2d,out_dim=dim,use_dropout=False)
        
    def forward(self, x_i:Tensor, x_e:Tensor, y:Tensor):
        assert x_e.shape == y.shape == x_i.shape
        b,c,h,w = x_e.shape
        fused_out = None
        for combo in [[x_e, x_i], [x_e, x_e], [x_i, x_i]]:
            q,k,v = self.q(y),self.k(combo[0]),self.v(combo[1])
            ## reshape
            q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
            k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
            v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
            ## norm
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
            ## attention
            attn = (q @ k.transpose(-2, -1)) * self.temperature
            attn = attn.softmax(dim=-1)
            out = (attn @ v)
            out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
            out = self.project_out(out)
            if fused_out is None:
                fused_out = out
            else:
                fused_out = torch.cat([fused_out, out], dim=1)
        
        out = self.project_fused(fused_out)

        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PoseGuidedFusionAttention(nn.Module):
    def __init__(self, dim, num_heads=9, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super().__init__()
        self.norm_vox = LayerNorm(dim, LayerNorm_type)
        self.norm_img = LayerNorm(dim, LayerNorm_type)
        self.norm_time = LayerNorm(dim, LayerNorm_type)
        self.attn = MutualAttention(dim, num_heads, bias)
        # mlp
        self.norm_out = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim*ffn_expansion_factor)
        self.ffn = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)

    def forward(self, image:Tensor, voxelgrid:Tensor, pose:Tensor):
        assert voxelgrid.shape == pose.shape == image.shape
        b,c,h,w = voxelgrid.shape
        fused = voxelgrid + self.attn(self.norm_img(voxelgrid), self.norm_vox(voxelgrid),self.norm_time(pose))
        fused = rearrange(fused, 'b c h w -> b (h w) c') # b, h*w, c
        fused = fused + self.ffn(self.norm_out(fused))
        fused = rearrange(fused, 'b (h w) c -> b c h w',h=h,w=w)
        return fused


class Flownet(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        embed_dim = 128 # config.flownet_embed_dim
        unet_layer = 4 # config.flownet_n_layers
        unet_scale = 3 # config.flownet_n_scale
        ts = 64        # config.ts
        self.init_vox = ResnetBlock(ts,nn.BatchNorm2d,out_dim=27,use_dropout=False)
        self.init_img = ResnetBlock(27,nn.BatchNorm2d,out_dim=27,use_dropout=False)
        self.init_pose = ResnetBlock(27,nn.BatchNorm2d,out_dim=27,use_dropout=False)
        self.pcm = PoseGuidedFusionAttention(dim=27)
        self.mvdem = MultiViewUniMatch()
        self.core_block = UNet(27,27,embed_dim,unet_layer,unet_scale,layer=DCNResBlock)
        # self.core_ev_block = UNet(ts,ts,embed_dim,unet_layer,unet_scale,layer=DCNResBlock)

    def forward(self,x_i:Tensor,x_e:Tensor,pose:Tensor,intr=None, extr=None)->Tensor:
        # input: [bs,ts,h,w]
        bs,ts,h,w = x_e.shape
        x_e = self.init_vox(uniform_norm(x_e))
        x_i = self.init_img(uniform_norm(x_i))
        pose = self.init_pose(pose.repeat(1,1,h,w))

        x = self.pcm(x_i,x_e,pose)
        mv_feature = self.mvdem(x.unsqueeze(2), intrinsics=intr,extrinsics=extr,min_depth=(1.0 / 0.1)*torch.ones((bs,27), device=x.device),max_depth=(1.0 / 20)*torch.ones((bs,27), device=x.device))
        x_27 = self.core_block(x, rearrange(mv_feature, "b v c h w -> b (v c) h w"))
        x_27 = torch.relu(x_27)
        return x_27


def uniform_norm(tensor:Tensor)->Tensor:
    tensor_shape = tensor.shape
    bs = tensor_shape[0]
    tensor = tensor.reshape(bs,-1)
    mean,var = torch.mean(tensor,dim=1,keepdim=True),torch.var(tensor,dim=1,keepdim=True)
    tensor = (tensor-mean)/var
    tensor = tensor.reshape(*tensor_shape)
    return tensor 


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias



class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return rearrange(self.body(rearrange(x, 'b c h w -> b (h w) c')), 'b (h w) c -> b c h w', h=h, w=w)


if __name__ == '__main__':
    import torch
    from config import GlobalConfig as Config


    inputf = torch.randn(2, 30, 260, 346).cuda()
    inpute = torch.randn(2, 64, 260, 346).cuda()
    size = count_parameters_in_MB(TransfuserBackbone(Config()).image_encoder)
    print(size)
    size = count_parameters_in_MB(TransfuserBackbone(Config()).event_layer1)
    print(size)
    size = count_parameters_in_MB(TransfuserBackbone(Config()).event_layer2)
    print(size)
    size = count_parameters_in_MB(TransfuserBackbone(Config()).event_layer3)
    print(size)
    size = count_parameters_in_MB(TransfuserBackbone(Config()).decoder)
    print(size)
    size = count_parameters_in_MB(TransfuserBackbone(Config()).transformer1)
    dnet = TransfuserBackbone(Config()).cuda()

    flow = dnet(inputf, inpute)
    print(size)
    print(flow.shape)