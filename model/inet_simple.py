from model.dnet import uniform_norm, ResnetBlock, LayerNorm, Mlp
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_

class UNet(nn.Module):
    def __init__(self,in_chans,out_chans,embed_dim,n_layers,n_scale,layer):
        super().__init__()
        self.init_conv_1 = nn.Sequential(*[
            nn.ReflectionPad2d(3),nn.Conv2d(in_chans[0],embed_dim//3,kernel_size=7),
            nn.BatchNorm2d(embed_dim//3),nn.ReLU(True)
        ])
        self.init_conv_2 = nn.Sequential(*[
            nn.ReflectionPad2d(3),nn.Conv2d(in_chans[1],embed_dim//3,kernel_size=7),
            nn.BatchNorm2d(embed_dim//3),nn.ReLU(True)
        ])
        self.init_conv_3 = nn.Sequential(*[
            nn.ReflectionPad2d(3),nn.Conv2d(in_chans[2],embed_dim//3,kernel_size=7),
            nn.BatchNorm2d(embed_dim//3),nn.ReLU(True)
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

    def forward(self, x):
        x_1 = self.init_conv_1(x[0])
        x_2 = self.init_conv_2(x[1])
        x_3 = self.init_conv_3(x[2])
        x = torch.cat([x_1, x_2,x_3],dim=1)
        x = self.down(x)
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


class iNet(nn.Module):
    def __init__(self):
        super().__init__()
        # self.init_fuse = ResnetBlock(27+64+27,nn.BatchNorm2d,out_dim=27,use_dropout=False)
        self.unet = UNet(in_chans=[64,27,27],out_chans=1,embed_dim=108,
        n_layers=7,n_scale=2,layer=CnnViT)
    
    def forward(self, x_e_ref, x_ref_i, x_i):
        # x_all = torch.cat([x_e_ref, x_ref_i, x_i], dim=1)
        x_e_ref = uniform_norm(x_e_ref)
        x_ref_i = uniform_norm(x_ref_i)
        x_i = uniform_norm(x_i)

        out = self.unet([x_e_ref, x_ref_i, x_i])
        return out

class CnnViT(nn.Module):
    def __init__(self, dim, num_heads=8, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super().__init__()
        self.cnn = ResnetBlock(dim,nn.BatchNorm2d,use_dropout=False)
        self.norm_in = LayerNorm(dim, LayerNorm_type)
        self.attn = SelfAttention(dim, num_heads, bias)
        self.norm_out = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim*ffn_expansion_factor)
        self.ffn = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)

    def forward(self, x):
        b,c,h,w = x.shape
        fused = self.norm_in(self.cnn(x))
        fused = fused + self.attn(x)
        fused = rearrange(fused, 'b c h w -> b (h w) c')
        fused = fused + self.ffn(self.norm_out(fused))
        fused = rearrange(fused, 'b (h w) c -> b c h w', h=h, w=w)
        return fused

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x):
        b,c,h,w = x.shape
        q,k,v = torch.split(self.qkv(x),[self.dim,self.dim,self.dim],dim=1)
        ## reshape
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # norm
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        ## attention
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out