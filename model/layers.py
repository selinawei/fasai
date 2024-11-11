import math
import torch
from torch import Tensor
import torch.nn as nn
import torchvision
import functools

class ResnetGenerator(nn.Module):
    def __init__(self,input_nc,output_nc,ngf,n_blocks,n_down,use_dropout=True,acf="sigmoid"):
        assert n_blocks >= 0
        super(ResnetGenerator, self).__init__()
        norm_layer = functools.partial(nn.BatchNorm2d,momentum=0.01,affine=True,track_running_stats=True)
        use_bias = True
        # init Conv
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc,ngf,kernel_size=7,padding=0,bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        # add downsampling layers
        for i in range(n_down):  
            mult = 2**i
            model += [nn.Conv2d(ngf*mult,ngf*mult*2,kernel_size=3,stride=2,padding=1,bias=use_bias),
                      norm_layer(ngf*mult*2),
                      nn.ReLU(True)]
        # add ResNet blocks
        mult = 2**n_down
        for i in range(n_blocks):       
            model += [ResnetBlock(ngf*mult,norm_layer,use_dropout,use_bias)]
        # add upsampling layers
        for i in range(n_down):  
            mult = 2**(n_down-i)
            model += [nn.ConvTranspose2d(ngf*mult,int(ngf*mult/2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf*mult/2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        assert acf in ['sigmoid','relu','none']
        if acf == 'sigmoid': model += [nn.Sigmoid()]
        if acf == 'relu': model += [nn.ReLU6(True)]
        # model
        self.model = nn.Sequential(*model)
        init_weights(self,init_type='kaiming')

    def forward(self, input):
        return self.model(input)

class ResnetBlock(nn.Module):

    def __init__(self,dim,norm_layer,use_dropout,use_bias=True):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim,norm_layer,use_dropout,use_bias)

    def build_conv_block(self,dim,norm_layer,use_dropout,use_bias):
        padding_type = 'reflect'
        conv_block = []
        # padding mode
        assert padding_type in ['zero','reflect']
        p = 1 if padding_type == 'zero' else 0
        if padding_type == 'reflect': conv_block += [nn.ReflectionPad2d(1)]
        # conv - 1
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),norm_layer(dim),nn.ReLU(True)]
        if use_dropout: conv_block += [nn.Dropout(0.5)]
        # conv - 2 
        if padding_type == 'reflect': conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]
        # # #
        return nn.Sequential(*conv_block)

    def forward(self,x:Tensor):
        inputs = x.clone()
        out = inputs + self.conv_block(x)
        return out

def init_weights(net:nn.Module,init_type='normal',init_gain=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data,0.0,init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data,gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data,a=0,mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data,gain=init_gain)
            else:
                raise NotImplementedError(f'initialization method [{init_type}] is not implemented.')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data,0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data,1.0,init_gain)
            nn.init.constant_(m.bias.data,0.0)
    # apply the initialization function <init_func>
    net.apply(init_func)

class DCNGenerator(nn.Module):
    def __init__(self,input_nc,n_dcn,fea_ratio=2):
        super().__init__()
        model = []
        assert n_dcn > 0
        nc = [input_nc]+[input_nc*fea_ratio for _ in range(n_dcn-1)]+[input_nc]
        group = input_nc
        for n in range(n_dcn):
            model += self.dcn_block(nc[n],nc[n+1],group)
        self.dcn = nn.Sequential(*model)
    
    def dcn_block(self,input_nc,output_nc,group):
        return nn.Sequential(*[
            DCNLayer(input_nc,output_nc,group,3,1,1),
            nn.BatchNorm2d(output_nc),nn.ReLU(True)
        ])
    
    def forward(self,x:Tensor):
        return self.dcn(x)

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

class ChannelAttention(nn.Module):
    def __init__(self,in_planes,ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes,in_planes//ratio,1,bias=False),nn.ReLU(True),
            nn.Conv2d(in_planes//ratio,in_planes,1, bias=False))

    def forward(self, x:Tensor):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return torch.sigmoid(avgout+maxout)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2,1,kernel_size=7,padding=3, bias=False)

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return torch.sigmoid(x)

if __name__ == "__main__":
    pass

