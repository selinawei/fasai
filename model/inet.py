import torch
import torch.nn as nn
import torch.nn.functional as F
from model.submodules import define_G,  ChannelAttentionv2, PatchEmbed, PatchUnEmbed, FusionSwinTransformerBlock
from timm.models.layers import  trunc_normal_
from torch import Tensor
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import model.layers as layers

class DeOccNet(nn.Module):
    def __init__(self, views):
        super(DeOccNet, self).__init__()
        # Feature Extraction
        self.init_feature = nn.Sequential(
            nn.Conv2d(27, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            ResASPPB(64),
        )

        # Encoding
        self.encoder_1 = nn.Sequential(RB(64), RB(64), EB(64))
        self.encoder_2 = nn.Sequential(RB(128), RB(128), EB(128))
        self.encoder_3 = nn.Sequential(RB(256), RB(256), EB(256))
        self.encoder_4 = nn.Sequential(RB(512), RB(512), EB(512))

        # Decoding
        self.decoder2label_4 = nn.Sequential(DB(1024), RB(512), RB(512))
        self.decoder2label_3 = nn.Sequential(nn.Conv2d(1024, 512, 1, 1, 0, bias=False), DB(512), RB(256), RB(256))
        self.decoder2label_2 = nn.Sequential(nn.Conv2d(512, 256, 1, 1, 0, bias=False), DB(256), RB(128), RB(128))
        self.decoder2label_1 = nn.Sequential(nn.Conv2d(256, 128, 1, 1, 0, bias=False), DB(128), RB(64), RB(64))
        self.ToLabel = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, x):
        # B，T，H，W
        Bf,Tf,Hf,Wf = x.shape
        # x = torch.cat([x_e_ref_i, x_ref_i,x_i], dim=1)
        buffer = self.init_feature(x)
        buffer_1 = self.encoder_1(buffer)
        buffer_2 = self.encoder_2(buffer_1)
        buffer_3 = self.encoder_3(buffer_2)
        codes = self.encoder_4(buffer_3)

        buffer_3L = self.decoder2label_4(codes)
        buffer_3L_ = torch.cat((buffer_3L, buffer_3), 1)
        buffer_2L = self.decoder2label_3(buffer_3L_)
        buffer_2L_ = torch.cat((buffer_2L, buffer_2), 1)
        buffer_1L = self.decoder2label_2(buffer_2L_)
        buffer_1L_ = torch.cat((buffer_1L, buffer_1), 1)
        buffer_L = self.decoder2label_1(buffer_1L_)
        label_out = self.ToLabel(buffer_L)

        return label_out

    # Encoder Block


class EB(nn.Module):
    def __init__(self, channels):
        super(EB, self).__init__()
        self.BN = nn.BatchNorm2d(channels)
        self.body = nn.Sequential(
            nn.Conv2d(channels, 2 * channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.shortpath = nn.Sequential(
            nn.Conv2d(channels, 2 * channels, kernel_size=3, stride=2, padding=1, bias=False)
        )

    def forward(self, x):
        buffer = self.BN(x)
        out_1 = self.body(buffer)
        out_2 = self.shortpath(buffer)
        out = out_1 + out_2
        return out


# Decoder Block
class DB(nn.Module):
    def __init__(self, channels):
        super(DB, self).__init__()
        self.BN = nn.BatchNorm2d(channels)
        self.body = nn.Sequential(
            nn.ConvTranspose2d(channels, int(0.5 * channels), kernel_size=3, stride=2, padding=1, output_padding=1,
                               bias=False),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.shortpath = nn.Sequential(
            nn.ConvTranspose2d(channels, int(0.5 * channels), kernel_size=3, stride=2, padding=1, output_padding=1,
                               bias=False),
        )

    def forward(self, x):
        buffer = self.BN(x)
        out_1 = self.body(buffer)
        out_2 = self.shortpath(buffer)
        out = out_1 + out_2
        return out


# Residual Block
class RB(nn.Module):
    def __init__(self, channels):
        super(RB, self).__init__()
        self.BN = nn.BatchNorm2d(channels)
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        buffer = self.BN(x)
        out_1 = self.body(buffer)
        out = out_1 + buffer
        return out


class ResASPPB(nn.Module):
    def __init__(self, channels):
        super(ResASPPB, self).__init__()
        self.conv_1_0 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, 1, bias=False),
                                      nn.LeakyReLU(0.1, inplace=True))
        self.conv_1_1 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 2, 2, bias=False),
                                      nn.LeakyReLU(0.1, inplace=True))
        self.conv_1_2 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 4, 4, bias=False),
                                      nn.LeakyReLU(0.1, inplace=True))
        self.conv_1_3 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 8, 8, bias=False),
                                      nn.LeakyReLU(0.1, inplace=True))
        self.conv_1_4 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 16, 16, bias=False),
                                      nn.LeakyReLU(0.1, inplace=True))
        self.conv_1_5 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 32, 32, bias=False),
                                      nn.LeakyReLU(0.1, inplace=True))
        self.conv_2_0 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, 1, bias=False),
                                      nn.LeakyReLU(0.1, inplace=True))
        self.conv_2_1 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 2, 2, bias=False),
                                      nn.LeakyReLU(0.1, inplace=True))
        self.conv_2_2 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 4, 4, bias=False),
                                      nn.LeakyReLU(0.1, inplace=True))
        self.conv_2_3 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 8, 8, bias=False),
                                      nn.LeakyReLU(0.1, inplace=True))
        self.conv_2_4 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 16, 16, bias=False),
                                      nn.LeakyReLU(0.1, inplace=True))
        self.conv_2_5 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 32, 32, bias=False),
                                      nn.LeakyReLU(0.1, inplace=True))
        self.conv_3_0 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, 1, bias=False),
                                      nn.LeakyReLU(0.1, inplace=True))
        self.conv_3_1 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 2, 2, bias=False),
                                      nn.LeakyReLU(0.1, inplace=True))
        self.conv_3_2 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 4, 4, bias=False),
                                      nn.LeakyReLU(0.1, inplace=True))
        self.conv_3_3 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 8, 8, bias=False),
                                      nn.LeakyReLU(0.1, inplace=True))
        self.conv_3_4 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 16, 16, bias=False),
                                      nn.LeakyReLU(0.1, inplace=True))
        self.conv_3_5 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 32, 32, bias=False),
                                      nn.LeakyReLU(0.1, inplace=True))
        self.conv_1 = nn.Conv2d(channels * 6, channels, 1, 1, 0, bias=False)
        self.conv_2 = nn.Conv2d(channels * 6, channels, 1, 1, 0, bias=False)
        self.conv_3 = nn.Conv2d(channels * 6, channels, 1, 1, 0, bias=False)

    def forward(self, x):
        buffer_1_0 = self.conv_1_0(x)
        buffer_1_1 = self.conv_1_1(x)
        buffer_1_2 = self.conv_1_2(x)
        buffer_1_3 = self.conv_1_3(x)
        buffer_1_4 = self.conv_1_4(x)
        buffer_1_5 = self.conv_1_5(x)
        buffer_1 = torch.cat((buffer_1_0, buffer_1_1, buffer_1_2, buffer_1_3, buffer_1_4, buffer_1_5), 1)
        buffer_1 = self.conv_1(buffer_1)

        buffer_2_0 = self.conv_2_0(x)
        buffer_2_1 = self.conv_2_1(x)
        buffer_2_2 = self.conv_2_2(x)
        buffer_2_3 = self.conv_2_3(x)
        buffer_2_4 = self.conv_2_4(x)
        buffer_2_5 = self.conv_2_5(x)
        buffer_2 = torch.cat((buffer_2_0, buffer_2_1, buffer_2_2, buffer_2_3, buffer_2_4, buffer_2_5), 1)
        buffer_2 = self.conv_2(buffer_2)

        buffer_3_0 = self.conv_3_0(x)
        buffer_3_1 = self.conv_3_1(x)
        buffer_3_2 = self.conv_3_2(x)
        buffer_3_3 = self.conv_3_3(x)
        buffer_3_4 = self.conv_3_4(x)
        buffer_3_5 = self.conv_3_5(x)
        buffer_3 = torch.cat((buffer_3_0, buffer_3_1, buffer_3_2, buffer_3_3, buffer_3_4, buffer_3_5), 1)
        buffer_3 = self.conv_3(buffer_3)
        return x + buffer_1 + buffer_2 + buffer_3

# if __name__ == '__main__':
#     a = MSSTB(in_chans=3, out_chans=32, patch_resolution=64, embed_dim=64, layer=DeformableBasicLayer, n_layers=4,
#               n_scale=2)
#     b = torch.randn((8, 3, 256, 256)).float()
#     print(a(b).shape)

def count_parameters_in_MB(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


if __name__ == '__main__':
    # input: 2 refocused acc: frames acc / event acc
    # shape: (B, 2, H, W), (B, 2, H, W)

    #output :(B, 2, H, W)

    inputf = torch.randn((8, 2, 260, 346)).cuda()
    inpute = torch.randn((8, 2, 260, 346)).cuda()
    net = EF_SAI_Net().cuda()
    size = count_parameters_in_MB(EF_SAI_Net())
    output = net(inpute, inputf)
    print(output.shape)
    print(size)
    # net = torch.nn.DataParallel(net, device_ids=[0])
    # net.load_state_dict(torch.load('/home_ssd/LW/AIOEdata/PreTraining/PreTraining1031_total_hybrid_swinv2/Hybrid_test_stage2.pth'),strict=False)
    # torch.save(net.state_dict(), './EF_SAI_Net.pth')
