import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import model.layers as layers



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


def window_partition(x:Tensor, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows:Tensor, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x:Tensor, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).contiguous().reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in [0,window_size]"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x:Tensor):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        else:
            shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C


        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x:Tensor):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1]
        x = self.proj(x) # [B,embed_dim,Ph,Pw]
        x = x.flatten(2).transpose(1, 2).contiguous()  # [B,Ph*Pw,embed_dim]
        if self.norm is not None: x = self.norm(x)
        return x

class PatchUnEmbed(nn.Module):
    def __init__(self, img_size, patch_size, out_chans, embed_dim):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.out_chans = out_chans
        self.embed_dim = embed_dim

        self.reproj = nn.ConvTranspose2d(embed_dim,out_chans,kernel_size=patch_size, stride=patch_size)

    def forward(self, x:Tensor):
        B = x.shape[0]
        x = x.transpose(1,2).contiguous().view(B,self.embed_dim,self.patches_resolution[0],self.patches_resolution[1])
        x = self.reproj(x)
        return x


# class EmbeddingsWithLearnedPositionalEncoding(nn.Module):
#     def __init__(self, d_model: int, n_vocab: int, max_len: int = 5000):
#         super().__init__()
#         self.linear = nn.Embedding(n_vocab, d_model)
#         self.d_model = d_model
#         self.positional_encodings = nn.Parameter(torch.zeros(max_len, 1, d_model), requires_grad=True)
        
#     def forward(self, x: torch.Tensor):
#         pe = self.positional_encodings[:x.shape[0]]
#         return self.linear(x) * math.sqrt(self.d_model) + pe
    
# class TransformerLayer(nn.Module):
#     def __init__(self, *,d_model: int,self_attn: MultiHeadAttention,src_attn: MultiHeadAttention = None,feed_forward: FeedForward,dropout_prob: float):
#         super().__init__()
#         self.size = d_model
#         self.self_attn = self_attn
#         self.src_attn = src_attn
#         self.feed_forward = feed_forward
#         self.dropout = nn.Dropout(dropout_prob)
#         self.norm_self_attn = nn.LayerNorm([d_model])
#         if self.src_attn is not None:
#             self.norm_src_attn = nn.LayerNorm([d_model])
#         self.norm_ff = nn.LayerNorm([d_model])
#         self.is_save_ff_input = False
        
#         def forward(self, *,
#             x: torch.Tensor,
#             mask: torch.Tensor,
#             src: torch.Tensor = None,
#             src_mask: torch.Tensor = None):
#             z = self.norm_self_attn(x)
#             self_attn = self.self_attn(query=z, key=z, value=z, mask=mask)
#             x = x + self.dropout(self_attn)
#             if src is not None:
#                 z = self.norm_src_attn(x)
#                 attn_src = self.src_attn(query=z, key=src, value=src, mask=src_mask)

class HybridBasicLayer(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, out_chans, embed_dim=64, 
                 depth=4, num_heads=4,window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1):
        super().__init__()
        assert img_size % patch_size == 0
        patch_resolution = img_size//patch_size
        assert patch_resolution % window_size == 0
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0,drop_path_rate,depth)]

        # build layer
        self.skip = nn.Conv2d(in_chans,out_chans,1,1,0)
        self.in_conv = nn.Sequential(*[
            nn.Conv2d(in_chans,embed_dim,3,1,1,bias=False),nn.BatchNorm2d(embed_dim),nn.ReLU(True),
        ])
        # self.cnn = nn.Sequential(*[layers.ResnetBlock(embed_dim,nn.BatchNorm2d,False,False) for _ in range(4)])
        self.patch_embed = PatchEmbed(img_size=img_size,patch_size=patch_size,in_chans=embed_dim,
                                        embed_dim=embed_dim,norm_layer=nn.LayerNorm)
        self.layer = BasicLayer(dim=embed_dim,input_resolution=(patch_resolution,patch_resolution),
                            depth=depth, num_heads=num_heads, window_size=window_size, mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dpr, norm_layer=nn.LayerNorm)
        self.patch_unembed = PatchUnEmbed(img_size=img_size,patch_size=patch_size,out_chans=embed_dim,embed_dim=embed_dim)
        self.out_conv = nn.Sequential(*[
            nn.Conv2d(embed_dim,out_chans,3,1,1,bias=False),nn.BatchNorm2d(out_chans),
        ])
        self.ca = layers.ChannelAttention(embed_dim)
        self.sa = layers.SpatialAttention()

    def forward(self, x:Tensor):
        shortcut = x.clone()
        x = self.in_conv(x)
        # x = self.cnn(x)
        x = self.patch_embed(x)
        x = self.layer(x)
        x = self.patch_unembed(x)
        x = self.out_conv(x)
        x = self.ca(x)*x
        x = self.sa(x)*x
        return F.relu(x + self.skip(shortcut),True)

class DeformableBasicLayer(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, out_chans, embed_dim=64, depth=4, 
                num_heads=4, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1):
        super().__init__()
        assert img_size % patch_size == 0
        patch_resolution = img_size//patch_size
        assert patch_resolution % window_size == 0
        assert embed_dim % num_heads == 0
        assert in_chans % num_heads == 0
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0,drop_path_rate,depth)]

        # build layer
        self.skip = nn.Conv2d(in_chans,out_chans,1,1,0)
        self.in_conv = nn.Sequential(*[
            layers.DCNLayer(in_chans,embed_dim,in_chans//num_heads,3,1,1),nn.BatchNorm2d(embed_dim),nn.ReLU(True),
            # nn.Conv2d(in_chans,embed_dim,3,1,1),nn.BatchNorm2d(embed_dim),nn.ReLU(True),
        ])
        # self.cnn = nn.Sequential(*[layers.ResnetBlock(embed_dim,nn.BatchNorm2d,False,False) for _ in range(4)])
        self.patch_embed = PatchEmbed(img_size=img_size,patch_size=patch_size,in_chans=embed_dim,
                                        embed_dim=embed_dim,norm_layer=nn.LayerNorm)
        self.layer = BasicLayer(dim=embed_dim,input_resolution=(patch_resolution,patch_resolution),
                            depth=depth, num_heads=num_heads, window_size=window_size, mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dpr, norm_layer=nn.LayerNorm)
        self.patch_unembed = PatchUnEmbed(img_size=img_size,patch_size=patch_size,out_chans=embed_dim,embed_dim=embed_dim)
        self.out_conv = nn.Sequential(*[
            layers.DCNLayer(embed_dim,out_chans,embed_dim//num_heads,3,1,1),nn.BatchNorm2d(out_chans),
            # nn.Conv2d(embed_dim,out_chans,3,1,1),nn.BatchNorm2d(embed_dim),nn.ReLU(True),
        ])

    def forward(self, x:Tensor):
        shortcut = x.clone()
        x = self.in_conv(x)
        # x = self.cnn(x)
        x = self.patch_embed(x)
        x = self.layer(x)
        x = self.patch_unembed(x)
        x = self.out_conv(x)
        return F.relu(x + self.skip(shortcut),True)


            
class MSSTB(nn.Module):
    def __init__(self,in_chans,out_chans,patch_resolution,embed_dim,layer,n_layers,n_scale):
        super().__init__()
        assert layer in [HybridBasicLayer,DeformableBasicLayer]
        
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
        # add HSTB layers
        self.hstb = []
        for i in range(n_layers):
            self.hstb += [layer(patch_resolution,1,embed_dim*ratio,embed_dim*ratio,embed_dim*ratio)]
        self.hstb = nn.Sequential(*self.hstb)
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

    def forward(self, x:Tensor):
        x = self.init_conv(x)
        x = self.down(x)
        x = self.hstb(x)
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
    
if __name__ == '__main__':
    a = MSSTB(in_chans=3,out_chans=32,patch_resolution=64,embed_dim=64,layer=DeformableBasicLayer,n_layers=4,n_scale=2)
    b = torch.randn((8,3,256,256)).float()
    print(a(b).shape)