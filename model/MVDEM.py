import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from einops import rearrange, repeat


class MultiViewUniMatch(nn.Module):
    def __init__(
        self,
        num_scales=1,
        feature_channels=64,
        upsample_factor=2,
        lowest_feature_resolution=8,
        num_head=1,
        ffn_dim_expansion=2,
        num_transformer_layers=1,
        num_depth_candidates=128,
        grid_sample_disable_cudnn=False,
    ):
        super(MultiViewUniMatch, self).__init__()

        # CNN
        self.feature_channels = feature_channels
        self.num_scales = num_scales
        self.lowest_feature_resolution = lowest_feature_resolution
        self.upsample_factor = upsample_factor

        # cost volume
        self.num_depth_candidates = num_depth_candidates

        # CNN
        self.backbone = CNNEncoder(
            output_dim=feature_channels,
            num_output_scales=num_scales,
            downsample_factor=upsample_factor,
            lowest_scale=lowest_feature_resolution,
            return_all_scales=True,
        )

        # Transformer
        self.transformer = MultiViewFeatureTransformer(
            num_layers=num_transformer_layers,
            d_model=feature_channels,
            nhead=num_head,
            ffn_dim_expansion=ffn_dim_expansion,
        )

        if self.num_scales > 1:
            # generate multi-scale features
            self.mv_pyramid = ViTFeaturePyramid(
                in_channels=128, scale_factors=[2**i for i in range(self.num_scales)]
            )

        self.grid_sample_disable_cudnn = grid_sample_disable_cudnn

    def normalize_images(self, images):
        """Normalize image to match the pretrained UniMatch model.
        images: (B, V, C, H, W)
        """
        shape = [*[1] * (images.dim() - 3), 3, 1, 1]
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(*shape).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(*shape).to(images.device)

        return (images - mean) / std

    def extract_feature(self, images):
        # images: [B, V, C, H, W]
        b, v = images.shape[:2]
        concat = rearrange(images, "b v c h w -> (b v) c h w")
        # list of [BV, C, H, W], resolution from high to low
        features = self.backbone(concat)
        # reverse: resolution from low to high
        features = features[::-1]

        return features

    def forward(
        self,
        images,
        attn_splits_list=[2],
        intrinsics=None,
        min_depth=1.0 / 0.1,  # inverse depth range
        max_depth=1.0 / 20,
        num_depth_candidates=128,
        extrinsics=None,
        nn_matrix=None,
        **kwargs,
    ):

        depth_preds = []
        match_probs = []

        # first normalize images
        images = self.normalize_images(images)
        b, v, _, ori_h, ori_w = images.shape

        intrinsics = intrinsics.clone()

        # max_depth, min_depth: [B, V] -> [BV]
        max_depth = max_depth.view(-1)
        min_depth = min_depth.view(-1)

        # list of features, resolution low to high
        # list of [BV, C, H, W]
        features_list_cnn = self.extract_feature(images)
        features_list_cnn_all_scales = features_list_cnn
        features_list_cnn = features_list_cnn[: self.num_scales]

        # mv transformer features
        # add position to features
        attn_splits = attn_splits_list[0]

        # [BV, C, H, W]
        features_cnn_pos = mv_feature_add_position(
            features_list_cnn[0], attn_splits, self.feature_channels
        )

        # list of [B, C, H, W]
        features_list = list(
            torch.unbind(
                rearrange(features_cnn_pos, "(b v) c h w -> b v c h w", b=b, v=v), dim=1
            )
        )
        features_list_mv = self.transformer(
            features_list,
            attn_num_splits=attn_splits,
            nn_matrix=nn_matrix,
        )

        features_mv = torch.stack(features_list_mv, dim=1)  # [BV, C, H, W]

        # if self.num_scales > 1:
        #     # multi-scale mv features: resolution from low to high
        #     # list of [BV, C, H, W]
        #     features_list_mv = self.mv_pyramid(features_mv)
        # else:
        #     features_list_mv = [features_mv]

        # depth = None

        # for scale_idx in range(self.num_scales):
        #     downsample_factor = self.upsample_factor * (
        #         2 ** (self.num_scales - 1 - scale_idx)
        #     )

        #     # scale intrinsics
        #     intrinsics_curr = intrinsics.clone()  # [B, V, 3, 3]
        #     intrinsics_curr[:, :, :2] = intrinsics_curr[:, :, :2] / downsample_factor

        #     # build cost volume
        #     features_mv = features_list_mv[scale_idx]  # [BV, C, H, W]

        #     # list of [B, C, H, W]
        #     features_mv_curr = list(
        #         torch.unbind(
        #             rearrange(features_mv, "(b v) c h w -> b v c h w", b=b, v=v), dim=1
        #         )
        #     )

        #     intrinsics_curr = list(
        #         torch.unbind(intrinsics_curr, dim=1)
        #     )  # list of [B, 3, 3]
        #     extrinsics_curr = list(torch.unbind(extrinsics, dim=1))  # list of [B, 4, 4]

        #     # ref: [BV, C, H, W], [BV, 3, 3], [BV, 4, 4]
        #     # tgt: [BV, V-1, C, H, W], [BV, V-1, 3, 3], [BV, V-1, 4, 4]
        #     (
        #         ref_features,
        #         ref_intrinsics,
        #         ref_extrinsics,
        #         tgt_features,
        #         tgt_intrinsics,
        #         tgt_extrinsics,
        #     ) = batch_features_camera_parameters(
        #         features_mv_curr,
        #         intrinsics_curr,
        #         extrinsics_curr,
        #         nn_matrix=nn_matrix,
        #     )

            # b_new, _, c, h, w = tgt_features.size()

            # # relative pose
            # # extrinsics: c2w
            # pose_curr = torch.matmul(
            #     tgt_extrinsics.inverse(), ref_extrinsics.unsqueeze(1)
            # )  # [BV, V-1, 4, 4]

            # if scale_idx > 0:
            #     # 2x upsample depth
            #     assert depth is not None
            #     depth = F.interpolate(
            #         depth, scale_factor=2, mode="bilinear", align_corners=True
            #     ).detach()

            # num_depth_candidates = self.num_depth_candidates // (4**scale_idx)

            # # generate depth candidates
            # if scale_idx == 0:
            #     # min_depth, max_depth: [BV]
            #     depth_interval = (max_depth - min_depth) / (
            #         self.num_depth_candidates - 1
            #     )  # [BV]

            #     linear_space = (
            #         torch.linspace(0, 1, num_depth_candidates)
            #         .type_as(features_list_cnn[0])
            #         .view(1, num_depth_candidates, 1, 1)
            #     )  # [1, D, 1, 1]

            #     depth_candidates = min_depth.view(-1, 1, 1, 1) + linear_space * (
            #         max_depth - min_depth
            #     ).view(
            #         -1, 1, 1, 1
            #     )  # [BV, D, 1, 1]
            # else:
            #     # half interval each scale
            #     depth_interval = (
            #         (max_depth - min_depth)
            #         / (self.num_depth_candidates - 1)
            #         / (2**scale_idx)
            #     )  # [BV]
            #     # [BV, 1, 1, 1]
            #     depth_interval = depth_interval.view(-1, 1, 1, 1)

            #     # [BV, 1, H, W]
            #     depth_range_min = (
            #         depth - depth_interval * (num_depth_candidates // 2)
            #     ).clamp(min=min_depth.view(-1, 1, 1, 1))
            #     depth_range_max = (
            #         depth + depth_interval * (num_depth_candidates // 2 - 1)
            #     ).clamp(max=max_depth.view(-1, 1, 1, 1))

            #     linear_space = (
            #         torch.linspace(0, 1, num_depth_candidates)
            #         .type_as(features_list_cnn[0])
            #         .view(1, num_depth_candidates, 1, 1)
            #     )  # [1, D, 1, 1]
            #     depth_candidates = depth_range_min + linear_space * (
            #         depth_range_max - depth_range_min
            #     )  # [BV, D, H, W]

            # if scale_idx == 0:
            #     # [BV*(V-1), D, H, W]
            #     depth_candidates_curr = (
            #         depth_candidates.unsqueeze(1)
            #         .repeat(1, tgt_features.size(1), 1, h, w)
            #         .view(-1, num_depth_candidates, h, w)
            #     )
            # else:
            #     depth_candidates_curr = (
            #         depth_candidates.unsqueeze(1)
            #         .repeat(1, tgt_features.size(1), 1, 1, 1)
            #         .view(-1, num_depth_candidates, h, w)
            #     )

            # intrinsics_input = torch.stack(intrinsics_curr, dim=1).view(
            #     -1, 3, 3
            # )  # [BV, 3, 3]
            # intrinsics_input = intrinsics_input.unsqueeze(1).repeat(
            #     1, tgt_features.size(1), 1, 1
            # )  # [BV, V-1, 3, 3]

            # warped_tgt_features = warp_with_pose_depth_candidates(
            #     rearrange(tgt_features, "b v ... -> (b v) ..."),
            #     rearrange(intrinsics_input, "b v ... -> (b v) ..."),
            #     rearrange(pose_curr, "b v ... -> (b v) ..."),
            #     1.0 / depth_candidates_curr,  # convert inverse depth to depth
            #     grid_sample_disable_cudnn=self.grid_sample_disable_cudnn,
            # )  # [BV*(V-1), C, D, H, W]

            # # ref: [BV, C, H, W]
            # # warped: [BV*(V-1), C, D, H, W] -> [BV, V-1, C, D, H, W]
            # warped_tgt_features = rearrange(
            #     warped_tgt_features,
            #     "(b v) ... -> b v ...",
            #     b=b_new,
            #     v=tgt_features.size(1),
            # )
            # # [BV, V-1, D, H, W] -> [BV, D, H, W]
            # # average cross other views
            # cost_volume = (
            #     (ref_features.unsqueeze(-3).unsqueeze(1) * warped_tgt_features).sum(2)
            #     / (c**0.5)
            # ).mean(1)

        return features_mv


class CNNEncoder(nn.Module):
    def __init__(
        self,
        output_dim=128,
        norm_layer=nn.InstanceNorm2d,
        num_output_scales=1,
        return_quarter=False,  # return 1/4 resolution feature
        lowest_scale=8,  # lowest resolution, 1/8 or 1/4
        return_all_scales=False,
        **kwargs,
    ):
        super(CNNEncoder, self).__init__()
        self.num_scales = num_output_scales
        self.return_quarter = return_quarter
        self.lowest_scale = lowest_scale
        self.return_all_scales = return_all_scales

        feature_dims = [64, 96, 128]

        self.conv1 = nn.Conv2d(
            3, feature_dims[0], kernel_size=7, stride=2, padding=3, bias=False
        )  # 1/2
        self.norm1 = norm_layer(feature_dims[0])
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = feature_dims[0]
        self.layer1 = self._make_layer(
            feature_dims[0], stride=1, norm_layer=norm_layer
        )  # 1/2

        if self.lowest_scale == 4:
            stride = 1
        else:
            stride = 2
        self.layer2 = self._make_layer(
            feature_dims[1], stride=stride, norm_layer=norm_layer
        )  # 1/2 or 1/4

        # lowest resolution 1/4 or 1/8
        self.layer3 = self._make_layer(
            feature_dims[2],
            stride=2,
            norm_layer=norm_layer,
        )  # 1/4 or 1/8

        self.conv2 = nn.Conv2d(feature_dims[2], output_dim, 1, 1, 0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1, dilation=1, norm_layer=nn.InstanceNorm2d):
        layer1 = ResidualBlock(
            self.in_planes, dim, norm_layer=norm_layer, stride=stride, dilation=dilation
        )
        layer2 = ResidualBlock(
            dim, dim, norm_layer=norm_layer, stride=1, dilation=dilation
        )

        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        output_all_scales = []
        output = []
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)  # 1/2

        if self.return_all_scales:
            output_all_scales.append(x)

        if self.num_scales >= 3:
            output.append(x)

        x = self.layer2(x)  # 1/2 or 1/4
        if self.return_quarter:
            output.append(x)

        if self.return_all_scales:
            output_all_scales.append(x)

        if self.num_scales >= 2:
            output.append(x)

        x = self.layer3(x)  # 1/4 or 1/8
        x = self.conv2(x)

        if self.return_all_scales:
            output_all_scales.append(x)

        if self.return_all_scales:
            return output_all_scales

        if self.return_quarter:
            output.append(x)
            return output

        if self.num_scales >= 1:
            output.append(x)
            return output

        out = [x]

        return out
    

class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_planes,
        planes,
        norm_layer=nn.InstanceNorm2d,
        stride=1,
        dilation=1,
    ):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            dilation=dilation,
            padding=dilation,
            stride=stride,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            dilation=dilation,
            padding=dilation,
            bias=False,
        )
        self.relu = nn.ReLU(inplace=True)

        self.norm1 = norm_layer(planes)
        self.norm2 = norm_layer(planes)
        if not stride == 1 or in_planes != planes:
            self.norm3 = norm_layer(planes)

        if stride == 1 and in_planes == planes:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3
            )

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)
    

class MultiViewFeatureTransformer(nn.Module):
    def __init__(
        self,
        num_layers=6,
        d_model=128,
        nhead=1,
        attention_type="swin",
        ffn_dim_expansion=4,
        add_per_view_attn=False,
        no_cross_attn=False,
        **kwargs,
    ):
        super(MultiViewFeatureTransformer, self).__init__()

        self.attention_type = attention_type

        self.d_model = d_model
        self.nhead = nhead

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    nhead=nhead,
                    attention_type=attention_type,
                    ffn_dim_expansion=ffn_dim_expansion,
                    with_shift=(
                        True if attention_type == "swin" and i % 2 == 1 else False
                    ),
                    add_per_view_attn=add_per_view_attn,
                    no_cross_attn=no_cross_attn,
                )
                for i in range(num_layers)
            ]
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # zero init layers beyond 6
        if num_layers > 6:
            for i in range(6, num_layers):
                self.layers[i].self_attn.norm1.weight.data.zero_()
                self.layers[i].self_attn.norm1.bias.data.zero_()
                self.layers[i].cross_attn_ffn.norm2.weight.data.zero_()
                self.layers[i].cross_attn_ffn.norm2.bias.data.zero_()

    def forward(
        self,
        multi_view_features,
        attn_num_splits=None,
        **kwargs,
    ):
        nn_matrix = kwargs.pop("nn_matrix", None)

        # multi_view_features: list of [B, C, H, W]
        b, c, h, w = multi_view_features[0].shape
        assert self.d_model == c

        num_views = len(multi_view_features)

        if self.attention_type == "swin" and attn_num_splits > 1:
            # global and refine use different number of splits
            window_size_h = h // attn_num_splits
            window_size_w = w // attn_num_splits

            # compute attn mask once
            shifted_window_attn_mask = generate_shift_window_attn_mask(
                input_resolution=(h, w),
                window_size_h=window_size_h,
                window_size_w=window_size_w,
                shift_size_h=window_size_h // 2,
                shift_size_w=window_size_w // 2,
                device=multi_view_features[0].device,
            )  # [K*K, H/K*W/K, H/K*W/K]
        else:
            shifted_window_attn_mask = None

        # [N*B, C, H, W], [N*B, N-1, C, H, W]
        concat0, concat1 = batch_features(multi_view_features, nn_matrix=nn_matrix)
        concat0 = concat0.reshape(num_views * b, c, -1).permute(
            0, 2, 1
        )  # [N*B, H*W, C]
        c1_v = num_views - 1 if nn_matrix is None else nn_matrix.shape[-1] - 1
        concat1 = concat1.reshape(num_views * b, c1_v, c, -1).permute(
            0, 1, 3, 2
        )  # [N*B, N-1, H*W, C]

        for i, layer in enumerate(self.layers):
            concat0 = layer(
                concat0,
                concat1,
                height=h,
                width=w,
                shifted_window_attn_mask=shifted_window_attn_mask,
                attn_num_splits=attn_num_splits,
            )

            if i < len(self.layers) - 1:
                # list of features
                features = list(concat0.chunk(chunks=num_views, dim=0))
                # [N*B, H*W, C], [N*B, N-1, H*W, C]
                concat0, concat1 = batch_features(features, nn_matrix=nn_matrix)

        features = concat0.chunk(chunks=num_views, dim=0)
        features = [
            f.view(b, h, w, c).permute(0, 3, 1, 2).contiguous() for f in features
        ]

        return features
    

def generate_shift_window_attn_mask(
    input_resolution,
    window_size_h,
    window_size_w,
    shift_size_h,
    shift_size_w,
    device=torch.device("cuda"),
):
    # Ref: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
    # calculate attention mask for SW-MSA
    h, w = input_resolution
    img_mask = torch.zeros((1, h, w, 1)).to(device)  # 1 H W 1
    h_slices = (
        slice(0, -window_size_h),
        slice(-window_size_h, -shift_size_h),
        slice(-shift_size_h, None),
    )
    w_slices = (
        slice(0, -window_size_w),
        slice(-window_size_w, -shift_size_w),
        slice(-shift_size_w, None),
    )
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = split_feature(
        img_mask, num_splits=input_resolution[-1] // window_size_w, channel_last=True
    )

    mask_windows = mask_windows.view(-1, window_size_h * window_size_w)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        attn_mask == 0, float(0.0)
    )

    return attn_mask


def split_feature(
    feature,
    num_splits=2,
    channel_last=False,
):
    if channel_last:  # [B, H, W, C]
        b, h, w, c = feature.size()
        assert h % num_splits == 0 and w % num_splits == 0

        b_new = b * num_splits * num_splits
        h_new = h // num_splits
        w_new = w // num_splits

        feature = (
            feature.view(b, num_splits, h // num_splits, num_splits, w // num_splits, c)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(b_new, h_new, w_new, c)
        )  # [B*K*K, H/K, W/K, C]
    else:  # [B, C, H, W]
        b, c, h, w = feature.size()
        assert h % num_splits == 0 and w % num_splits == 0

        b_new = b * num_splits * num_splits
        h_new = h // num_splits
        w_new = w // num_splits

        feature = (
            feature.view(b, c, num_splits, h // num_splits, num_splits, w // num_splits)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(b_new, c, h_new, w_new)
        )  # [B*K*K, C, H/K, W/K]

    return feature
    

class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=1,
        attention_type="swin",
        no_ffn=False,
        ffn_dim_expansion=4,
        with_shift=False,
        add_per_view_attn=False,
        **kwargs,
    ):
        super(TransformerLayer, self).__init__()

        self.dim = d_model
        self.nhead = nhead
        self.attention_type = attention_type
        self.no_ffn = no_ffn
        self.add_per_view_attn = add_per_view_attn

        self.with_shift = with_shift

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.merge = nn.Linear(d_model, d_model, bias=False)

        self.norm1 = nn.LayerNorm(d_model)

        # no ffn after self-attn, with ffn after cross-attn
        if not self.no_ffn:
            in_channels = d_model * 2
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, in_channels * ffn_dim_expansion, bias=False),
                nn.GELU(),
                nn.Linear(in_channels * ffn_dim_expansion, d_model, bias=False),
            )

            self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        source,
        target,
        height=None,
        width=None,
        shifted_window_attn_mask=None,
        attn_num_splits=None,
        **kwargs,
    ):
        if "attn_type" in kwargs:
            attn_type = kwargs["attn_type"]
        else:
            attn_type = self.attention_type

        # source, target: [B, L, C] for 2-view
        # for multi-view cross-attention, source: [B, L, C], target: [B, N-1, L, C]
        query, key, value = source, target, target

        # single-head attention
        query = self.q_proj(query)  # [B, L, C]
        key = self.k_proj(key)  # [B, L, C] or [B, N-1, L, C]
        value = self.v_proj(value)  # [B, L, C] or [B, N-1, L, C]

        if attn_type == "swin" and attn_num_splits > 1:
            message = single_head_split_window_attention(
                        query,
                        key,
                        value,
                        num_splits=attn_num_splits,
                        with_shift=self.with_shift,
                        h=height,
                        w=width,
                        attn_mask=shifted_window_attn_mask,
                    )

        message = self.merge(message)  # [B, L, C]
        message = self.norm1(message)

        if not self.no_ffn:
            message = self.mlp(torch.cat([source, message], dim=-1))
            message = self.norm2(message)

        return source + message
    

def single_head_split_window_attention(
    q,
    k,
    v,
    num_splits=1,
    with_shift=False,
    h=None,
    w=None,
    attn_mask=None,
):
    # Ref: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
    # q, k, v: [B, L, C] for 2-view
    # for multi-view cross-attention, q: [B, L, C], k, v: [B, N-1, L, C]

    # multi(>2)-view corss-attention
    if not (q.dim() == k.dim() == v.dim() == 3):
        assert k.dim() == v.dim() == 4
        assert h is not None and w is not None
        assert q.size(1) == h * w

        m = k.size(1)  # m + 1 is num of views

        b, _, c = q.size()

        b_new = b * num_splits * num_splits

        window_size_h = h // num_splits
        window_size_w = w // num_splits

        q = q.view(b, h, w, c)  # [B, H, W, C]
        k = k.view(b, m, h, w, c)  # [B, N-1, H, W, C]
        v = v.view(b, m, h, w, c)

        scale_factor = c**0.5

        if with_shift:
            assert attn_mask is not None  # compute once
            shift_size_h = window_size_h // 2
            shift_size_w = window_size_w // 2

            q = torch.roll(q, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))
            k = torch.roll(k, shifts=(-shift_size_h, -shift_size_w), dims=(2, 3))
            v = torch.roll(v, shifts=(-shift_size_h, -shift_size_w), dims=(2, 3))

        q = split_feature(
            q, num_splits=num_splits, channel_last=True
        )  # [B*K*K, H/K, W/K, C]
        k = split_feature(
            k.permute(0, 2, 3, 4, 1).reshape(b, h, w, -1),
            num_splits=num_splits,
            channel_last=True,
        )  # [B*K*K, H/K, W/K, C*(N-1)]
        v = split_feature(
            v.permute(0, 2, 3, 4, 1).reshape(b, h, w, -1),
            num_splits=num_splits,
            channel_last=True,
        )  # [B*K*K, H/K, W/K, C*(N-1)]

        k = (
            k.view(b_new, h // num_splits, w // num_splits, c, m)
            .permute(0, 3, 1, 2, 4)
            .reshape(b_new, c, -1)
        )  # [B*K*K, C, H/K*W/K*(N-1)]
        v = (
            v.view(b_new, h // num_splits, w // num_splits, c, m)
            .permute(0, 1, 2, 4, 3)
            .reshape(b_new, -1, c)
        )  # [B*K*K, H/K*W/K*(N-1), C]

        scores = (
            torch.matmul(q.view(b_new, -1, c), k) / scale_factor
        )  # [B*K*K, H/K*W/K, H/K*W/K*(N-1)]

        if with_shift:
            scores += attn_mask.repeat(b, 1, m)

        attn = torch.softmax(scores, dim=-1)

        out = torch.matmul(attn, v)  # [B*K*K, H/K*W/K, C]

        out = merge_splits(
            out.view(b_new, h // num_splits, w // num_splits, c),
            num_splits=num_splits,
            channel_last=True,
        )  # [B, H, W, C]

        # shift back
        if with_shift:
            out = torch.roll(out, shifts=(shift_size_h, shift_size_w), dims=(1, 2))

        out = out.view(b, -1, c)
    else:
        # 2-view self-attention or cross-attention
        assert q.dim() == k.dim() == v.dim() == 3

        assert h is not None and w is not None
        assert q.size(1) == h * w

        b, _, c = q.size()

        b_new = b * num_splits * num_splits

        window_size_h = h // num_splits
        window_size_w = w // num_splits

        q = q.view(b, h, w, c)  # [B, H, W, C]
        k = k.view(b, h, w, c)
        v = v.view(b, h, w, c)

        scale_factor = c**0.5

        if with_shift:
            assert attn_mask is not None  # compute once
            shift_size_h = window_size_h // 2
            shift_size_w = window_size_w // 2

            q = torch.roll(q, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))
            k = torch.roll(k, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))
            v = torch.roll(v, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))

        q = split_feature(
            q, num_splits=num_splits, channel_last=True
        )  # [B*K*K, H/K, W/K, C]
        k = split_feature(k, num_splits=num_splits, channel_last=True)
        v = split_feature(v, num_splits=num_splits, channel_last=True)

        scores = (
            torch.matmul(q.view(b_new, -1, c), k.view(b_new, -1, c).permute(0, 2, 1))
            / scale_factor
        )  # [B*K*K, H/K*W/K, H/K*W/K]

        if with_shift:
            scores += attn_mask.repeat(b, 1, 1)

        attn = torch.softmax(scores, dim=-1)

        out = torch.matmul(attn, v.view(b_new, -1, c))  # [B*K*K, H/K*W/K, C]

        out = merge_splits(
            out.view(b_new, h // num_splits, w // num_splits, c),
            num_splits=num_splits,
            channel_last=True,
        )  # [B, H, W, C]

        # shift back
        if with_shift:
            out = torch.roll(out, shifts=(shift_size_h, shift_size_w), dims=(1, 2))

        out = out.view(b, -1, c)

    return out

def merge_splits(
    splits,
    num_splits=2,
    channel_last=False,
):
    if channel_last:  # [B*K*K, H/K, W/K, C]
        b, h, w, c = splits.size()
        new_b = b // num_splits // num_splits

        splits = splits.view(new_b, num_splits, num_splits, h, w, c)
        merge = (
            splits.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(new_b, num_splits * h, num_splits * w, c)
        )  # [B, H, W, C]
    else:  # [B*K*K, C, H/K, W/K]
        b, c, h, w = splits.size()
        new_b = b // num_splits // num_splits

        splits = splits.view(new_b, num_splits, num_splits, c, h, w)
        merge = (
            splits.permute(0, 3, 1, 4, 2, 5)
            .contiguous()
            .view(new_b, c, num_splits * h, num_splits * w)
        )  # [B, C, H, W]

    return merge


class TransformerBlock(nn.Module):
    """self attention + cross attention + FFN"""

    def __init__(
        self,
        d_model=256,
        nhead=1,
        attention_type="swin",
        ffn_dim_expansion=4,
        with_shift=False,
        add_per_view_attn=False,
        no_cross_attn=False,
        **kwargs,
    ):
        super(TransformerBlock, self).__init__()

        self.no_cross_attn = no_cross_attn

        if no_cross_attn:
            self.self_attn = TransformerLayer(
                d_model=d_model,
                nhead=nhead,
                attention_type=attention_type,
                ffn_dim_expansion=ffn_dim_expansion,
                with_shift=with_shift,
                add_per_view_attn=add_per_view_attn,
            )
        else:
            self.self_attn = TransformerLayer(
                d_model=d_model,
                nhead=nhead,
                attention_type=attention_type,
                no_ffn=True,
                ffn_dim_expansion=ffn_dim_expansion,
                with_shift=with_shift,
            )

            self.cross_attn_ffn = TransformerLayer(
                d_model=d_model,
                nhead=nhead,
                attention_type=attention_type,
                ffn_dim_expansion=ffn_dim_expansion,
                with_shift=with_shift,
                add_per_view_attn=add_per_view_attn,
            )

    def forward(
        self,
        source,
        target,
        height=None,
        width=None,
        shifted_window_attn_mask=None,
        attn_num_splits=None,
        **kwargs,
    ):
        # source, target: [B, L, C]
        # self attention
        source = self.self_attn(
            source,
            source,
            height=height,
            width=width,
            shifted_window_attn_mask=shifted_window_attn_mask,
            attn_num_splits=attn_num_splits,
            **kwargs,
        )

        if self.no_cross_attn:
            return source

        # cross attention and ffn
        source = self.cross_attn_ffn(
            source,
            target,
            height=height,
            width=width,
            shifted_window_attn_mask=shifted_window_attn_mask,
            attn_num_splits=attn_num_splits,
            **kwargs,
        )

        return source


def batch_features(features, nn_matrix=None):
    # construct inputs to multi-view transformer in batch
    # features: list of [B, C, H, W] or [B, H*W, C]

    # query, key and value for transformer
    q = []
    kv = []

    num_views = len(features)
    if nn_matrix is not None:
        # (b v c h w) or (b v hw c)
        features_tensor = torch.stack(features, dim=1)

    for i in range(num_views):
        x = features.copy()
        q.append(x.pop(i))  # [B, C, H, W] or [B, H*W, C]

        # [B, N-1, C, H, W] or [B, N-1, H*W, C]
        if nn_matrix is not None:
            # select views based on the provided nn matrix
            if features_tensor.dim() == 5:
                c, h, w = features_tensor.shape[-3:]
                index = repeat(nn_matrix[:, i, 1:], "b v -> b v c h w", c=c, h=h, w=w)
            elif features_tensor.dim() == 4:
                hw, c = features_tensor.shape[-2:]
                index = repeat(nn_matrix[:, i, 1:], "b v -> b v hw c", hw=hw, c=c)

            kv_x = torch.gather(features_tensor, dim=1, index=index)
        else:
            kv_x = torch.stack(x, dim=1)
        kv.append(kv_x)

    q = torch.cat(q, dim=0)  # [N*B, C, H, W] or [N*B, H*W, C]
    kv = torch.cat(kv, dim=0)  # [N*B, N-1, C, H, W] or [N*B, N-1, H*W, C]

    return q, kv

class ViTFeaturePyramid(nn.Module):
    """
    This module implements SimpleFeaturePyramid in :paper:`vitdet`.
    It creates pyramid features built on top of the input feature map.
    """

    def __init__(
        self,
        in_channels,
        scale_factors,
    ):
        """
        Args:
            scale_factors (list[float]): list of scaling factors to upsample or downsample
                the input features for creating pyramid features.
        """
        super(ViTFeaturePyramid, self).__init__()

        self.scale_factors = scale_factors

        out_dim = dim = in_channels
        self.stages = nn.ModuleList()
        for idx, scale in enumerate(scale_factors):
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                ]
                out_dim = dim // 4
            elif scale == 2.0:
                layers = [nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)]
                out_dim = dim // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            if scale != 1.0:
                layers.extend(
                    [
                        nn.GELU(),
                        nn.Conv2d(out_dim, out_dim, 3, 1, 1),
                    ]
                )
            layers = nn.Sequential(*layers)

            self.stages.append(layers)

    def forward(self, x):
        results = []

        for stage in self.stages:
            results.append(stage(x))

        return results


def mv_feature_add_position(features, attn_splits, feature_channels):
    pos_enc = PositionEmbeddingSine(num_pos_feats=feature_channels // 2)

    assert features.dim() == 4  # [B*V, C, H, W]

    if attn_splits > 1:  # add position in splited window
        features_splits = split_feature(features, num_splits=attn_splits)
        position = pos_enc(features_splits)
        features_splits = features_splits + position
        features = merge_splits(features_splits, num_splits=attn_splits)
    else:
        position = pos_enc(features)
        features = features + position

    return features

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        # x = tensor_list.tensors  # [B, C, H, W]
        # mask = tensor_list.mask  # [B, H, W], input with padding, valid as 0
        b, c, h, w = x.size()
        mask = torch.ones((b, h, w), device=x.device)  # [B, H, W]
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
    
def batch_features_camera_parameters(
    features,
    intrinsics,
    extrinsics,
    nn_matrix=None,
    no_batch=False,
):
    # construct inputs for warping with plane-sweep stereo
    # features: list of [B, C, H, W]
    # intrinsics: list of [B, 3, 3]
    # extrinsics: list of [B, 4, 4]

    assert (
        features[0].dim() == 4 and intrinsics[0].dim() == 3 and extrinsics[0].dim() == 3
    )
    assert intrinsics[0].size(-1) == intrinsics[0].size(-2) == 3
    assert extrinsics[0].size(-1) == extrinsics[0].size(-2) == 4

    # query, key and value for transformer
    q = []
    q_intrinsics = []
    q_extrinsics = []
    kv = []
    kv_intrinsics = []
    kv_extrinsics = []

    num_views = len(features)
    if nn_matrix is not None:
        features_tensor = torch.stack(features, dim=1)  # [B, V, C, H, W]
        intrinsics_tensor = torch.stack(intrinsics, dim=1)  # [B, V, 3, 3]
        extrinsics_tensor = torch.stack(extrinsics, dim=1)  # [B, V, 4, 4]

        num_selected_views = nn_matrix.size(-1) - 1
    else:
        num_selected_views = num_views - 1

    for i in range(num_views):
        # features
        x = features.copy()
        q.append(x.pop(i))  # [B, C, H, W]

        # camera
        y = intrinsics.copy()
        q_intrinsics.append(y.pop(i))
        z = extrinsics.copy()
        q_extrinsics.append(z.pop(i))

        # [B, V-1, C, H, W]
        if nn_matrix is not None:
            # select views based on the provided nn matrix
            if features_tensor.dim() == 5:
                c, h, w = features_tensor.shape[-3:]
                index = repeat(nn_matrix[:, i, 1:], "b v -> b v c h w", c=c, h=h, w=w)
            elif features_tensor.dim() == 4:
                hw, c = features_tensor.shape[-2:]
                index = repeat(nn_matrix[:, i, 1:], "b v -> b v hw c", hw=hw, c=c)

            kv_x = torch.gather(features_tensor, dim=1, index=index)

            # select intrinsics and extrinsics
            index = repeat(nn_matrix[:, i, 1:], "b v -> b v 3 3")
            kv_y_intrinsics = torch.gather(intrinsics_tensor, dim=1, index=index)

            index = repeat(nn_matrix[:, i, 1:], "b v -> b v 4 4")
            kv_z_extrinsics = torch.gather(extrinsics_tensor, dim=1, index=index)

        else:
            kv_x = torch.stack(x, dim=1)
            kv_y_intrinsics = torch.stack(y, dim=1)
            kv_z_extrinsics = torch.stack(z, dim=1)

        kv.append(kv_x)
        kv_intrinsics.append(kv_y_intrinsics)
        kv_extrinsics.append(kv_z_extrinsics)

    if no_batch:
        # list of [B, C, H, W]
        return q, q_intrinsics, q_extrinsics, kv, kv_intrinsics, kv_extrinsics

    c, h, w = q[0].shape[1:]

    q = torch.stack(q, dim=1).view(-1, c, h, w)  # [BV, C, H, W]
    q_intrinsics = torch.stack(q_intrinsics, dim=1).view(-1, 3, 3)  # [BV, 3, 3]
    q_extrinsics = torch.stack(q_extrinsics, dim=1).view(-1, 4, 4)  # [BV, 4, 4]
    kv = torch.stack(kv, dim=1).view(
        -1, num_selected_views, c, h, w
    )  # [BV, V-1, C, H, W]
    kv_intrinsics = torch.stack(kv_intrinsics, dim=1).view(
        -1, num_selected_views, 3, 3
    )  # [BV, V-1, 3, 3]
    kv_extrinsics = torch.stack(kv_extrinsics, dim=1).view(
        -1, num_selected_views, 4, 4
    )  # [BV, V-1, 4, 4]

    return q, q_intrinsics, q_extrinsics, kv, kv_intrinsics, kv_extrinsics


def warp_with_pose_depth_candidates(
    feature1,
    intrinsics,
    pose,
    depth,
    clamp_min_depth=1e-3,
    grid_sample_disable_cudnn=False,
):
    """
    feature1: [B, C, H, W]
    intrinsics: [B, 3, 3]
    pose: [B, 4, 4]
    depth: [B, D, H, W]
    """

    assert intrinsics.size(1) == intrinsics.size(2) == 3
    assert pose.size(1) == pose.size(2) == 4
    assert depth.dim() == 4

    b, d, h, w = depth.size()
    c = feature1.size(1)

    with torch.no_grad():
        # pixel coordinates
        grid = coords_grid(
            b, h, w, homogeneous=True, device=depth.device
        )  # [B, 3, H, W]
        # back project to 3D and transform viewpoint
        points = torch.inverse(intrinsics).bmm(grid.view(b, 3, -1))  # [B, 3, H*W]
        points = torch.bmm(pose[:, :3, :3], points).unsqueeze(2).repeat(
            1, 1, d, 1
        ) * depth.view(
            b, 1, d, h * w
        )  # [B, 3, D, H*W]
        points = points + pose[:, :3, -1:].unsqueeze(-1)  # [B, 3, D, H*W]
        # reproject to 2D image plane
        points = torch.bmm(intrinsics, points.view(b, 3, -1)).view(
            b, 3, d, h * w
        )  # [B, 3, D, H*W]
        pixel_coords = points[:, :2] / points[:, -1:].clamp(
            min=clamp_min_depth
        )  # [B, 2, D, H*W]

        # normalize to [-1, 1]
        x_grid = 2 * pixel_coords[:, 0] / (w - 1) - 1
        y_grid = 2 * pixel_coords[:, 1] / (h - 1) - 1

        grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, D, H*W, 2]

    # sample features
    # ref: https://github.com/pytorch/pytorch/issues/88380
    with torch.backends.cudnn.flags(enabled=not grid_sample_disable_cudnn):
        warped_feature = F.grid_sample(
            feature1,
            grid.view(b, d * h, w, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).view(
            b, c, d, h, w
        )  # [B, C, D, H, W]

    return warped_feature

def coords_grid(b, h, w, homogeneous=False, device=None):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))  # [H, W]

    stacks = [x, y]

    if homogeneous:
        ones = torch.ones_like(x)  # [H, W]
        stacks.append(ones)

    grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]

    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    if device is not None:
        grid = grid.to(device)

    return grid
