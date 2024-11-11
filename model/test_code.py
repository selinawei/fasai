import torch


bs = 6
ts = 30
#[bs,ts,H,W]
reft = torch.randn(bs, 2, 1, 1)
time = torch.randn(bs, ts, 1, 1)
depth = torch.randn(bs,ts,260,346)
slices = [torch.min(torch.abs(time - reft[:, i:i+1, :, :]), dim=1, keepdim=True)[1].reshape(bs) for i in range(reft.shape[1])]
depth_refts = [torch.stack([depth[bs, ts:ts+1, ...] for bs, ts in enumerate(slice)], dim=0) for slice in slices]
depth_reft = torch.cat(depth_refts, dim=1)  # [bs, 2, H, W]
print(depth_reft.shape)
