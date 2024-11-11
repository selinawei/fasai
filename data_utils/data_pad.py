import os
import numpy as np
import torch

data_base = '/workspace/xak1wx/FEASAI/data/FEA_processed_new'
save_base = '/workspace/xak1wx/FEASAI/data/FEA_preprocessed'
for folder in os.listdir(data_base):
    folder_path = f'{data_base}/{folder}'
    for dataf in os.listdir(folder_path):
        file_path = f'{folder_path}/{dataf}'
        os.makedirs(f'{save_base}/{folder}',exist_ok=True)
        save_path = f'{save_base}/{folder}/{dataf}'
        data = np.load(file_path,allow_pickle=True)
        npz_data = dict(data)  # 将现有数据存入字典

        occ_aps = data['occ_aps']
        occ_t = data['occ_t']
        depth_gt = data['depth']

        current_length_depth = depth_gt.shape[0]
        current_length = occ_aps.shape[0]
        target_length = 27
        if current_length<target_length:
            pad_length = target_length - current_length
            occ_aps = F.pad(occ_aps, (0, 0, 0, 0, 0, pad_length), "constant", 0)
            occ_t = F.pad(occ_t, (occ_t[current_length-1],  pad_length), value=0)
        else:
        # 如果长于目标长度，则裁剪
            occ_aps = occ_aps[:target_length]
            occ_t = occ_t[:target_length]

        if current_length_depth<target_length:
            pad_length = target_length - current_length_depth
            depth_gt = F.pad(depth_gt, (0, 0, 0,0, 0, pad_length), "constant", 0)
        else:
        # 如果长于目标长度，则裁剪
            depth_gt = depth_gt[:target_length]

         # 增加新的key 'depth'
        npz_data['depth'] = depth_gt
        npz_data['occ_t'] = occ_t
        npz_data['occ_aps'] = occ_aps

        # 将更新后的数据保存回npz文件
        np.savez(save_path, **npz_data)