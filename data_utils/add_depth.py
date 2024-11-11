import os
import numpy as np

def process_files(a_folder, b_folder):
    # 遍历Test和Train目录
    for split in ['Test', 'Train']:
        # a文件夹中的Test和Train路径
        a_split_path = os.path.join(a_folder, split)

        # b文件夹中的Test和Train路径
        b_split_path = os.path.join(b_folder, split)

        # 遍历a文件夹中的所有04d命名的npz文件
        for npz_file in sorted(os.listdir(a_split_path)):
            if npz_file.endswith('.npz'):
                # 取出文件名（不带扩展名），这应该是04d格式的名称
                folder_name = os.path.splitext(npz_file)[0]

                # 在b文件夹中找到对应的04d文件夹
                corresponding_folder = os.path.join(b_split_path, folder_name)
                if not os.path.exists(corresponding_folder):
                    print(f"Warning: {corresponding_folder} does not exist")
                    continue

                # 收集所有npy文件并加载它们
                npy_files = sorted([os.path.join(corresponding_folder, f) for f in os.listdir(corresponding_folder) if
                                    f.endswith('.npy')])
                depth_images = []

                # 遍历npy文件，将其加载为numpy数组并添加到列表
                for npy_file in npy_files:
                    depth_image = np.load(npy_file)  # 每个npy文件应该是（260, 346）
                    assert depth_image.shape == (260, 346), f"Unexpected shape {depth_image.shape} in {npy_file}"
                    depth_images.append(depth_image)

                # 将所有深度图像cat为（n，260,346）的形状
                depth_stack = np.stack(depth_images, axis=0)

                # 读取现有的npz文件
                npz_file_path = os.path.join(a_split_path, npz_file)
                with np.load(npz_file_path, allow_pickle=True) as data:
                    npz_data = dict(data)  # 将现有数据存入字典

                # 增加新的key 'depth'
                npz_data['depth'] = depth_stack

                # 将更新后的数据保存回npz文件
                np.savez(npz_file_path, **npz_data)

                print(f"Updated {npz_file_path} with new key 'depth', shape: {depth_stack.shape}")

# 示例调用
a_folder = '/scratch365/lwei5/FEASAI_data'  # 替换为a文件夹路径
b_folder = '/scratch365/lwei5/FEASAI_compare_exp/raftstereo/result'  # 替换为b文件夹路径

process_files(a_folder, b_folder)
