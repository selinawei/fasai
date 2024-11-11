import os
import numpy as np
import shutil
from sklearn.model_selection import train_test_split

# 设置源目录和目标目录
source_dir = "/scratch365/lwei5/npz_file/FEA_data/Train"  # 替换为你的 train 文件夹路径
train_dir = "/scratch365/lwei5/FEASAI_data/Train/"  # 替换为保存训练集的目录
test_dir = "/scratch365/lwei5/FEASAI_data/Test/"  # 替换为保存测试集的目录

# 创建输出目录
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 获取所有 .npz 文件
files = [f for f in os.listdir(source_dir) if f.endswith('.npz')]

# 分割数据集，使用 3:7 的比例
train_files, test_files = train_test_split(files, test_size=0.1, random_state=42)


# 函数用于重命名文件并复制到目标文件夹
def copy_and_rename_files(file_list, target_dir, start_index):
    for i, file_name in enumerate(file_list):
        # 构造新的文件名
        new_file_name = f"{start_index + i:04d}.npz"
        src_file_path = os.path.join(source_dir, file_name)
        dst_file_path = os.path.join(target_dir, new_file_name)

        # 复制文件到目标文件夹并重命名
        shutil.copy(src_file_path, dst_file_path)


# 复制并重命名文件到训练集和测试集
copy_and_rename_files(train_files, train_dir, 0)  # 从 0000 开始命名训练集
copy_and_rename_files(test_files, test_dir, 0)  # 从 0000 开始命名测试集

print("Data split into training and testing sets successfully.")
