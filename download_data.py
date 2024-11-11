import os
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

# 设置仓库信息
repo_id = "pearpp/FEASAI"  # 你的 Hugging Face repo id
local_dir = "/scratch365/lwei5/FEA_data"  # 保存文件的本地目录
os.makedirs(local_dir, exist_ok=True)  # 创建目标文件夹

# 获取 Hugging Face API 实例
api = HfApi()

# 获取远程仓库中的文件列表
files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", token="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")

# 过滤掉不需要下载的文件，比如 .gitattributes
# files_to_download = [file for file in files if not file.startswith(".")]
files_to_download = ["blender_processed.tar.gz"]
# 初始化进度条并下载文件
with tqdm(total=len(files_to_download), desc="Downloading files", unit="file") as pbar:
    for file_path in files_to_download:
        # 下载每个文件并保存到本地目录
        local_file_path = os.path.join(local_dir, file_path)
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        try:
            # 使用 hf_hub_download 下载文件
            hf_hub_download(
                repo_id=repo_id,
                filename=file_path,
                repo_type="dataset",
                local_dir=local_dir,
                token="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"
            )
        except Exception as e:
            print(f"Failed to download {file_path}: {e}")

        # 更新进度条
        pbar.update(1)

print(f"All files have been downloaded to {local_dir}")
