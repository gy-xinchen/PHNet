import os
import glob
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd

# 图像目录
PAH_dir = r"G:\CMR-res\muti_center_data0927\mix_train_data\slice05_224x224\PAH"
IPAH_dir = r"G:\CMR-res\muti_center_data0927\mix_train_data\slice05_224x224\noPAH"

# 获取图像路径
image_IPAH = glob.glob(os.path.join(IPAH_dir, "*nii.gz*"))
image_PAH = glob.glob(os.path.join(PAH_dir, "*nii.gz*"))

# 生成标签
PAH_id = [1] * len(image_PAH)
IPAH_id = [0] * len(image_IPAH)

# 合并图像路径和标签
data_dicts = [{'image': image_name, 'label': label} for image_name, label in zip(image_PAH + image_IPAH, PAH_id + IPAH_id)]

# 分层五折交叉验证
floder = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
train_files = []
val_files = []

labels = [d['label'] for d in data_dicts]  # 提取标签列表

for Trindex, Tsindex in floder.split(data_dicts, labels):
    train_files.append([data_dicts[i] for i in Trindex])
    val_files.append([data_dicts[i] for i in Tsindex])

# 保存到 CSV 文件
def save_to_csv(data, filepath):
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)

save_to_csv(train_files, r'G:\CMR-res\muti_center_data0927\mix_train_data\slice05_224x224\slice05_StratifiedKFold_train_224x224.csv')
save_to_csv(val_files, r'G:\CMR-res\muti_center_data0927\mix_train_data\slice05_224x224\slice05_StratifiedKFold_val_224x224.csv')
