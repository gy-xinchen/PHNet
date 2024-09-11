import torch
import SimpleITK as sitk
import os
from torch.utils.data import DataLoader, Dataset
from pretrained_moco.MoCo_model import MoCoLoss, MoCo
from net_hub.PHNet import densenet121
import torchio as tio
import numpy as np
import math

def adjust_learning_rate(optimizer, epoch, lr, schedule, cos, epochs):
    """Decay the learning rate based on schedule"""
    if cos:  # cosine lr schedule
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / epochs))
    else:  # stepwise lr schedule
        for milestone in schedule:
            lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

def create_densenet_encoder():
    """创建 DenseNet 编码器"""
    base_encoder = densenet121(pretrained=False)
    return base_encoder

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """训练模型一个周期"""
    model.train()  # 设置模型为训练模式
    total_loss = 0.0  # 用于累积总损失

    for batch_idx, (query_images, key_images) in enumerate(train_loader):
        query_images = query_images.to(device)
        key_images = key_images.to(device)

        # 清除旧的梯度
        optimizer.zero_grad()

        # 前向传播
        query_features, key_features = model(query_images, key_images)

        # 更新队列
        model.update_queue(key_features)

        # 计算损失
        loss = criterion(query_features, key_features, model.get_queue())

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        # 累加损失
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)  # 计算平均损失
    print(f"Average Loss: {avg_loss:.4f}")

    return avg_loss

def get_query_transform():
    """定义 query 数据增强（保持不变）"""
    return tio.Compose([
        tio.RandomFlip(p=0.5),
        tio.RandomAffine(p=0.5),
        tio.ToCanonical(),
    ])

def get_key_transform():
    """定义 key 数据增强（应用较强的数据增强）"""
    return tio.Compose([
        tio.RandomFlip(p=0.3),
        tio.RandomAffine(p=0.3),
        tio.RandomBiasField(p=0.3),
        tio.RandomNoise(p=0.3),
        tio.RandomGamma(p=0.3),
        tio.ToCanonical(),
    ])

class NiftiDataset(Dataset):
    def __init__(self, nii_file, query_transform=None, key_transform=None, num_samples=5):
        self.nii_file = nii_file
        self.query_transform = query_transform
        self.key_transform = key_transform
        self.num_samples = num_samples

        # 读取图像
        self.image = sitk.ReadImage(self.nii_file)
        self.image_array = sitk.GetArrayFromImage(self.image).astype(np.float32)
        self.image_tensor = torch.tensor(self.image_array, dtype=torch.float32).unsqueeze(0)  # 添加通道维度

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # 使用数据增强生成 query 和 key 图像
        query_transform = self.query_transform
        key_transform = self.key_transform

        query_image_tensor = self.image_tensor.clone()
        key_image_tensor = self.image_tensor.clone()

        if query_transform:
            query_image_tensor = tio.ScalarImage(tensor=query_image_tensor)
            query_image_tensor = query_transform(query_image_tensor).tensor

        if key_transform:
            key_image_tensor = tio.ScalarImage(tensor=key_image_tensor)
            key_image_tensor = key_transform(key_image_tensor).tensor

        return query_image_tensor, key_image_tensor

def collate_fn(batch):
    """自定义 collate 函数，处理数据批次"""
    query_images, key_images = zip(*batch)
    query_images = torch.stack(query_images)
    key_images = torch.stack(key_images)
    return query_images, key_images

def save_model(model, filepath):
    """保存模型权重"""
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_encoder = create_densenet_encoder().to(device)
    model = MoCo(base_encoder, mlp=True).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-4)
    criterion = MoCoLoss()

    # Define data augmentation transformations
    query_transform = get_query_transform()
    key_transform = get_key_transform()

    # Data file path
    nii_file = r'D:\\patient002_BP.nii.gz'

    # Check if NIfTI file exists
    if not os.path.exists(nii_file):
        raise FileNotFoundError(f"NIfTI file not found at {nii_file}")

    train_dataset = NiftiDataset(
        nii_file=nii_file,
        query_transform=query_transform,
        key_transform=key_transform,
        num_samples=8  # Number of samples to generate from one NIfTI file
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,  # Use multiple workers to speed up data loading
        collate_fn=collate_fn  # Use custom collate_fn
    )

    num_epochs = 200
    schedule = [120, 160]  # Modify schedule as needed
    cos = True  # Enable cosine annealing

    best_loss = float('inf')  # Initialize best_loss to a large value

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Adjust learning rate and print
        current_lr = adjust_learning_rate(optimizer, epoch, 0.03, schedule, cos, num_epochs)
        print(f"Learning Rate: {current_lr:.6f}")

        # Train the model for one epoch and get the average loss
        avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Save model weights if the current loss is the best (minimum) loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, 'best_model_weights.pth')
            print(f"Model weights saved at 'best_model_weights.pth' with loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
