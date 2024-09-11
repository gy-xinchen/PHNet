# -*- coding: utf-8 -*-
# @Time    : 2024/9/10 19:48
# @Author  : yuan
# @File    : MoCo_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoCo(nn.Module):
    def __init__(self, base_encoder, dim=128, k=65536, m=0.999, t=0.07, mlp=False):
        super(MoCo, self).__init__()
        self.dim = dim
        self.m = m
        self.t = t
        self.k = k

        # 主网络 (query network)
        self.encoder_q = base_encoder

        # 动量编码器 (key network)
        self.encoder_k = self._build_momentum_encoder(base_encoder)

        # 初始化动量编码器
        self._initialize_momentum_encoder()

        # MLP head for projection
        if mlp:
            self.fc = nn.Sequential(
                nn.Linear(self.dim, self.dim),
                nn.ReLU(),
                nn.Linear(self.dim, self.dim)
            )
        else:
            self.fc = nn.Identity()

        # 队列
        self.queue = torch.randn(self.dim, k).cuda()
        self.queue = F.normalize(self.queue, dim=0)
        self.queue_ptr = 0

    def _build_momentum_encoder(self, base_encoder):
        """构建动量编码器"""
        feature_extractor = nn.Sequential(*list(base_encoder.children())[:-1])  # 例如去掉最后的全连接层

        # 定义动量编码器
        encoder = nn.Sequential(
            feature_extractor,  # 特征提取部分
            nn.MaxPool3d(kernel_size=(3, 7, 7), stride=1, padding=0),
            nn.Flatten(),  # 如果需要展平特征图
            nn.Linear(base_encoder.classifier.in_features, self.dim),  # 映射到目标维度
            nn.ReLU(),
            nn.Linear(self.dim, self.dim)  # 最终投影
        )
        return encoder

    def _initialize_momentum_encoder(self):
        """初始化动量编码器"""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data.clone()

    def forward(self, im_q, im_k):
        """前向传播"""
        # 查询图像的特征
        q = self.encoder_q(im_q)
        q = self.fc(q)
        q = F.normalize(q, dim=1)

        # 键图像的特征
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k)
            k = self.fc(k)
            k = F.normalize(k, dim=1)

        return q, k

    def _momentum_update_key_encoder(self):
        """更新动量编码器"""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def update_queue(self, keys):
        """更新负样本队列"""
        batch_size = keys.size(0)
        if self.queue_ptr + batch_size > self.k:
            self.queue[:, self.queue_ptr:] = keys.t()
            self.queue[:, :batch_size - (self.k - self.queue_ptr)] = keys.t()[:batch_size - (self.k - self.queue_ptr)]
        else:
            self.queue[:, self.queue_ptr:self.queue_ptr + batch_size] = keys.t()
        self.queue_ptr = (self.queue_ptr + batch_size) % self.k

    def get_queue(self):
        """获取负样本队列"""
        return self.queue.clone().detach()

class MoCoLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(MoCoLoss, self).__init__()
        self.temperature = temperature

    def forward(self, q, k, queue):
        # 计算正样本相似度
        pos_sim = torch.matmul(q, k.t()) / self.temperature
        # 计算负样本相似度
        neg_sim = torch.matmul(q, queue) / self.temperature
        # 损失函数
        loss = -torch.mean(F.log_softmax(torch.cat([pos_sim, neg_sim], dim=1), dim=1)[:, 0])
        return loss
