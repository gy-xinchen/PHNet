import torch
import torch.nn as nn


class AlexNet(nn.Module):

    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        # 第一次卷积操作
        self.conv1 = nn.Sequential(nn.Conv3d(1, 64, kernel_size=(5,11,11), stride=(4,4,4), padding=(2,2,2)),
                                   nn.ReLU(inplace=True))
        # 第一次池化操作
        self.pool1 = nn.MaxPool3d(kernel_size=(3,3,3), stride=(1,2,2))
        # 第二次卷积操作
        self.conv2 = nn.Sequential(nn.Conv3d(64, 192, kernel_size=(2,5,5), padding=(1,2,2)),
                                   nn.ReLU(inplace=True))
        # 第二次池化
        self.pool2 = nn.MaxPool3d(kernel_size=(3,3,3), stride=(1,2,2))
        # 第三次卷积操作
        self.conv3 = nn.Sequential(nn.Conv3d(192, 384, kernel_size=(3,3,3), padding=1),
                                   nn.ReLU(inplace=True))
        # 第四次卷积操作
        self.conv4 = nn.Sequential(nn.Conv3d(384, 256, kernel_size=(3,3,3), padding=1),
                                   nn.ReLU(inplace=True))
        # 第五次卷积操作
        self.conv5 = nn.Sequential(nn.Conv3d(256, 256, kernel_size=(3,3,3), padding=1),
                                   nn.ReLU(inplace=True))
        # 第三次池化
        self.pool3 = nn.MaxPool3d(kernel_size=(3,3,3), stride=(1,2,2))
        # 平均池化
        self.avgpool = nn.AdaptiveAvgPool3d((2, 6, 6))
        #
        self.classifier = nn.Sequential(
            # 随机丢弃部分神经元
            nn.Dropout(),
            # 第一次全连接操作
            nn.Linear(256 * 2 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            # 第二次全连接操作
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # 第三次全连接操作
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # flatten(x,1)是按照x的第1个维度拼接（按照列来拼接，横向拼接）
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    import torch as t
    input = t.randn(1, 1, 25, 224, 224)

    net = AlexNet(2)

    out = net(input)
