import torch
import torch.nn as nn


class AlexNet(nn.Module):

    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        # The first convolution operation
        self.conv1 = nn.Sequential(nn.Conv3d(1, 64, kernel_size=(5,11,11), stride=(4,4,4), padding=(2,2,2)),
                                   nn.ReLU(inplace=True))
        # First pooling operation
        self.pool1 = nn.MaxPool3d(kernel_size=(3,3,3), stride=(1,2,2))
        # The second convolution operation
        self.conv2 = nn.Sequential(nn.Conv3d(64, 192, kernel_size=(2,5,5), padding=(1,2,2)),
                                   nn.ReLU(inplace=True))
        # Second pooling
        self.pool2 = nn.MaxPool3d(kernel_size=(3,3,3), stride=(1,2,2))
        # The third convolution operation
        self.conv3 = nn.Sequential(nn.Conv3d(192, 384, kernel_size=(3,3,3), padding=1),
                                   nn.ReLU(inplace=True))
        # The fourth convolution operation
        self.conv4 = nn.Sequential(nn.Conv3d(384, 256, kernel_size=(3,3,3), padding=1),
                                   nn.ReLU(inplace=True))
        # The fifth convolution operation
        self.conv5 = nn.Sequential(nn.Conv3d(256, 256, kernel_size=(3,3,3), padding=1),
                                   nn.ReLU(inplace=True))
        # The third pooling
        self.pool3 = nn.MaxPool3d(kernel_size=(3,3,3), stride=(1,2,2))
        # average pooling
        self.avgpool = nn.AdaptiveAvgPool3d((2, 6, 6))
        #
        self.classifier = nn.Sequential(
            # Randomly discard some neurons
            nn.Dropout(),
            # First full connection operation
            nn.Linear(256 * 2 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            # The second full connection operation
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # The third full connection operation
            nn.Linear(4096, 128),
        )

        self.classifier1 = nn.Linear(128, num_classes)

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
        x = torch.flatten(x, 1)  # flatten(x,1) is spliced according to
                                 # the first dimension of x (spliced according to columns, spliced horizontally)
        x1 = self.classifier(x)
        embedding = x1
        x = self.classifier1(x1)
        return x, embedding


if __name__ == "__main__":
    import torch as t
    input = t.randn(1, 1, 25, 224, 224)

    net = AlexNet(2)

    out = net(input)
