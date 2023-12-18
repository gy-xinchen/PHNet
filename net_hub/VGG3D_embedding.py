import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torch.nn.functional as F


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_In',
    'vgg19_In', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.fc = nn.Linear(512, num_classes)
        self.fc_embedding = nn.Linear(num_classes,128)
        self.fc1 = nn.Linear(128, 2)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        a = x.shape[2]
        if x.shape[2] > 1:
            x = F.max_pool3d(x, kernel_size=(x.shape[2],7,7), stride=1).view(x.size(0), -1)
        else:
            x = F.max_pool3d(x, kernel_size=(1,7,7), stride=1).view(x.size(0), -1)
        x = self.fc(x)
        x_embedding = self.fc_embedding(x)
        x = self.fc1(x)
        return x, x_embedding

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2),padding=(1,0,0))]
        else:
            conv3d = nn.Conv3d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv3d, nn.InstanceNorm3d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv3d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(pretrained=False, model_root=None, **kwargs):
    """VGG 11-layer model (configuration "A")"""
    model = VGG(make_layers(cfg['A']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11'], model_root))
    return model


def vgg11_bn(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    kwargs.pop('model_root', None)
    return VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)


def vgg13(pretrained=False, model_root=None, **kwargs):
    """VGG 13-layer model (configuration "B")"""
    model = VGG(make_layers(cfg['B']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13'], model_root))
    return model


def vgg13_bn(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    kwargs.pop('model_root', None)
    return VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)


def vgg16(pretrained=False, model_root=None, **kwargs):
    """VGG 16-layer model (configuration "D")"""
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16'], model_root))
    return model


def vgg16_In(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    kwargs.pop('model_root', None)
    return VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)


def vgg19(pretrained=False, model_root=None, **kwargs):
    """VGG 19-layer model (configuration "E")"""
    model = VGG(make_layers(cfg['E']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19'], model_root))
    return model


def vgg19_In(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    kwargs.pop('model_root', None)
    return VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)

if __name__ == "__main__":
    input = torch.randn([1,1,27,224,224])
    net = vgg16()
    output = net(input)
    print()
