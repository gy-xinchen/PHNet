import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, epsilon=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

    def forward(self, output, target):
        num_classes = output.size(-1)
        log_probs = F.log_softmax(output, dim=-1)
        targets_one_hot = F.one_hot(target, num_classes=num_classes).float()
        targets_smoothed = (1 - self.epsilon) * targets_one_hot + self.epsilon / num_classes
        nll_loss = -log_probs * targets_smoothed
        nll_loss = nll_loss.sum(dim=-1)
        return nll_loss.mean()