import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Head, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            inplanes = 256
        elif backbone == 'xception':
            inplanes = 128
        elif backbone == 'mobilenet':
            inplanes = 24
        else:
            raise NotImplementedError

        self.convblock = nn.Sequential([nn.Conv2d(inplanes, 1024, 1, bias=False),
                                        BatchNorm(1024),
                                        nn.ReLU(),
        ])

        self.out = nn.Linear(1024, num_classes)

        self._init_weight()


    def forward(self, x):
        x = self.convblock(x)
        x = nn.Softmax(self.out(x))
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder(num_classes, backbone, BatchNorm):
    return Head(num_classes, backbone, BatchNorm)