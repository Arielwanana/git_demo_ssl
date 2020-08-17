import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbone import resnet
from model.head import build_head

class resnet50_cls_v1(nn.Module):
    def __init__(self, num_classes, backbone='resnet18', output_stride=16,
                 sync_bn=True, freeze_bn=False):
        super(resnet50_cls_v1, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            # todo: use syn_bn
            BatchNorm = nn.BatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = resnet.resnet18(num_classes)

        self.freeze_bn = freeze_bn

    def forward(self, input):
        x = self.backbone(input)
        x = F.softmax(x, dim=1)
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

class dnn_8(nn.Module):
    """
    回归网络
    """
    def __init__(self, in_channel, out_channel):
        super(dnn_8, self).__init__()
        self.fc1 = nn.Linear(in_channel, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, 1024)
        self.fc6 = nn.Linear(1024, 1024)
        self.fc7 = nn.Linear(1024, 1024)
        self.fc8 = nn.Linear(1024, out_channel)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.softmax(self.fc8(x), dim=-1)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        for p in self.modules().parameters():
            if p.requires_grad:
                yield p


class FNN(nn.Module):

    def __init__(self,input_size,hidden_size,num_classes):

        super(FNN,self).__init__()# Inherited from the parent class nn.Module

        self.fc1 = nn.Linear(input_size,hidden_size)# 1st Full-Connected Layer: 784 (input data) -> 500 (hidden node)
        self.relu = nn.ReLU()  # Non-Linear ReLU Layer: max(0,x)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, 1024)
        self.fc6 = nn.Linear(1024, 1024)
        self.fc7 = nn.Linear(1024, 1024)

        self.fc8 = nn.Linear(hidden_size,
                             num_classes)  # 2nd Full-Connected Layer: 500 (hidden node) -> 10 (output class)


    def forward(self, x):  # Forward pass: stacking each layer together

        out = self.fc1(x)
        out = self.relu(out)
        x = F.relu(self.fc2(out))

        out = self.fc8(x)
        return out



if __name__ == "__main__":
    model = resnet50_cls_v1(num_classes=100, backbone='mobilenet', output_stride=16)
    model = dnn_8(272, 185)
    model.train()
    input = torch.rand(1, 1, 272)
    output = model(input)
    print(output.size())
    print(output.mean())