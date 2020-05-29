from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torchvision.models as models
from dbml_benchmark.modeling import registry


@registry.BACKBONES.register('resnet50')
class ResNet50(nn.Module):

    def __init__(self):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(pretrained=True)

        for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
            module.eval()
            module.train = lambda _: None

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = self.model.fc(x)  --remove
        return x

    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'last_linear' in i:
                continue
            self.model.state_dict()[i].copy_(param_dict[i])

@registry.BACKBONES.register('resnet18')
class ResNet18(nn.Module):

    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=True)

        for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
            module.eval()
            module.train = lambda _: None

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = self.model.fc(x)  --remove
        return x

    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'last_linear' in i:
                continue
            self.model.state_dict()[i].copy_(param_dict[i])

@registry.BACKBONES.register('resnet34')
class ResNet34(nn.Module):

    def __init__(self):
        super(ResNet34, self).__init__()
        self.model = models.resnet34(pretrained=True)

        for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
            module.eval()
            module.train = lambda _: None

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = self.model.fc(x)  --remove
        return x

    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'last_linear' in i:
                continue
            self.model.state_dict()[i].copy_(param_dict[i])

@registry.BACKBONES.register('resnet101')
class ResNet101(nn.Module):

    def __init__(self):
        super(ResNet101, self).__init__()
        self.model = models.resnet101(pretrained=True)

        for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
            module.eval()
            module.train = lambda _: None

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = self.model.fc(x)  --remove
        return x

    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'last_linear' in i:
                continue
            self.model.state_dict()[i].copy_(param_dict[i])

@registry.BACKBONES.register('resnet152')
class ResNet152(nn.Module):

    def __init__(self):
        super(ResNet152, self).__init__()
        self.model = models.resnet152(pretrained=True)

        for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
            module.eval()
            module.train = lambda _: None

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = self.model.fc(x)  --remove
        return x

    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'last_linear' in i:
                continue
            self.model.state_dict()[i].copy_(param_dict[i])