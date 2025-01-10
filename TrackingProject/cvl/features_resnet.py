"""
Created by: Andreas Robinsson
Edits by: Johan Edstedt
"""

import torch
from torch import nn
import torchvision.models.resnet as resnet
from torchvision.models import *

class DeepFeatureExtractor(nn.Module):
    def __init__(self, network_type='resnet50', pretrained=True):
        """
        :param network_type: network constructor function name (string). See torchvision.models.resnet.__all__
        :param pretrained:
        """
        super().__init__()
        assert network_type in resnet.__all__ and network_type != "ResNet", f"{network_type} not supported, try: {resnet.__all__}"
        if pretrained:
            if network_type == 'resnet18':
                weights = ResNet18_Weights.DEFAULT
            elif network_type == 'resnet34':
                weights = ResNet34_Weights.DEFAULT
            elif network_type == 'resnet50':
                weights = ResNet50_Weights.DEFAULT
            elif network_type == 'resnet101':
                weights = ResNet101_Weights.DEFAULT
        else:
            weights = None
        backbone = eval(f"resnet.{network_type}(weights={weights})")
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).reshape(3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).reshape(3,1,1))
        if torch.cuda.is_available():
            self.to("cuda:0")
            self.device = "cuda:0"
        else:
            self.device = "cpu"
    
    def preprocess(self, im):
        x = (torch.tensor(im).permute(2,0,1)[None].float().to(self.device) / 255.0 - self.mean) / self.std
        return x
    def forward(self, im):
        x = self.preprocess(im)
        with torch.no_grad():
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x1 = self.layer1(x)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x4 = self.layer4(x3)
            return x1, x2, x3, x4
