import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    
    def forward(self, feature_map):
        return F.adaptive_avg_pool2d(feature_map, 1).squeeze(-1).squeeze(-1)

class ImageClassifier(torch.nn.Module):
    def __init__(self, P):
        super(ImageClassifier, self).__init__()
        
        self.arch = P['arch']

        feature_extractor = torchvision.models.resnet50(pretrained=P['use_pretrained'])
        feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-2])

        if P['freeze_feature_extractor']:
            for param in feature_extractor.parameters():
                param.requires_grad = False
        else:
            for param in feature_extractor.parameters():
                param.requires_grad = True

        self.feature_extractor = feature_extractor
        self.avgpool = GlobalAvgPool2d()
        self.onebyone_conv = nn.Conv2d(P['feat_dim'], P['num_classes'], 1)
        
        alpha = torch.ones(1, P['num_classes'], dtype=torch.float)
        self.register_buffer("alpha", alpha)

    def unfreeze_feature_extractor(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
        
    def forward(self, x, fw_cam=False):
        feats = self.feature_extractor(x)
        CAM = self.onebyone_conv(feats)
        CAM = torch.where(CAM > 0, CAM * self.alpha.view(1, -1, 1, 1), CAM) 
        logits = F.adaptive_avg_pool2d(CAM, 1).squeeze(-1).squeeze(-1)

        if fw_cam:
            return logits, CAM

        return logits
