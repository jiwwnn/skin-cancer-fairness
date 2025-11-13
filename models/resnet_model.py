import torch
# import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import torch.nn.functional as F
import yaml

config = yaml.load(open("resnet_config.yaml", "r"), Loader=yaml.FullLoader)

# class ModelResNet(torch.nn.Module):
#     def __init__(self) :
#         super(ModelResNet, self).__init__()
        
#         self.model = models.resnet50(pretrained=config['pretrained'])     
#         self.model.fc = nn.Linear(self.model.fc.in_features, config['classes'])
    
#     def forward(self, x):
#        return self.model(x)
   
   
class ModelResNet(torch.nn.Module):
    def __init__(self) :
        super(ModelResNet, self).__init__()
        self.feature_extractor = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = self.feature_extractor.fc.in_features
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])   
        self.classifier = nn.Linear(num_ftrs, config['classes'])

    def forward(self, x):
        output = self.feature_extractor(x)
        output = output.view(x.size(0), -1)
        out1 = self.classifier(output)
        return out1