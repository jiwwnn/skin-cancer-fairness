import torch
# import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import torch.nn.functional as F
import yaml

config = yaml.load(open("disco_config.yaml", "r"), Loader=yaml.FullLoader)

class ModelDisCo(torch.nn.Module):
    def __init__(self) :
        super(ModelDisCo, self).__init__()
        bottle_neck = 256
            
        self.feature_extractor = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(num_ftrs, bottle_neck)
        # for contrastive loss
        self.project_head = nn.Sequential(
                nn.Linear(bottle_neck, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 128),
        )
        # self.activation = torch.nn.ReLU()
        # branch 1
        self.branch_1 = nn.Linear(bottle_neck, config['classes']) # cls
        # branch 2
        self.branch_2 = nn.Linear(bottle_neck, config['fitz_classes']) # fitz cls

    def forward(self, x):
        feature_map = self.feature_extractor(x)  # (bs, bottle_neck)
        out_1 = self.branch_1(feature_map)
        out_2 = self.branch_2(feature_map)
        out_4 = self.project_head(feature_map)
        # detach feature map and pass though branch 2 again
        feature_map_detach = feature_map.detach() # 역전파 과정에서 계산되지 않는 별도의 출력 생성
        out_3 = self.branch_2(feature_map_detach)
        return [out_1, out_2, out_3, out_4]
        # return [out_1, out_2, out_3]