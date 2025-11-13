import torch
import torch.nn as nn
# import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights
import yaml
from models.models_losses import Network, Supervised_Contrastive_Loss
import sys
from transformers import ViTModel,AutoTokenizer,AutoModel
import torch.nn.functional as F

config = yaml.load(open("resm_config.yaml", "r"), Loader=yaml.FullLoader)

# class Network(torch.nn.Module): # 이부분 코드 틀렷엇음 수정 20241112
#     def __init__(self, choice='resnet18', output_size=9, pretrained=config['pretrained']) :
#         '''
#         output_size: int  only one output
#                      list  first is skin type, second is skin conditipon (used in disentangle, attribute_aware)
#         '''
#         super(Network, self).__init__()
#         self.choice = choice
#         bottle_neck = 256

#         if self.choice == 'vgg16':
#             self.feature_extractor = models.vgg16(pretrained=pretrained)
#             num_ftrs = self.feature_extractor.classifier[6].in_features
#             self.feature_extractor.classifier[6] = nn.Linear(num_ftrs, output_size)
            
        
#         if self.choice == 'resnet50':
#             self.feature_extractor = models.resnet50(pretrained=pretrained)
#             num_ftrs = self.feature_extractor.fc.in_features
#             self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])
#             self.classifier = nn.Linear(num_ftrs, output_size)
#             self.project_head = nn.Sequential(
#                  nn.Linear(num_ftrs, 512),
#                  nn.BatchNorm1d(512),
#                  nn.ReLU(inplace=True),
#                  nn.Linear(512, 128),
#             )
            
#         if self.choice == 'disentangle':
#             self.feature_extractor = models.resnet18(pretrained=pretrained)
#             num_ftrs = self.feature_extractor.fc.in_features
#             self.feature_extractor.fc = nn.Linear(num_ftrs, bottle_neck)
#             # for contrastive loss
#             self.project_head = nn.Sequential(
#                  nn.Linear(bottle_neck, 512),
#                  nn.BatchNorm1d(512),
#                  nn.ReLU(inplace=True),
#                  nn.Linear(512, 128),
#             )
#             # self.activation = torch.nn.ReLU()
#             # branch 1
#             self.branch_1 = nn.Linear(bottle_neck, output_size[0])
#             # branch 2
#             self.branch_2 = nn.Linear(bottle_neck, output_size[1])

#         if self.choice == "vit":
#             self.feature_extractor = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
#             layers = []
#             hidden_size = 128

#             layers.append(nn.Linear(self.feature_extractor.config.hidden_size, hidden_size))
#             layers.append(nn.ReLU())


#             layers.append(nn.Linear(hidden_size, output_size))
#             layers.append(nn.Sigmoid())

#             self.classifier = nn.Sequential(*layers)
#             #self.classifier = nn.Linear(self.feature_extractor.config.hidden_size, output_size)
                
        
#         if self.choice == 'attribute_aware':
#             # use sensitive information into the network to train
#             bottle_neck = 256
#             self.feature_extractor = models.resnet18(pretrained=pretrained)
#             num_ftrs = self.feature_extractor.fc.in_features
#             self.feature_extractor.fc = nn.Linear(num_ftrs, bottle_neck)
#             self.attribute_layer = nn.Linear(output_size[1], bottle_neck) 
#             self.classifier = nn.Linear(bottle_neck, output_size[0])

    
#     def forward(self, x, attribute=None):
#         if self.choice == 'disentangle':
#             feature_map = self.feature_extractor(x)  # (bs, bottle_neck)
#             out_1 = self.branch_1(feature_map)
#             out_2 = self.branch_2(feature_map)
#             out_4 = self.project_head(feature_map)
#             # detach feature map and pass though branch 2 again
#             feature_map_detach = feature_map.detach()
#             out_3 = self.branch_2(feature_map_detach)
#             return [out_1, out_2, out_3, out_4]
#             # return [out_1, out_2, out_3]
            
#         elif self.choice == 'attribute_aware':
#             feature_map = self.feature_extractor(x) # (bs, bottle_neck)
#             attribute_upsample = self.attribute_layer(attribute) # (bs, bottle_neck)
#             fused_feature = feature_map+attribute_upsample # (bs, bottle_neck)
#             fused_feature = F.relu(fused_feature) # (bs, bottle_neck)
#             out = self.classifier(fused_feature)
#             return out
        
#         elif self.choice == "vit":
#             output = self.feature_extractor(x)
#             output = output.last_hidden_state
#             out1 = self.classifier(output)
#             #out2 = self.project_head(output)
#             return [torch.mean(out1, dim=1), output]           

#         else:
#             output = self.feature_extractor(x)
#             output = output.view(x.size(0), -1)
#             out1 = self.classifier(output)
#             out2 = self.project_head(output)
#             return [out1, out2, output]


# class ModelRESM(torch.nn.Module):
#     def __init__(self) :
#         super(ModelRESM, self).__init__()
        
#         self.model = Network(choice='resnet50', 
#                              output_size=len(label_codes), 
#                              pretrained=config['pretrained'])
        
#     def forward(self, x):
#         out1, out2, output = self.model(x)
#         return out1

# 코드수정 20250106
# class ModelRESM(nn.Module):  
#     def __init__(self):
#         super(ModelRESM, self).__init__()
#         bottle_neck = 256
#         self.feature_extractor = models.resnet50(pretrained=config['pretrained'])
#         num_ftrs = self.feature_extractor.fc.in_features
#         self.feature_extractor.fc = nn.Linear(num_ftrs, bottle_neck)
#         self.attribute_layer = nn.Linear(config['fitz_classes'], bottle_neck)
#         self.classifier = nn.Linear(bottle_neck, config['classes'])

#     def forward(self, x, attribute):
#         feature_map = self.feature_extractor(x)
#         attribute_upsample = self.attribute_layer(attribute)  # 원핫 인코딩된 attribute를 처리
#         fused_feature = feature_map + attribute_upsample
#         fused_feature = F.relu(fused_feature)
#         out = self.classifier(fused_feature)
#         return out


# 원본코드 
class ModelRESM(nn.Module):
    def __init__(self):
        super(ModelRESM, self).__init__()
        self.feature_extractor = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = self.feature_extractor.fc.in_features
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])
            
        self.classifier = nn.Linear(num_ftrs, config['classes'])

    def forward(self, x):
        output = self.feature_extractor(x)
        output = output.view(x.size(0), -1)
        out1 = self.classifier(output)
        return out1
    