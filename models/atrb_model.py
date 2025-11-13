import torch
import torch.nn as nn
# import torchvision.models as models
from torchvision.models import resnet50,ResNet50_Weights
import torch.nn.functional as F
import yaml

config = yaml.load(open("atrb_config.yaml", "r"), Loader=yaml.FullLoader)

# class Network(torch.nn.Module):
#     def __init__(self, choice='attribute_aware', output_size=9, pretrained=True):
#         '''
#         output_size: int  only one output
#                      list  first is skin type, second is skin condition (used in disentangle, attribute_aware)
#         '''
#         super(Network, self).__init__()
#         self.choice = choice
#         bottle_neck = 256

#         if self.choice == 'vgg16':
#             self.feature_extractor = models.vgg16(pretrained=pretrained)
#             num_ftrs = self.feature_extractor.classifier[6].in_features
#             self.feature_extractor.classifier[6] = nn.Linear(num_ftrs, output_size)
        
#         elif self.choice == 'resnet18':
#             self.feature_extractor = models.resnet18(pretrained=pretrained)
#             num_ftrs = self.feature_extractor.fc.in_features
#             self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])
#             self.classifier = nn.Linear(num_ftrs, output_size) # 여기가 문제
#             self.project_head = nn.Sequential(
#                  nn.Linear(num_ftrs, 512),
#                  nn.BatchNorm1d(512),
#                  nn.ReLU(inplace=True),
#                  nn.Linear(512, 128),
#             )
            
#         elif self.choice == 'disentangle':
#             self.feature_extractor = models.resnet18(pretrained=pretrained)
#             num_ftrs = self.feature_extractor.fc.in_features
#             self.feature_extractor.fc = nn.Linear(num_ftrs, bottle_neck)
#             self.project_head = nn.Sequential(
#                  nn.Linear(bottle_neck, 512),
#                  nn.BatchNorm1d(512),
#                  nn.ReLU(inplace=True),
#                  nn.Linear(512, 128),
#             )
#             self.branch_1 = nn.Linear(bottle_neck, output_size[0])
#             self.branch_2 = nn.Linear(bottle_neck, output_size[1])

#         elif self.choice == "vit":
#             self.feature_extractor = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
#             layers = []
#             hidden_size = 128
#             layers.append(nn.Linear(self.feature_extractor.config.hidden_size, hidden_size))
#             layers.append(nn.ReLU())
#             layers.append(nn.Linear(hidden_size, output_size))
#             layers.append(nn.Sigmoid())
#             self.classifier = nn.Sequential(*layers)
                
#         elif self.choice == 'attribute_aware':
#             bottle_neck = 256
#             self.feature_extractor = models.resnet50(pretrained=pretrained)
#             num_ftrs = self.feature_extractor.fc.in_features
#             self.feature_extractor.fc = nn.Linear(num_ftrs, bottle_neck)
            
#             # self.attribute_layer 수정 (input_size=6, 원핫 인코딩 클래스 수와 일치)
#             self.attribute_layer = nn.Linear(6, bottle_neck)
#             self.classifier = nn.Linear(bottle_neck, output_size[0])
    
#     def forward(self, x, attribute=None):
#         if self.choice == 'disentangle':
#             feature_map = self.feature_extractor(x)
#             out_1 = self.branch_1(feature_map)
#             out_2 = self.branch_2(feature_map)
#             out_4 = self.project_head(feature_map)
#             feature_map_detach = feature_map.detach()
#             out_3 = self.branch_2(feature_map_detach)
#             return [out_1, out_2, out_3, out_4]
            
#         elif self.choice == 'attribute_aware':
            
#             ## 라벨인코딩 방식
#             # feature_map = self.feature_extractor(x)
            
#             # # This code snippet is checking if the `attribute` tensor is one-dimensional and then
#             # # converting it to a Float type. Here's a breakdown of what each part of the code is doing:
#             # # attribute가 1차원인지 검사하고 Float 타입으로 변환
#             # if attribute.dim() == 1:
#             #     attribute_upsample = self.attribute_layer(attribute[:4].unsqueeze(0).float())
#             # else:
#             #     attribute_upsample = self.attribute_layer(attribute[:, :4].float())
                
#             # fused_feature = feature_map + attribute_upsample
#             # fused_feature = F.relu(fused_feature)
#             # out = self.classifier(fused_feature)
#             # return out
            
#             ## 원핫인코딩 방식 -> attribute가 원핫 인코딩된 상태로 들어온다고 가정
#             feature_map = self.feature_extractor(x)
#             attribute_upsample = self.attribute_layer(attribute)  # 원핫 인코딩된 attribute를 처리
#             fused_feature = feature_map + attribute_upsample
#             fused_feature = F.relu(fused_feature)
#             out = self.classifier(fused_feature)
#             return out
        
#         elif self.choice == "vit":
#             output = self.feature_extractor(x).last_hidden_state
#             out1 = self.classifier(output)
#             return [torch.mean(out1, dim=1), output]

#         else:
#             output = self.feature_extractor(x)
#             output = output.view(x.size(0), -1)
#             out1 = self.classifier(output)
#             out2 = self.project_head(output)
#             return [out1, out2, output]

# class ModelATRB(nn.Module):  
#     def __init__(self, label_codes, config):
#         super(ModelATRB, self).__init__()
        
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
#         self.model = Network(choice='attribute_aware', 
#                              output_size=[len(label_codes), 4],  # [classification_output_size, attribute_output_size] , 이거 다시 확인 fitscale 4
#                              pretrained=config['pretrained'])
        

#     def forward(self, x, attribute):
#         out = self.model(x, attribute) 
#         return out  


class ModelATRB(nn.Module):  
    def __init__(self):
        super(ModelATRB, self).__init__()
        bottle_neck = 256
        self.feature_extractor = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(num_ftrs, bottle_neck)
        self.attribute_layer = nn.Linear(config['fitz_classes'], bottle_neck)
        self.classifier = nn.Linear(bottle_neck, config['classes'])

    def forward(self, x, attribute):
        feature_map = self.feature_extractor(x)
        attribute_upsample = self.attribute_layer(attribute)  # 원핫 인코딩된 attribute를 처리
        fused_feature = feature_map + attribute_upsample
        fused_feature = F.relu(fused_feature)
        out = self.classifier(fused_feature)
        return out

