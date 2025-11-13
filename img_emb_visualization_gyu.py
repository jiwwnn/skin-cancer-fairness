import pandas as pd
import numpy as np
import torch
import matplotlib
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from models.ours_model import ModelOurs
from models.resnet_model import ModelResNet
from dataloader.ours_dataset_wrapper import TrainDataSetWrapper, TestDataSetWrapper
import yaml
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler  
matplotlib.use('Agg')
plt.ion()
ours_config = yaml.load(open("ours_config.yaml", "r"), Loader=yaml.FullLoader)

class ResNetWithLinear(nn.Module):
    def __init__(self, original_resnet, target_dim=512):
        super(ResNetWithLinear, self).__init__()
        self.feature_extractor = nn.Sequential(*list(original_resnet.children())[:-1])
        num_ftrs = original_resnet.fc.in_features  
        
        self.linear = nn.Linear(num_ftrs, target_dim)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)  
        reduced_features = self.linear(features)
        return reduced_features


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ours_model = ModelOurs(**ours_config["model"]).to(device)
resnet_model = ModelResNet().to(device)

# 저장된 가중치 로드 (Parallel 모델이어서 module. 제거하여 형식 맞추기)
ours_state_dict = torch.load('/dshome/ddualab/jiwon/skin_cancer_fairness/runs/ours_model_01-21_17-38.pth', weights_only=False)
ours_state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in ours_state_dict.items())

resnet_state_dict = torch.load('/dshome/ddualab/jiwon/skin_cancer_fairness/runs/resnet_model_01-20_16-24.pth', weights_only=False)
resnet_state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in resnet_state_dict.items())

ours_model.load_state_dict(ours_state_dict, strict=False) # ours 11번 실험 / 8번 / 6번 / 9번
resnet_model.load_state_dict(resnet_state_dict, strict=False) # resnet 2번 실험

# test_dataset = pd.read_csv(ours_config['test_csv']) 
# test_wrapper = TestDataSetWrapper(test_dataset, ours_config['batch_size'], **ours_config['dataset'])
# test_loader = test_wrapper.get_data_loaders()


train_dataset = pd.read_csv(ours_config['train_csv']) 
train_wrapper = TrainDataSetWrapper(train_dataset, ours_config['batch_size'], **ours_config['dataset'])
# train_loader = train_wrapper.get_data_loaders()
train_loader, valid_loader = train_wrapper.get_data_loaders()

def extract_embs(model, dataloader, encoder_type):
    embs = []
    model.eval()
    with torch.no_grad(): 
        for idx, (images, phrases1, phrases2, labels, fitz_scales) in enumerate(dataloader):
            images = images.to(device)
            if encoder_type == 'image_encoder_1':
                emb = model.image_encoder_1(images)
            elif encoder_type == 'image_encoder_2':
                emb = model.image_encoder_2(images)
            else:
                emb = model.feature_extractor(images)
            # gap_emb = torch.mean(emb, dim=[2,3]).cpu().numpy()
            flat_emb = emb.view(emb.size(0), -1).cpu().numpy()
            embs.extend(flat_emb)
    return embs 

# 임베딩 저장 
lesion_embs = extract_embs(ours_model, train_loader, encoder_type='image_encoder_1')
skin_color_embs = extract_embs(ours_model, train_loader, encoder_type='image_encoder_2')
resnet_embs = extract_embs(resnet_model, train_loader, encoder_type='resnet')


lesion_emb_path = 'visualization/img_lesion_emb.pt'
skin_color_emb_path = 'visualization/img_skin_color_emb.pt'
resnet_emb_path = 'visualization/img_resnet_emb.pt'

torch.save(lesion_embs, lesion_emb_path)
torch.save(skin_color_embs, skin_color_emb_path)
torch.save(resnet_embs, resnet_emb_path)

# 임베딩 load
lesion_embs = np.array(torch.load(lesion_emb_path, weights_only=False))
skin_color_embs = np.array(torch.load(skin_color_emb_path, weights_only=False))
resnet_embs = np.array(torch.load(resnet_emb_path, weights_only=False))

pca = PCA(n_components=512)
resnet_embs = pca.fit_transform(resnet_embs)
# 전체 임베딩 concat
total_embs = np.concatenate([lesion_embs, skin_color_embs, resnet_embs])
len_lesion = len(lesion_embs)
len_skin = len(skin_color_embs)
len_resnet = len(resnet_embs)

# 전체 2차원 시각화
tsne_2d = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto', perplexity=30, max_iter = 1000)
total_emb_2d = tsne_2d.fit_transform(total_embs)

plt.figure(figsize=(8, 6))
plt.scatter(total_emb_2d[:len_lesion, 0], total_emb_2d[:len_lesion, 1], c='brown', label='Lesion Condition')
plt.scatter(total_emb_2d[len_lesion:len_lesion+len_skin, 0], total_emb_2d[len_lesion:len_lesion+len_skin, 1], c='orange', label='Skin Color')
plt.scatter(total_emb_2d[len_lesion+len_skin:, 0],total_emb_2d[len_lesion+len_skin:, 1], c='gray', label='ResNet')
plt.legend()
plt.savefig('visualization/img_emb2d_p30_gyu.png')

# 전체  3차원 시각화
tsne = TSNE(n_components=3, random_state=42, init='pca', learning_rate='auto', perplexity=30, max_iter=1000)
total_emb_3d = tsne.fit_transform(total_embs)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(total_emb_3d[:864, 0], total_emb_3d[:864, 1], total_emb_3d[:864, 2], c='brown', label='Lesion Condition')
ax.scatter(total_emb_3d[864:1728, 0], total_emb_3d[864:1728, 1], total_emb_3d[864:1728, 2], c='orange', label='Skin Color')
ax.scatter(total_emb_3d[1728:, 0], total_emb_3d[1728:, 1], total_emb_3d[1728:, 2], c='gray', label='ResNet')
ax.legend()
# ax.view_init(elev=90, azim=0)
plt.savefig('visualization/img_emb3d_p30_gyu.png')

#  Ours 모델 (Lesion + Skin) 2D 시각화
ours_embs = np.concatenate([lesion_embs, skin_color_embs])
len_lesion_ours = len(lesion_embs)
len_skin_ours = len(skin_color_embs)

ours_tsne_2d = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto', perplexity=30, max_iter=1000)
ours_emb_2d = ours_tsne_2d.fit_transform(ours_embs)

plt.figure(figsize=(8, 6))
plt.scatter(ours_emb_2d[:len_lesion_ours, 0], ours_emb_2d[:len_lesion_ours, 1], c='brown', label='Lesion Condition')
plt.scatter(ours_emb_2d[len_lesion_ours:, 0], ours_emb_2d[len_lesion_ours:, 1], c='orange', label='Skin Color')
plt.legend()
plt.savefig('visualization/img_emb2d_ours_p30_gyu.png')

#  Ours 모델 (Lesion + Skin) 3D 시각화
ours_tsne = TSNE(n_components=3, random_state=42, init='pca', learning_rate='auto', perplexity=30, max_iter=1000)
ours_emb_3d = ours_tsne.fit_transform(np.concatenate([lesion_embs, skin_color_embs]))

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(ours_emb_3d[:len_lesion_ours, 0], ours_emb_3d[:len_lesion_ours, 1], ours_emb_3d[:len_lesion_ours, 2], c='brown', label='Lesion Condition')
ax.scatter(ours_emb_3d[len_lesion_ours:, 0], ours_emb_3d[len_lesion_ours:, 1], ours_emb_3d[len_lesion_ours:, 2], c='orange', label='Skin Color')
ax.legend()
# ax.view_init(elev=45, azim=20)
# for elev in range(0, 90, 5):  # 0도에서 90도까지 10도 간격
#     for azim in range(0, 360, 5):  # 0도에서 360도까지 10도 간격
#         ax.view_init(elev=elev, azim=azim)
plt.savefig(f'visualization/img_emb3d_ours_p30_gyu.png')

# resnet 2차원 시각화
resnet_tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto', perplexity=30, max_iter=1000)
resnet_emb_2d = resnet_tsne.fit_transform(resnet_embs)

plt.figure(figsize=(8, 6))
plt.scatter(resnet_emb_2d[:, 0], resnet_emb_2d[:, 1], c='gray', label='ResNet')
plt.legend()
plt.savefig('visualization/img_emb2d_resnet_p30_gyu.png')



# #ours pca 2차원 시각화
# pca = PCA(n_components=2)
# ours_emb_2d_pca = pca.fit_transform(np.concatenate([lesion_embs, skin_color_embs]))

# plt.figure(figsize=(8, 6))
# plt.scatter(ours_emb_2d_pca[:864, 0], ours_emb_2d_pca[:864, 1], c='brown', label='Lesion Condition')
# plt.scatter(ours_emb_2d_pca[864:, 0], ours_emb_2d_pca[864:, 1], c='orange', label='Skin Color')
# plt.legend()
# plt.savefig('visualization/img_emb2d_pca_ours.png')

# # ours pca 3차원 시각화
# pca = PCA(n_components=3)
# ours_emb_3d_pca = pca.fit_transform(np.concatenate([lesion_embs, skin_color_embs]))

# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(ours_emb_3d_pca[:864, 0], ours_emb_3d_pca[:864, 1], ours_emb_3d_pca[:864, 2], c='brown', label='Lesion Condition')
# ax.scatter(ours_emb_3d_pca[864:, 0], ours_emb_3d_pca[864:, 1], ours_emb_3d_pca[864:, 2], c='orange', label='Skin Color')
# ax.legend()
# plt.savefig('visualization/img_emb3d_pca_ours.png')

# # total pca 2차원 시각화
# pca = PCA(n_components=2)
# total_emb_2d_pca = pca.fit_transform(total_embs)

# plt.figure(figsize=(8, 6))
# plt.scatter(total_emb_2d_pca[:864, 0], total_emb_2d_pca[:864, 1], c='brown', label='Lesion Condition')
# plt.scatter(total_emb_2d_pca[864:1728, 0], total_emb_2d_pca[864:1728, 1], c='orange', label='Skin Color')
# plt.scatter(total_emb_2d_pca[1728:, 0], total_emb_2d_pca[1728:, 1], c='gray', label='ResNet')
# plt.legend()
# plt.savefig('visualization/img_emb2d_pca.png')

# # total pca 3차원 시각화
# pca = PCA(n_components=3)
# total_emb_3d_pca = pca.fit_transform(total_embs)

# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(total_emb_3d_pca[:864, 0], total_emb_3d_pca[:864, 1], total_emb_3d_pca[:864, 2], c='brown', label='Lesion Condition')
# ax.scatter(total_emb_3d_pca[864:1728, 0], total_emb_3d_pca[864:1728, 1], total_emb_3d_pca[864:1728, 2], c='orange', label='Skin Color')
# ax.scatter(total_emb_3d_pca[1728:, 0], total_emb_3d_pca[1728:, 1], total_emb_3d_pca[1728:, 2], c='gray', label='ResNet')
# ax.legend()
# plt.savefig('visualization/img_emb3d_pca.png')