import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from models.ours_model import ModelOurs
from models.resnet_model import ModelResNet
from models.disco_model import ModelDisCo
from models.patchalign_model import ModelPatchAlign
from dataloader.ours_dataset_wrapper import TrainDataSetWrapper, TestDataSetWrapper
import yaml
from collections import OrderedDict
matplotlib.use('Agg')
plt.ion()

ours_config = yaml.load(open("ours_config.yaml", "r"), Loader=yaml.FullLoader)
patchalign_config = yaml.load(open('patchalign_config.yaml', 'r'), Loader=yaml.FullLoader)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 정의
ours_model = ModelOurs(**ours_config["model"]).to(device)
resnet_model = ModelResNet().to(device)
disco_model = ModelDisCo().to(device)
patchalign_model = ModelPatchAlign(**patchalign_config["model"]).to(device)
# resnet_with_test = ResNetWithLinear(resnet_model, target_dim=512)

# 저장된 가중치 로드 (Parallel 모델이어서 module. 제거하여 형식 맞추기)
# Ours
ours_state_dict = torch.load('/dshome/ddualab/jiwon/skin_cancer_fairness/runs/ours_model_01-26_22-52.pth', weights_only=False)
ours_state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in ours_state_dict.items())

# ResNet
resnet_state_dict = torch.load('/dshome/ddualab/jiwon/skin_cancer_fairness/runs/resnet_model_01-26_18-39.pth', weights_only=False)
resnet_state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in resnet_state_dict.items())

# FairDisCo
disco_state_dict = torch.load('/dshome/ddualab/jiwon/skin_cancer_fairness/runs/disco_model_01-26_23-02.pth', weights_only=False)
disco_state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in disco_state_dict.items())

# PatchAlign
patchalign_state_dict = torch.load('/dshome/ddualab/jiwon/skin_cancer_fairness/runs/patchalign_model_01-28_15-01.pth', weights_only=False)
patchalign_state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in patchalign_state_dict.items())

ours_model.load_state_dict(ours_state_dict, strict=False) 
resnet_model.load_state_dict(resnet_state_dict, strict=False)
disco_model.load_state_dict(resnet_state_dict, strict=False)
patchalign_model.load_state_dict(resnet_state_dict, strict=False)

test_dataset = pd.read_csv(ours_config['test_csv']) 
test_wrapper = TestDataSetWrapper(test_dataset, ours_config['batch_size'], ours_config['pad_img_root_dir'], **ours_config['dataset'])
test_loader = test_wrapper.get_data_loaders()

# train_dataset = pd.read_csv(ours_config['train_csv']) 
# train_wrapper = TrainDataSetWrapper(train_dataset, ours_config['batch_size'], **ours_config['dataset'])
# train_loader, valid_loader = train_wrapper.get_data_loaders()

def extract_embs(model, dataloader, encoder_type):
    embs = []
    model.eval() # 평가 모드로 설정
    with torch.no_grad(): # 그래디언트 계산 비활성화
        for idx, (images, phrases1, phrases2, labels, fitz_scales) in enumerate(dataloader):
            images = images.to(device)
            if encoder_type == 'image_encoder_1':
                emb = model.image_encoder_1(images)
            elif encoder_type == 'image_encoder_2':
                emb = model.image_encoder_2(images)
            elif encoder_type == 'patchalign':
                emb = model.feature_extractor(images).last_hidden_state
            else:
                emb = model.feature_extractor(images)
            # gap_emb = torch.mean(emb, dim=[2,3]).cpu().numpy()
            flat_emb = emb.view(emb.size(0), -1).cpu().numpy()
            embs.extend(flat_emb)
    return embs 

lesion_embs = extract_embs(ours_model, test_loader, encoder_type='image_encoder_1')
skin_color_embs = extract_embs(ours_model, test_loader, encoder_type='image_encoder_2')
resnet_embs = extract_embs(resnet_model, test_loader, encoder_type='resnet')
disco_embs = extract_embs(disco_model, test_loader, encoder_type='disco')
patchalign_embs = extract_embs(patchalign_model, test_loader, encoder_type='patchalign')

lesion_emb_path = 'visualization/img_lesion_emb_encoder1.pt'
skin_color_emb_path = 'visualization/img_skin_color_emb_encoder2.pt'
resnet_emb_path = 'visualization/img_resnet_emb.pt'
disco_emb_path = 'visualization/img_disco_emb.pt'
patchalign_emb_path = 'visualization/img_patchalign_emb.pt'

torch.save(lesion_embs, lesion_emb_path)
torch.save(skin_color_embs, skin_color_emb_path)
torch.save(resnet_embs, resnet_emb_path)
torch.save(disco_embs, disco_emb_path)
torch.save(patchalign_embs, patchalign_emb_path)

#########

lesion_embs = np.array(torch.load(lesion_emb_path, weights_only=False))
skin_color_embs = np.array(torch.load(skin_color_emb_path, weights_only=False))
resnet_embs = np.array(torch.load(resnet_emb_path, weights_only=False))
disco_embs = np.array(torch.load(disco_emb_path, weights_only=False))
patchalign_embs = np.array(torch.load(patchalign_emb_path, weights_only=False))

# print(len(lesion_embs), len(resnet_embs))

# # ResNet 임베딩 크기 PCA로 축소
# pca = PCA(n_components=512)
# resnet_embs = pca.fit_transform(resnet_embs)

target_dim = 128
pca = PCA(n_components=target_dim)

lesion_embs = pca.fit_transform(lesion_embs)
skin_color_embs = pca.fit_transform(skin_color_embs)
resnet_embs = pca.fit_transform(resnet_embs)
disco_embs = pca.fit_transform(disco_embs)
patchalign_embs = pca.fit_transform(patchalign_embs)

total_embs = np.concatenate([lesion_embs, skin_color_embs, resnet_embs])
# total_embs = np.concatenate([lesion_embs, skin_color_embs, patchalign_embs])

# total 2차원 시각화
tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto', perplexity=30)
total_emb_2d = tsne.fit_transform(total_embs)

plt.figure(figsize=(8, 6))
plt.scatter(total_emb_2d[:len(lesion_embs), 0], total_emb_2d[:len(lesion_embs), 1], c='brown', label='Ours_Lesion Condition')
plt.scatter(total_emb_2d[len(lesion_embs):len(lesion_embs)*2, 0], total_emb_2d[len(lesion_embs):len(lesion_embs)*2, 1], c='orange', label='Ours_Skin Color')
plt.scatter(total_emb_2d[len(lesion_embs)*2:, 0], total_emb_2d[len(lesion_embs)*2:, 1], c='gray', label='BASE')
# plt.scatter(total_emb_2d[len(lesion_embs):, 0], total_emb_2d[len(lesion_embs):, 1], c='pink', label='FairDisCo')
# plt.scatter(total_emb_2d[len(lesion_embs)*2:, 0], total_emb_2d[len(lesion_embs)*2:, 1], c='violet', label='PatchAlign')
plt.legend()
plt.savefig('visualization/img_resnet_ours_emb2d_p30_pca128.png')

# # total 3차원 시각화
# tsne = TSNE(n_components=3, random_state=42, init='pca', learning_rate='auto', perplexity=30)
# total_emb_3d = tsne.fit_transform(total_embs)

# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(total_emb_3d[:len(lesion_embs), 0], total_emb_3d[:len(lesion_embs), 1], total_emb_3d[:len(lesion_embs), 2], c='brown', label='Lesion Condition')
# ax.scatter(total_emb_3d[len(lesion_embs):len(lesion_embs)*2, 0], total_emb_3d[len(lesion_embs):len(lesion_embs)*2, 1], total_emb_3d[len(lesion_embs):len(lesion_embs)*2, 2], c='orange', label='Skin Color')
# ax.scatter(total_emb_3d[len(lesion_embs)*2:, 0], total_emb_3d[len(lesion_embs)*2:, 1], total_emb_3d[len(lesion_embs)*2:, 2], c='gray', label='ResNet')
# ax.legend()
# # ax.view_init(elev=90, azim=0)
# plt.savefig('visualization/img_emb3d_p30_train_pca128.png')

# # ours 2차원 시각화
# ours_tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto', perplexity=30)
# ours_emb_2d = ours_tsne.fit_transform(np.concatenate([lesion_embs, skin_color_embs]))

# plt.figure(figsize=(8, 6))
# plt.scatter(ours_emb_2d[:len(lesion_embs), 0], ours_emb_2d[:len(lesion_embs), 1], c='brown', label='Lesion Condition')
# plt.scatter(ours_emb_2d[len(lesion_embs):, 0], ours_emb_2d[len(lesion_embs):, 1], c='orange', label='Skin Color')
# plt.legend()
# plt.savefig('visualization/img_emb2d_ours_p30_enc_test.png')

# # ours 3차원 시각화
# ours_tsne = TSNE(n_components=3, random_state=42, init='pca', learning_rate='auto', perplexity=30)
# ours_emb_3d = ours_tsne.fit_transform(np.concatenate([lesion_embs, skin_color_embs]))

# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(ours_emb_3d[:len(lesion_embs), 0], ours_emb_3d[:len(lesion_embs), 1], ours_emb_3d[:len(lesion_embs), 2], c='brown', label='Lesion Condition')
# ax.scatter(ours_emb_3d[len(lesion_embs):, 0], ours_emb_3d[len(lesion_embs):, 1], ours_emb_3d[len(lesion_embs):, 2], c='orange', label='Skin Color')
# ax.legend()
# # ax.view_init(elev=45, azim=20)
# # for elev in range(0, 90, 5):  # 0도에서 90도까지 10도 간격
# #     for azim in range(0, 360, 5):  # 0도에서 360도까지 10도 간격
# #         ax.view_init(elev=elev, azim=azim)
# plt.savefig(f'visualization/img_emb3d_ours_p30_enc_test.png')

# # resnet 2차원 시각화
# resnet_tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto', perplexity=30)
# resnet_emb_2d = resnet_tsne.fit_transform(resnet_embs)

# plt.figure(figsize=(8, 6))
# plt.scatter(resnet_emb_2d[:864, 0], resnet_emb_2d[:864, 1], c='gray', label='ResNet')
# plt.legend()
# plt.savefig('visualization/img_emb2d_resnet_p30_enc_test.png')

# # ours pca 2차원 시각화
# pca = PCA(n_components=2)
# ours_emb_2d_pca = pca.fit_transform(np.concatenate([lesion_embs, skin_color_embs]))

# plt.figure(figsize=(8, 6))
# plt.scatter(ours_emb_2d_pca[:len(lesion_embs), 0], ours_emb_2d_pca[:len(lesion_embs), 1], c='brown', label='Lesion Condition')
# plt.scatter(ours_emb_2d_pca[len(lesion_embs):, 0], ours_emb_2d_pca[len(lesion_embs):, 1], c='orange', label='Skin Color')
# plt.legend()
# plt.savefig('visualization/img_emb2d_pca_ours_enc_test.png')

# # ours pca 3차원 시각화
# pca = PCA(n_components=3)
# ours_emb_3d_pca = pca.fit_transform(np.concatenate([lesion_embs, skin_color_embs]))

# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(ours_emb_3d_pca[:len(lesion_embs), 0], ours_emb_3d_pca[:len(lesion_embs), 1], ours_emb_3d_pca[:len(lesion_embs), 2], c='brown', label='Lesion Condition')
# ax.scatter(ours_emb_3d_pca[len(lesion_embs):, 0], ours_emb_3d_pca[len(lesion_embs):, 1], ours_emb_3d_pca[len(lesion_embs):, 2], c='orange', label='Skin Color')
# ax.legend()
# plt.savefig('visualization/img_emb3d_pca_ours_enc_test.png')

# # total pca 2차원 시각화
# pca = PCA(n_components=2)
# total_emb_2d_pca = pca.fit_transform(total_embs)

# plt.figure(figsize=(8, 6))
# plt.scatter(total_emb_2d_pca[:864, 0], total_emb_2d_pca[:864, 1], c='brown', label='Lesion Condition')
# plt.scatter(total_emb_2d_pca[864:1728, 0], total_emb_2d_pca[864:1728, 1], c='orange', label='Skin Color')
# plt.scatter(total_emb_2d_pca[1728:, 0], total_emb_2d_pca[1728:, 1], c='gray', label='ResNet')
# plt.legend()
# plt.savefig('visualization/img_emb2d_pca_enc_test.png')

# # total pca 3차원 시각화
# pca = PCA(n_components=3)
# total_emb_3d_pca = pca.fit_transform(total_embs)

# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(total_emb_3d_pca[:864, 0], total_emb_3d_pca[:864, 1], total_emb_3d_pca[:864, 2], c='brown', label='Lesion Condition')
# ax.scatter(total_emb_3d_pca[864:1728, 0], total_emb_3d_pca[864:1728, 1], total_emb_3d_pca[864:1728, 2], c='orange', label='Skin Color')
# ax.scatter(total_emb_3d_pca[1728:, 0], total_emb_3d_pca[1728:, 1], total_emb_3d_pca[1728:, 2], c='gray', label='ResNet')
# ax.legend()
# plt.savefig('visualization/img_emb3d_pca_enc_test.png')