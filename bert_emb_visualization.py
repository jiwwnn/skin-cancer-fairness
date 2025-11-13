import pandas as pd
import numpy as np
import matplotlib
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import yaml
from transformers import AutoTokenizer
from models.ours_model import ModelOurs
matplotlib.use('Agg')
plt.ion()

config = yaml.load(open("ours_config.yaml", "r"), Loader=yaml.FullLoader)

df = pd.read_csv('/dshome/ddualab/jiwon/skin_cancer_fairness/data/sep_vlm_generated_df.csv')
df = df[~((df['fitzpatrick_scale']== 5.0) | (df['fitzpatrick_scale']== 6.0))]

def text_preprocessing(col):
    texts = df[col].astype(str)
    processed_texts = []
    
    for text in texts:
        text = text.replace("\n", "")
        text = text.split('.')
        text = [t.strip() for t in text if t.strip()]
        phrase = ". ".join(text)
        processed_texts.append(phrase)
        
    return processed_texts

lesion_col, fitz_col = 'lesion_description', 'fitzpatrick_description'
lesion_description = text_preprocessing(lesion_col)
fitz_description = text_preprocessing(fitz_col)

tokenizer = AutoTokenizer.from_pretrained(config['model']['bert_base_model'])
lesion_tokenized = tokenizer(lesion_description, return_tensors='pt', padding=True, truncation=config['truncation'])
fitz_tokenized = tokenizer(fitz_description, return_tensors='pt', padding=True, truncation=config['truncation'])

model = ModelOurs(**config["model"])

lesion_emb = model.text_encoder(lesion_tokenized).detach().numpy()
fitz_emb = model.text_encoder(fitz_tokenized).detach().numpy()

lesion_emb_path = 'visualization/clinical_bert_lesion_emb_final.pt'
fitz_emb_path = 'visualization/clinical_bert_fitz_emb_final.pt'

torch.save(lesion_emb, lesion_emb_path)
torch.save(fitz_emb, fitz_emb_path)

lesion_emb = np.array(torch.load(lesion_emb_path))
fitz_emb = np.array(torch.load(fitz_emb_path))
total_emb = np.concatenate([lesion_emb, fitz_emb])

# print(len(total_emb))

# 2차원 시각화
tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto', perplexity=40)
total_emb_2d = tsne.fit_transform(total_emb)

plt.figure(figsize=(8, 6))
plt.scatter(total_emb_2d[:1494, 0], total_emb_2d[:1494, 1], c='brown', label='Lesion Description')
plt.scatter(total_emb_2d[1494:, 0], total_emb_2d[1494:, 1], c='orange', label='Skin Color Description')
plt.legend()
plt.savefig('visualization/clinical_bert_emb2d_final_v2.png')
    
    
# 3차원 시각화
tsne = TSNE(n_components=3, random_state=42, init='pca', learning_rate='auto', perplexity=40)
total_emb_3d = tsne.fit_transform(total_emb)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(total_emb_3d[:1494, 0], total_emb_3d[:1494, 1], total_emb_3d[:1494, 2], c='brown', label='Lesion Description')
ax.scatter(total_emb_3d[1494:, 0], total_emb_3d[1494:, 1], total_emb_3d[1494:, 2], c='orange', label='Skin Color Description')
ax.legend()
# ax.view_init(elev=90, azim=0)
plt.savefig('visualization/clinical_bert_emb3d_final_v2.png')


# ## 클래스별 시각화
# lesion_color_dict = {'ACK':'#87CEEB', 'NEV':'#4169E1', 'SEK':'#00008B', 'BCC':'#F08080', 'MEL':'#DC143C', 'SCC':'#8B0000'}
# fitz_color_dict={1.0:'#D3D3D3', 2.0:'#A9A9A9', 3.0:'#696969', 4.0:'#505050'}

# # 2차원 시각화
# tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto', perplexity=30)
# total_emb_2d = tsne.fit_transform(total_emb)

# plt.figure(figsize=(8, 6))
# for lesion_label in set(df['label']):
#     idx = [idx for idx, value in enumerate(df['label']) if value == lesion_label] 
#     plt.scatter(total_emb_2d[idx, 0], total_emb_2d[idx, 1], c=lesion_color_dict[lesion_label])
    
# for fitz_label in set(df['fitzpatrick_scale']):
#     idx = [idx for idx, value in enumerate(df['fitzpatrick_scale']) if value == fitz_label] 
#     plt.scatter(total_emb_2d[[i+len(lesion_emb) for i in idx], 0], total_emb_2d[[i+len(lesion_emb) for i in idx], 1], c=fitz_color_dict[fitz_label])
# plt.savefig('visualization/clinical_bert_emb2d_classes_.png')
    

# # 3차원 시각화
# tsne = TSNE(n_components=3, random_state=42, init='pca', learning_rate='auto', perplexity=30)
# total_emb_3d = tsne.fit_transform(total_emb)

# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# for lesion_label in set(df['label']):
#     idx = [idx for idx, value in enumerate(df['label']) if value == lesion_label] 
#     ax.scatter(total_emb_3d[idx, 0], total_emb_3d[idx, 1], total_emb_3d[idx, 2], c=lesion_color_dict[lesion_label])
    
# for fitz_label in set(df['fitzpatrick_scale']):
#     idx = [idx for idx, value in enumerate(df['fitzpatrick_scale']) if value == fitz_label] 
#     ax.scatter(total_emb_3d[[i+len(lesion_emb) for i in idx], 0], total_emb_3d[[i+len(lesion_emb) for i in idx], 1], total_emb_3d[[i+len(lesion_emb) for i in idx], 2], c=fitz_color_dict[fitz_label])
# # ax.view_init(elev=90, azim=0)
# plt.savefig('visualization/clinical_bert_emb3d_classes.png')