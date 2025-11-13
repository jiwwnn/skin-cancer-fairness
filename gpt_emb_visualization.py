import pandas as pd
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from transformers import AutoTokenizer
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
import requests
from openai import api_requestor
matplotlib.use('Agg')
plt.ion()

# # gpt 임베딩 추출
# OPENAI_API_KEY = 'sk-proj-G1BLQt4pDrBeNM3AR2swT3BlbkFJJsDgCsEwiuNzTI6sGdlr'

df = pd.read_csv('/dshome/ddualab/jiwon/skin_cancer_fairness/data/sep_vlm_generated_df.csv')
df = df[~((df['fitzpatrick_scale']== 5.0) | (df['fitzpatrick_scale']== 6.0))]

# session = requests.Session()
# adapter = requests.adapters.HTTPAdapter(max_retries=3)
# session.mount('https://', adapter)

# api_requestor._thread_context.session = session

# def text_preprocessing(col):
#     texts = df[col].astype(str)
#     processed_texts = []
    
#     for text in texts:
#         text = text.replace("\n", "")
#         text = text.split('.')
#         text = [t.strip() for t in text if t.strip()]
#         phrase = ". ".join(text)
#         processed_texts.append(phrase)
        
#     return processed_texts

# lesion_col, fitz_col = 'lesion_description', 'fitzpatrick_description'
# lesion_description = text_preprocessing(lesion_col)
# fitz_description = text_preprocessing(fitz_col)


# def get_gpt_embedding(texts, model_name='text-embedding-ada-002'):
#     embeddings = []
#     for text in texts:
#         response = openai.Embedding.create(
#             input=text,
#             model=model_name,
#             api_key=OPENAI_API_KEY, 
#             timeout=900
#         )
#         embeddings.append(response['data'][0]['embedding'])
#     return embeddings

# def get_gpt_embedding_batch(texts, batch_size=5, model_name='text-embedding-ada-002'):
#     embeddings = []
#     for i in range(0, len(texts), batch_size):
#         batch_texts = texts[i:i + batch_size]
#         response = openai.Embedding.create(
#             input=batch_texts,
#             model=model_name,
#             api_key=OPENAI_API_KEY
#         )
#         embeddings.extend([res['embedding'] for res in response['data']])
#     return embeddings

# lesion_emb = get_gpt_embedding_batch(lesion_description)
# fitz_emb = get_gpt_embedding_batch(fitz_description)

lesion_emb_path = 'visualization/gpt_lesion_emb.pt'
fitz_emb_path = 'visualization/gpt_fitz_emb.pt'

# torch.save(lesion_emb, lesion_emb_path)
# torch.save(fitz_emb, fitz_emb_path)

lesion_emb = np.array(torch.load(lesion_emb_path))
fitz_emb = np.array(torch.load(fitz_emb_path))
total_emb = np.concatenate([lesion_emb, fitz_emb])

# # 2차원 시각화
# tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto', perplexity=30)
# total_emb_2d = tsne.fit_transform(total_emb)

# plt.figure(figsize=(8, 6))
# plt.scatter(total_emb_2d[:1494, 0], total_emb_2d[:1494, 1], c='brown', label='Lesion Description')
# plt.scatter(total_emb_2d[1494:, 0], total_emb_2d[1494:, 1], c='orange', label='Fitzpatrick Description')
# plt.legend()
# plt.savefig('gpt_emb2d.png')
    
    
# # 3차원 시각화
# tsne = TSNE(n_components=3, random_state=42, init='pca', learning_rate='auto', perplexity=30)
# total_emb_3d = tsne.fit_transform(total_emb)

# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(total_emb_3d[:1494, 0], total_emb_3d[:1494, 1], total_emb_3d[:1494, 2], c='brown', label='Lesion Description')
# ax.scatter(total_emb_3d[1494:, 0], total_emb_3d[1494:, 1], total_emb_3d[1494:, 2], c='orange', label='Fitzpatrick Description')
# ax.legend()
# # ax.view_init(elev=90, azim=0)
# plt.savefig('gpt_emb3d.png')

## 클래스별 시각화
lesion_color_dict = {'ACK':'#87CEEB', 'NEV':'#4169E1', 'SEK':'#00008B', 'BCC':'#F08080', 'MEL':'#DC143C', 'SCC':'#8B0000'}
fitz_color_dict={1.0:'#D3D3D3', 2.0:'#A9A9A9', 3.0:'#696969', 4.0:'#505050'}

# 2차원 시각화
tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto', perplexity=30)
total_emb_2d = tsne.fit_transform(total_emb)

plt.figure(figsize=(8, 6))
for lesion_label in set(df['label']):
    idx = [idx for idx, value in enumerate(df['label']) if value == lesion_label] 
    plt.scatter(total_emb_2d[idx, 0], total_emb_2d[idx, 1], c=lesion_color_dict[lesion_label])
    
for fitz_label in set(df['fitzpatrick_scale']):
    idx = [idx for idx, value in enumerate(df['fitzpatrick_scale']) if value == fitz_label] 
    plt.scatter(total_emb_2d[[i+len(lesion_emb) for i in idx], 0], total_emb_2d[[i+len(lesion_emb) for i in idx], 1], c=fitz_color_dict[fitz_label])
plt.savefig('visualization/gpt_emb2d_classes_.png')
    

# 3차원 시각화
tsne = TSNE(n_components=3, random_state=42, init='pca', learning_rate='auto', perplexity=30)
total_emb_3d = tsne.fit_transform(total_emb)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
for lesion_label in set(df['label']):
    idx = [idx for idx, value in enumerate(df['label']) if value == lesion_label] 
    ax.scatter(total_emb_3d[idx, 0], total_emb_3d[idx, 1], total_emb_3d[idx, 2], c=lesion_color_dict[lesion_label])
    
for fitz_label in set(df['fitzpatrick_scale']):
    idx = [idx for idx, value in enumerate(df['fitzpatrick_scale']) if value == fitz_label] 
    ax.scatter(total_emb_3d[[i+len(lesion_emb) for i in idx], 0], total_emb_3d[[i+len(lesion_emb) for i in idx], 1], total_emb_3d[[i+len(lesion_emb) for i in idx], 2], c=fitz_color_dict[fitz_label])
# ax.view_init(elev=90, azim=0)
plt.savefig('visualization/gpt_emb3d_classes.png')