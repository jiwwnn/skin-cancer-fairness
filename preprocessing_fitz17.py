import os
import pandas as pd

img_folder = 'data/fitzpatrick17_images'
img_list = []
for img in os.listdir(img_folder):
    img_list.append(img)
# print(len(img_list))

img_count = sum(1 for img in os.listdir(img_folder))
# print(img_count)
# 전체 16577 -> 16518

img_ids = set()
for img in os.listdir(img_folder):
    img_id = os.path.splitext(img)[0]
    img_ids.add(img_id)

# print(len(img_ids))

df = pd.read_csv('/dshome/ddualab/jiwon/skin_cancer_fairness/data/fitz17_metadata.csv')
# print(df.head(5))

df.rename(columns={'md5hash':'img_id', 'label':'detailed_label', 'three_partition_label':'label'}, inplace=True)

df[['lesion_description', 'fitzpatrick_description']] = 'There is no available data.'
# df = df[~((df['fitzpatrick_scale']==-1)|(df['label']=='non-neoplastic'))]
# print(len(df)) # Fitz 누락 & 염증성 제외: 4320개
df = df[~((df['fitzpatrick_scale']==-1)|(df['fitzpatrick_scale']==5)|(df['fitzpatrick_scale']==6)|(df['label']=='non-neoplastic'))]
# df = df[~((df['fitzpatrick_scale']==-1)|(df['fitzpatrick_scale']==5)|(df['fitzpatrick_scale']==6)|(df['fitzpatrick_scale']==1)|(df['fitzpatrick_scale']==2)|(df['label']=='non-neoplastic'))]

binary_label = []

for label in df['label']:
    if label == 'benign':
        binary_label.append(0)
    else:
        binary_label.append(1)
        
df['label'] = binary_label

jpg_img_id = []

for i in df['img_id']:
    jpg_img_id.append(i + '.jpg')

df['img_id'] = jpg_img_id
# print(len((df['img_id'])))

not_exist_list = []
for i in df['img_id']:
    if i not in img_list:
        not_exist_list.append(i)
# print(not_exist_list)
print(len(not_exist_list))

df = df.drop(df[df["img_id"].isin(not_exist_list)].index)
print(len(df))
print(len((df[(df['label']==1) & (df['fitzpatrick_scale']==1)])))
# # df.to_csv('data/in_b_fitz17_test_only_df.csv', index=False)
# df.to_csv('data/out_fitz34_b_fitz17_test_only_df.csv', index=False)