import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('/dshome/ddualab/jiwon/ConVIRT-pytorch/data/description_df.csv')
df = df[~((df['fitzpatrick_scale']== 5.0) | (df['fitzpatrick_scale']== 6.0))]

# binary classification
# ben = ['ACK', 'NEV', 'SEK']
# mal = ['BCC', 'MEL', 'SCC']
# binary_label = []

# for label in df['label']:
#     if label in ben:
#         binary_label.append(0)
#     elif label in mal:
#         binary_label.append(1)

# df['label'] = binary_label

# multiclass_classification
multi_label = []
for label in df['label']:
    if label == 'ACK':
        multi_label.append(0)
    elif label == 'NEV':
        multi_label.append(1)
    elif label == 'SEK':
        multi_label.append(2)
    elif label == 'BCC':
        multi_label.append(3)
    elif label == 'MEL':
        multi_label.append(4)
    else:
        multi_label.append(5)
        
df['label'] = multi_label


train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[['fitzpatrick_scale', 'label']])
# print(len(train_df))
# print(train_df['fitzpatrick_scale'].value_counts())
# print(train_df['label'].value_counts())

# print(len(test_df))
# print(test_df['fitzpatrick_scale'].value_counts())
# print(test_df['label'].value_counts())

train_df.to_csv('data/f_m_train_df.csv', index=False)
test_df.to_csv('data/f_m_test_df.csv', index=False)