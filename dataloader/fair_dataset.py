import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import random


# ImageFile.LOAD_TRUNCATED_IMAGES = True

class TrainDataset(Dataset):
    """Contrastive Learning Representations Dataset."""

    def __init__(self, csv_file, img_root_dir, input_shape, img_path_col, text_root_dir, text_col1, text_col2, label_col, fitz_scale_col, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_root_dir = img_root_dir
        self.input_shape = input_shape
        self.img_path_col = img_path_col
        self.text_root_dir = text_root_dir
        self.text_col1 = text_col1
        self.text_col2 = text_col2     
        self.label_col = label_col
        self.fitz_scale_col = fitz_scale_col
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.img_root_dir, self.data_frame.loc[idx, self.img_path_col])
        image = Image.open(img_name).convert('RGB')
        
        text1 = self.data_frame.loc[idx, self.text_col1]
        text1 = text1.replace("\n", "")
        text1 = text1.split('.')
        text1 = [t.strip() for t in text1 if t.strip()]
        phrase1 = ". ".join(text1)
        # phrase1 = random.sample(text1, min(4, len(text1)))
        # phrase1 = ". ".join(phrase1)

        text2 = self.data_frame.loc[idx, self.text_col2]
        text2 = text2.replace("\n", "")
        text2 = text2.split('.')
        text2 = [t.strip() for t in text2 if t.strip()]
        phrase2 = ". ".join(text2)
        # phrase2 = random.sample(text2, min(4, len(text2)))  
        # phrase2 = ". ".join(phrase2)
        
        label = self.data_frame.loc[idx, self.label_col]
        fitz_scale = self.data_frame.loc[idx, self.fitz_scale_col]
        
        sample = {'image': image, 'phrase1': phrase1, 'phrase2' : phrase2, 'label':label, 'fitz_scale':fitz_scale}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
class TestDataset(Dataset):
    def __init__(self, csv_file, img_root_dir, input_shape, img_path_col, text_root_dir, text_col1, text_col2, label_col, fitz_scale_col, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_root_dir = img_root_dir
        self.input_shape = input_shape
        self.img_path_col = img_path_col
        self.text_root_dir = text_root_dir
        self.text_col1 = text_col1
        self.text_col2 = text_col2    
        self.label_col = label_col
        self.fitz_scale_col = fitz_scale_col
        self.transform = transform
         
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.img_root_dir, self.data_frame.loc[idx, self.img_path_col])
        image = Image.open(img_name).convert('RGB')
        
        text1 = self.data_frame.loc[idx, self.text_col1]
        text1 = text1.replace("\n", "")
        text1 = text1.split('.')
        text1 = [t.strip() for t in text1 if t.strip()]
        phrase1 = ". ".join(text1)
        # phrase1 = random.sample(text1, min(4, len(text1)))
        # phrase1 = ". ".join(phrase1)

        text2 = self.data_frame.loc[idx, self.text_col2]
        text2 = text2.replace("\n", "")
        text2 = text2.split('.')
        text2 = [t.strip() for t in text2 if t.strip()]
        phrase2 = ". ".join(text2)
        # phrase2 = random.sample(text2, min(4, len(text2)))  
        # phrase2 = ". ".join(phrase2)
        
        label = self.data_frame.loc[idx, self.label_col]
        fitz_scale = self.data_frame.loc[idx, self.fitz_scale_col]
        
        sample = {'image': image, 'phrase1': phrase1, 'phrase2' : phrase2, 'label':label, 'fitz_scale':fitz_scale}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
         
    
import yaml  
if __name__ == '__main__':    
    config = yaml.load(open("/dshome/ddualab/jiwon/ConVIRT-pytorch/config.yaml", "r"), Loader=yaml.FullLoader)
    sample = TrainDataset(**config['train_dataset'])
    print(sample[0])
    print(sample[1])
