import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image


class TrainDataset(Dataset):
    """Contrastive Learning Representations Dataset."""

    def __init__(self, csv_file, img_root_dir, input_shape, img_path_col, text_root_dir, label_col, fitz_scale_col, label_text_col, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_root_dir = img_root_dir
        self.input_shape = input_shape
        self.img_path_col = img_path_col
        self.text_root_dir = text_root_dir   
        self.label_col = label_col
        self.fitz_scale_col = fitz_scale_col
        self.label_text_col = label_text_col
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.img_root_dir, self.data_frame.loc[idx, self.img_path_col])
        image = Image.open(img_name).convert('RGB')
        
        label = self.data_frame.loc[idx, self.label_col]
        fitz_scale = self.data_frame.loc[idx, self.fitz_scale_col]
        label_text = self.data_frame.loc[idx, self.label_text_col]
        
        sample = {'image': image, 'label':label, 'fitz_scale':fitz_scale, 'label_text':label_text}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
class TestDataset(Dataset):
    def __init__(self, csv_file, img_root_dir, input_shape, img_path_col, text_root_dir, label_col, fitz_scale_col, label_text_col, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_root_dir = img_root_dir
        self.input_shape = input_shape
        self.img_path_col = img_path_col
        self.text_root_dir = text_root_dir
        self.label_col = label_col
        self.fitz_scale_col = fitz_scale_col
        self.label_text_col = label_text_col
        self.transform = transform
         
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.img_root_dir, self.data_frame.loc[idx, self.img_path_col])
        image = Image.open(img_name).convert('RGB')
        
        label = self.data_frame.loc[idx, self.label_col]
        fitz_scale = self.data_frame.loc[idx, self.fitz_scale_col]
        label_text = self.data_frame.loc[idx, self.label_text_col]
        
        sample = {'image': image, 'label':label, 'fitz_scale':fitz_scale, 'label_text':label_text}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
         
    
import yaml  
if __name__ == '__main__':    
    config = yaml.load(open("/dshome/ddualab/jiwon/ConVIRT-pytorch/config.yaml", "r"), Loader=yaml.FullLoader)
    sample = TrainDataset(**config['train_dataset'])
    print(sample[0])
    print(sample[1])
