import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image

class TrainImageDataset(Dataset):
    def __init__(self, csv_file, img_root_dir, text_root_dir, img_path_col, label_col, fitz_scale_col, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_root_dir = img_root_dir        
        self.text_root_dir = text_root_dir
        self.img_path_col = img_path_col
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
        
        labels = self.data_frame.loc[idx, self.label_col]
        
        fitz_scale = self.data_frame.loc[idx, self.fitz_scale_col]
        
        if self.transform:
            image = self.transform(image)
        
        return image, labels, fitz_scale

class TestImageDataset(Dataset):
    def __init__(self, csv_file, img_root_dir, text_root_dir, img_path_col, label_col, fitz_scale_col, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_root_dir = img_root_dir        
        self.text_root_dir = text_root_dir
        self.img_path_col = img_path_col
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
        
        labels = self.data_frame.loc[idx, self.label_col]
        
        fitz_scale = self.data_frame.loc[idx, self.fitz_scale_col]

        if self.transform:
            image = self.transform(image)
        
        return image, labels, fitz_scale

# import yaml  
# if __name__ == '__main__':    
#     config = yaml.load(open("/dshome/ddualab/jiwon/ConVIRT-pytorch/config.yaml", "r"), Loader=yaml.FullLoader)
#     train_dataset = TrainImageDataset(**config['train_cls_dataset'])
#     test_dataset = TestImageDataset(**config['test_cls_dataset'])
#     print(len(train_dataset), len(test_dataset))
#     print(train_dataset[0][2])    
        