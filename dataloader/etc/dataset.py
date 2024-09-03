import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image, ImageFile
import random
import pickle

# ImageFile.LOAD_TRUNCATED_IMAGES = True

class ClrDataset(Dataset):
    """Contrastive Learning Representations Dataset."""

    def __init__(self, 
                csv_file, 
                img_root_dir, 
                input_shape, 
                img_path_col, 
                text_col1,
                text_col2, 
                text_from_files,
                text_root_dir, 
                transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_root_dir (string): Directory with all the images.
            input_shape: shape of input image
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.img_root_dir = img_root_dir
        self.input_shape = input_shape
        self.img_path_col = img_path_col
        self.text_col1 = text_col1, 
        self.text_col2 = text_col2,       
        self.text_from_files = text_from_files 
        self.text_root_dir = text_root_dir
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
        phrase1 = random.sample(text1, min(4, len(text1)))
        phrase1 = ". ".join(phrase1)

        text2 = self.data_frame.loc[idx, self.text_col2]
        text2 = text2.replace("\n", "")
        text2 = text2.split('.')
        text2 = [t.strip() for t in text2 if t.strip()]
        phrase2 = random.sample(text2, min(4, len(text2)))  
        phrase2 = ". ".join(phrase2)
              
            
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        
        # img_name = os.path.join(self.img_root_dir,
        #                         self.data_frame.iloc[idx, self.img_path_col]
        #                         )
        # image = Image.open(img_name)
        # if self.input_shape[2] == 3:
        #     image = image.convert('RGB')
        # phrase = ''

        # #chooosig a phrase
        # try:            
        #     if not self.text_from_files:
        #         text1 = self.data_frame.iloc[idx, self.text_col1]
        #         text1 = text1.replace("\n", "")
        #         ls_text1 = text1.split(".")
        #         if '' in ls_text1:
        #             ls_text1.remove('')
        #         phrase1 = random.choice(ls_text1)
                
        #         text2 = self.data_frame.iloc[idx, self.text_col2]
        #         text2 = text2.replace("\n", "")
        #         ls_text2 = text2.split(".")
        #         if '' in ls_text2:
        #             ls_text2.remove('')
        #         phrase2 = random.choice(ls_text2)

        #     else:
        #         text_path1 = os.path.join(self.text_root_dir, 
        #                                 self.data_frame.iloc[idx, self.text_col1]
        #                                 )
                
        #         with open(text_path1) as f:
        #             content1 = f.readlines()
        #         content1 = content1.replace("\n", "")
        #         ls_text1 = content1.split(".")
        #         if '' in ls_text1:
        #             ls_text1.remove('')
        #         phrase1 = random.choice(ls_text1)
                
        #         text_path2 = os.path.join(self.text_root_dir, 
        #                                 self.data_frame.iloc[idx, self.text_col2]
        #                                 )
                
        #         with open(text_path2) as f:
        #             content2 = f.readlines()
        #         content2 = content2.replace("\n", "")
        #         ls_text2 = content2.split(".")
        #         if '' in ls_text2:
        #             ls_text2.remove('')
        #         phrase2 = random.choice(ls_text2)

        # except IndexError as e:
        #     pass
        # tokens1 = self.tokenizer(phrase1, return_tensors="pt", padding=True, truncation=True)
        # tokens2 = self.tokenizer(phrase2, return_tensors="pt", padding=True, truncation=True)
        
        sample = {'image': image, 'phrase1': phrase1, 'phrase2' : phrase2}
                
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
import yaml  
if __name__ == '__main__':    
    config = yaml.load(open("/dshome/ddualab/jiwon/ConVIRT-pytorch/config.yaml", "r"), Loader=yaml.FullLoader)
    dataset = ClrDataset(**config['dataset'])
    sample = dataset[3]
    print(sample['phrase1'])
    print(len([x for x in sample['phrase1'].split('. ')]))
        

    