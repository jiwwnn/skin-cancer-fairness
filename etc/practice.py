# import torch 

# model = torch.load('/dshome/ddualab/jiwon/ConVIRT-pytorch/runs/Jun19_14-08-58_daintlabB/checkpoints/image_encoder1.pth')
# print(model)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from tqdm import tqdm
import os
import yaml

from dataloader.dataset_wrapper import DataSetWrapper, SimCLRDataTransform
from dataloader.cls_dataset_wrapper import ClsDataSetWrapper

config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
cls_dataset_wrapper = ClsDataSetWrapper(config['batch_size'], **config['cls_dataset'])
cls_train_loader, cls_valid_loader = cls_dataset_wrapper.get_data_loaders()

print(len(cls_valid_loader))    