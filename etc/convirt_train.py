import torch
import torch.nn as nn
# from models.resnet_clr import ResNetSimCLR
from skin_cancer_fairness.etc.convirt_model import ModelCLCLS
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from loss.f_nt_xent import NTXentLoss
# from loss.l2_distance import L2DistanceLoss
import numpy as np
import os
import shutil
import sys
from tqdm import tqdm
from transformers import AdamW
from transformers import AutoTokenizer
from sklearn.metrics import f1_score
import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys, os

# apex_support = False
# try:
#     sys.path.append('./apex')
#     from apex import amp

#     apex_support = True
# except:
#     print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
#     apex_support = False

torch.manual_seed(0)

def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))
        
# def save_resnet_weights(model, path):
#     state_dict = model.state_dict()
#     resnet_keys = {k:v for k, v in state_dict.items() if 'res_features1' in k or 'res_l1_1' in k or 'res_l2_1' in k}
#     torch.save(resnet_keys, path)

class SimCLR(object):
    def __init__(self, config):
        self.config = config
        self.device = self._get_device()
        self.writer = SummaryWriter()
        self.nt_xent_criterion = NTXentLoss(self.device, config['batch_size'], **config['cl_loss'])
        self.cls_criterion = nn.CrossEntropyLoss()
        self.truncation = config['truncation']
        self.tokenizer = AutoTokenizer.from_pretrained(config['model']['bert_base_model'])#, do_lower_case=config['model_bert']['do_lower_case'])
        self.cl1_cls_model = nn.DataParallel(ModelCLCLS(**self.config["model"])).to(self.device)        
        self.cl1_cls_optimizer = torch.optim.Adam(self.cl1_cls_model.parameters(), 
                                        eval(self.config['learning_rate']), 
                                        weight_decay=eval(self.config['weight_decay']))
        

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def train(self, train_loader):
        self.cl1_cls_model.train()
        
        epoch_total_loss = 0
        
        correct = 0
        lighter_correct = 0
        darker_correct = 0
        total = 0
        lighter_total = 0
        darker_total = 0
        
        all_labels = []
        all_predictions = []
        
        for images, phrases1, labels, fitz_scales in tqdm(train_loader):
            self.cl1_cls_optimizer.zero_grad()

            xls1 = self.tokenizer(list(phrases1), 
                                return_tensors="pt", 
                                padding=True, 
                                truncation=self.truncation)
            

            xis = images.to(self.device)
            xls1 = {key: value.to(self.device) for key, value in xls1.items()}
            labels = labels.to(self.device)
            fitz_scales = fitz_scales.to(self.device) 
        
            # get the representations and the projections
            zis1, zls1, outputs =self.cl1_cls_model(xis, xls1)  # [N,C]
            cl1_loss = self.nt_xent_criterion(zis1, zls1)
            
            ## classifier  
            cls_loss = self.cls_criterion(outputs, labels)
        
            cl1_cls_loss = cl1_loss + cls_loss
        
            cl1_cls_loss.backward()
            self.cl1_cls_optimizer.step()
            
            epoch_total_loss += cl1_cls_loss.item()
                            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            lighter_mask = (fitz_scales == 1.0) | (fitz_scales == 2.0)
            lighter_correct += ((predicted == labels) & lighter_mask).sum().item()
            lighter_total += lighter_mask.sum().item()

            darker_mask = (fitz_scales == 3.0) | (fitz_scales == 4.0)
            darker_correct += ((predicted == labels) & darker_mask).sum().item()
            darker_total += darker_mask.sum().item()     
            
        epoch_total_loss /= len(train_loader)
   
        accuracy = correct / total
        lighter_accuracy = lighter_correct / lighter_total
        darker_accuracy = darker_correct / darker_total
        f1 = f1_score(all_labels, all_predictions, average='weighted')
    
        return epoch_total_loss, accuracy, lighter_accuracy, darker_accuracy, f1
             

    def validate(self, valid_loader):        
        self.cl1_cls_model.eval()

        epoch_total_loss = 0
        
        correct = 0
        lighter_correct = 0
        darker_correct = 0
        total = 0
        lighter_total = 0
        darker_total = 0
        
        all_labels = []
        all_predictions = []
        
        
        with torch.no_grad():
            print(f'Validation step')
            for images, phrases1,labels, fitz_scales in tqdm(valid_loader):
                xls1 = self.tokenizer(list(phrases1), return_tensors="pt", padding=True, truncation=self.truncation)

                xis = images.to(self.device)
                xls1 = {key: value.to(self.device) for key, value in xls1.items()}
                labels = labels.to(self.device)
                fitz_scales = fitz_scales.to(self.device) 

                # get the representations and the projections
                zis1, zls1, outputs =self.cl1_cls_model(xis, xls1) # [N,C]
                cl1_loss = self.nt_xent_criterion(zis1, zls1)
                cls_loss = self.cls_criterion(outputs, labels)
                
                cl1_cls_loss = cl1_loss + cls_loss
        
                epoch_total_loss += cl1_cls_loss.item()
                                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

                lighter_mask = (fitz_scales == 1.0) | (fitz_scales == 2.0)
                lighter_correct += ((predicted == labels) & lighter_mask).sum().item()
                lighter_total += lighter_mask.sum().item()

                darker_mask = (fitz_scales == 3.0) | (fitz_scales == 4.0)
                darker_correct += ((predicted == labels) & darker_mask).sum().item()
                darker_total += darker_mask.sum().item()     
                
            epoch_total_loss /= len(valid_loader)
            
            accuracy = correct / total
            lighter_accuracy = lighter_correct / lighter_total
            darker_accuracy = darker_correct / darker_total
            f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        return epoch_total_loss, accuracy, lighter_accuracy, darker_accuracy, f1


    def test(self, test_loader):
        self.cl1_cls_model.eval()
        
        epoch_total_loss = 0
        
        correct = 0
        lighter_correct = 0
        darker_correct = 0
        total = 0
        lighter_total = 0
        darker_total = 0
        
        all_labels = []
        all_predictions = []
        
        
        with torch.no_grad():
            for images, phrases1,labels, fitz_scales in tqdm(test_loader):
                xls1 = self.tokenizer(list(phrases1), return_tensors="pt", padding=True, truncation=self.truncation)

                xis = images.to(self.device)
                xls1 = {key: value.to(self.device) for key, value in xls1.items()}
                labels = labels.to(self.device)
                fitz_scales = fitz_scales.to(self.device) 

                # get the representations and the projections
                zis1, zls1, outputs =self.cl1_cls_model(xis, xls1) # [N,C]
                cl1_loss = self.nt_xent_criterion(zis1, zls1)
                cls_loss = self.cls_criterion(outputs, labels)
                
                cl1_cls_loss = cl1_loss + cls_loss
                
                epoch_total_loss += cl1_cls_loss.item() 
                                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

                lighter_mask = (fitz_scales == 1.0) | (fitz_scales == 2.0)
                lighter_correct += ((predicted == labels) & lighter_mask).sum().item()
                lighter_total += lighter_mask.sum().item()

                darker_mask = (fitz_scales == 3.0) | (fitz_scales == 4.0)
                darker_correct += ((predicted == labels) & darker_mask).sum().item()
                darker_total += darker_mask.sum().item()     
                
            epoch_total_loss /= len(test_loader)

            
            accuracy = correct / total
            lighter_accuracy = lighter_correct / lighter_total
            darker_accuracy = darker_correct / darker_total
            f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        return epoch_total_loss, accuracy, lighter_accuracy, darker_accuracy, f1