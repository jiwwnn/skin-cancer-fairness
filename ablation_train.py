import torch
import torch.nn as nn
from models.ablation_model import ModelCL1CLS
import matplotlib.pyplot as plt
import torch.nn.functional as F
from loss.ours_loss import NTXentLoss, L2Distance, RBFKernel, MahalanobisDistance
from fairness_metric import compute_fairness_metrics, calculate_fairness_metrics
from lr_scheduler import CosineAnnealingWarmUpRestarts
import numpy as np
import os
import shutil
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics import f1_score
from sklearn.manifold import TSNE
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
        shutil.copy('./ablation_config.yaml', os.path.join(model_checkpoints_folder, 'ablation_config.yaml'))
        
class CL1CLS(object):
    def __init__(self, config):
        self.config = config
        self.device = self._get_device()
        self.nt_xent_criterion = NTXentLoss(self.device, config['batch_size'], **config['cl_loss'])
        self.cls_criterion = nn.CrossEntropyLoss()
        self.truncation = config['truncation']
        self.tokenizer = AutoTokenizer.from_pretrained(config['model']['bert_base_model'])#, do_lower_case=config['model_bert']['do_lower_case'])
        self.model = nn.DataParallel(ModelCL1CLS(**self.config["model"])).to(self.device)   
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                self.config['learning_rate'], 
                                weight_decay=eval(self.config['weight_decay']))
        self.scheduler = CosineAnnealingWarmUpRestarts(self.optimizer,
                                self.config['T_0'], self.config['T_mult'], 
                                self.config['eta_max'], self.config['T_up'], 
                                self.config['gamma'])
        self.cl1_weight, self.cls_weight = 1, 1
        self.classes = config['classes']
        
    def get_model(self):
        return self.model

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def run_epoch(self, loader, phase):
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()
    
        total_loss = 0
        cl_loss1_sum = 0
        cls_loss_sum = 0

        correct = 0
        lighter_correct = 0
        darker_correct = 0
        total = 0
        lighter_total = 0
        darker_total = 0

        all_labels = []
        all_predictions = []
        all_fitz_types = []
        lighter_labels = []
        lighter_predictions = []
        darker_labels = []
        darker_predictions = []

        with torch.set_grad_enabled(phase == 'train'):
            for images, phrases1, phrases2, labels, fitz_scales in tqdm(loader):
                if phase == 'train':
                    self.optimizer.zero_grad()

                xls1 = self.tokenizer(list(phrases1), return_tensors="pt", padding=True, truncation=self.truncation)

                xis = images.to(self.device)
                xls1 = {key: value.to(self.device) for key, value in xls1.items()}
                labels = labels.to(self.device)
                fitz_scales = fitz_scales.to(self.device)

                zis1, zls1, outputs = self.model(xis, xls1)
                
                cl_loss1 = self.nt_xent_criterion(zis1, zls1)  
                cls_loss = self.cls_criterion(outputs, labels)
                
                loss = (self.cl1_weight * cl_loss1) + (self.cls_weight * cls_loss)
                
                if phase == 'train':
                    loss.backward()
                    self.optimizer.step()
                
                total_loss += loss.item()
                cl_loss1_sum += (self.cl1_weight * cl_loss1).item()
                cls_loss_sum += (self.cls_weight * cls_loss).item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_fitz_types.extend(fitz_scales.cpu().numpy())

                lighter_mask = (fitz_scales == 1.0) | (fitz_scales == 2.0)
                lighter_correct += ((predicted == labels) & lighter_mask).sum().item()
                lighter_total += lighter_mask.sum().item()
                
                lighter_labels.extend((labels & lighter_mask).cpu().numpy())
                lighter_predictions.extend((predicted & lighter_mask).cpu().numpy())

                darker_mask = (fitz_scales == 3.0) | (fitz_scales == 4.0)
                darker_correct += ((predicted == labels) & darker_mask).sum().item()
                darker_total += darker_mask.sum().item()
                
                darker_labels.extend((labels & darker_mask).cpu().numpy())
                darker_predictions.extend((predicted & darker_mask).cpu().numpy())

        total_loss /= len(loader)
        cl_loss1_sum /= len(loader)
        cls_loss_sum /= len(loader)
        
        accuracy = correct / total
        lighter_accuracy = lighter_correct / lighter_total
        darker_accuracy = darker_correct / darker_total

        f1 = f1_score(all_labels, all_predictions, average='macro')
        lighter_f1 = f1_score(lighter_labels, lighter_predictions, average='macro')
        darker_f1 = f1_score(darker_labels, darker_predictions, average='macro')

        eopp0, eopp1, eodd = compute_fairness_metrics(lighter_labels, lighter_predictions, darker_labels, darker_predictions, self.classes)
        fairness_metrics = calculate_fairness_metrics(all_predictions, all_labels, all_fitz_types)

        return {
            'total_loss': total_loss,
            'cl_loss1': cl_loss1_sum,
            'cls_loss': cls_loss_sum,
            'accuracy': accuracy,
            'lighter_accuracy': lighter_accuracy,
            'darker_accuracy': darker_accuracy,
            'f1': f1,
            'lighter_f1': lighter_f1,
            'darker_f1': darker_f1,
            'eopp0': eopp0,
            'eopp1': eopp1,
            'eodd': eodd,
            'acc_avg': fairness_metrics['acc_avg'], 
            'PQD': fairness_metrics['PQD'],
            'DPM': fairness_metrics['DPM'],
            'EOM': fairness_metrics['EOM']
        }

    def train(self, train_loader):
        return self.run_epoch(train_loader, phase='train')

    def valid(self, valid_loader):
        return self.run_epoch(valid_loader, phase='valid')

    def test(self, test_loader):
        return self.run_epoch(test_loader, phase='test')
