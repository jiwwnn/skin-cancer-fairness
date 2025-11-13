import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import balanced_accuracy_score, f1_score
from fairness_metric import compute_fairness_metrics, calculate_fairness_metrics
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from skimage import io
import shutil
from torch.utils.data import DataLoader, WeightedRandomSampler
from lr_scheduler import CosineAnnealingWarmUpRestarts
from tqdm import tqdm
from models.models_losses import Network, Supervised_Contrastive_Loss
from models.resm_model import ModelRESM
import yaml

config = yaml.load(open("resm_config.yaml", "r"), Loader=yaml.FullLoader)
torch.manual_seed(0)

def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./resm_config.yaml', os.path.join(model_checkpoints_folder, 'resm_config.yaml'))


class RESM:
    def __init__(self, config):
        self.config = config
        self.device = self._get_device()
        self.criterion = nn.CrossEntropyLoss() 
        self.model = nn.DataParallel(ModelRESM().to(self.device))
        self.optimizer = optim.Adam(self.model.parameters(), 
                                     lr=self.config['learning_rate'], 
                                     weight_decay=eval(self.config['weight_decay']))
        self.scheduler = CosineAnnealingWarmUpRestarts(self.optimizer,
                                self.config['T_0'], self.config['T_mult'], 
                                self.config['eta_max'], self.config['T_up'], 
                                self.config['gamma'])
        # self.scheduler = lr_scheduler.StepLR(self.optimizer, 
        #                                 step_size=self.config['step_size'], 
        #                                 gamma=self.config['gamma'])
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
            for images, labels, fitz_scales in tqdm(loader):
                if phase == 'train':
                    self.optimizer.zero_grad()
                    
                images = images.to(self.device)
                labels = labels.to(self.device)
                fitz_scales = fitz_scales.to(self.device)
                
                outputs = self.model.to(self.device)(images)
                loss = self.criterion(outputs, labels)
                
                if phase == 'train':
                    loss.backward()
                    self.optimizer.step()
                
                total_loss += loss.item()
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
            
        avg_loss = total_loss / len(loader)
        accuracy = correct / total
        
        lighter_accuracy = lighter_correct / lighter_total
        darker_accuracy = darker_correct / darker_total
        
        f1 = f1_score(all_labels, all_predictions, average='macro')
        lighter_f1 = f1_score(lighter_labels, lighter_predictions, average='macro')
        darker_f1 = f1_score(darker_labels, darker_predictions, average='macro')   
        eopp0, eopp1, eodd = compute_fairness_metrics(lighter_labels, lighter_predictions, darker_labels, darker_predictions, self.classes) 
        fairness_metrics = calculate_fairness_metrics(all_predictions, all_labels, all_fitz_types, config['fitz_classes'])
        
        return {
            'avg_loss': avg_loss,
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
            'acc_per_type': fairness_metrics['acc_per_type'],
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