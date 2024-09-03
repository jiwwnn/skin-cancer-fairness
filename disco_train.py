import torch
import torch.nn as nn
from models.disco_model import ModelDisCo
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from loss.DisCo_loss import Confusion_Loss, Supervised_Contrastive_Loss
import numpy as np
import os
import shutil
import sys
from tqdm import tqdm
from transformers import AdamW
from sklearn.metrics import f1_score
import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys, os

torch.manual_seed(0)

def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))
        
class DisCo(object):
    def __init__(self, config):
        self.config = config
        self.device = self._get_device()
        self.writer = SummaryWriter()
        self.criterion = [nn.CrossEntropyLoss(), Confusion_Loss(), 
            nn.CrossEntropyLoss(), Supervised_Contrastive_Loss()]
        self.truncation = config['truncation']
        self.model = nn.DataParallel(ModelDisCo()).to(self.device)   
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                        eval(self.config['learning_rate']), 
                                        weight_decay=eval(self.config['weight_decay']))                                             
        
    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device
        
    def train(self, train_loader, alpha=1.0, beta=0.8):
        self.model.train()
        
        total_loss = 0
        correct = 0
        lighter_correct = 0
        darker_correct = 0
        total = 0
        lighter_total = 0
        darker_total = 0
        
        all_labels = []
        all_predictions = []
        
        for images, labels, fitz_scales in tqdm(train_loader):
            self.optimizer.zero_grad()
            images = images.to(self.device)
            labels = labels.to(self.device)
            fitz_scales = fitz_scales.to(self.device)
            
            output = self.model(images)
            loss0 = self.criterion[0](output[0], labels)
            # note!!! skin type starts from 1, so subtract 1
            loss1 = self.criterion[1](output[1], fitz_scales-1)
            loss2 = self.criterion[2](output[2].float(), (fitz_scales-1).long())
            loss3 = self.criterion[3](output[3], labels)
            loss = loss0+loss1*alpha+loss2+loss3*beta
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(output[0], 1)
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

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        lighter_accuracy = lighter_correct / lighter_total
        darker_accuracy = darker_correct / darker_total
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        return avg_loss, accuracy, lighter_accuracy, darker_accuracy, f1
            
        
    def validate(self, valid_loader, alpha=1.0, beta=0.8):
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
        
        with torch.no_grad():
            print(f'Validation step')
            for images, labels, fitz_scales in tqdm(valid_loader):
                self.optimizer.zero_grad()
                images = images.to(self.device)
                labels = labels.to(self.device)
                fitz_scales = fitz_scales.to(self.device)
                
                output = self.model(images)
                loss0 = self.criterion[0](output[0], labels)
                loss1 = self.criterion[1](output[1], fitz_scales-1)
                loss2 = self.criterion[2](output[2].float(), (fitz_scales-1).long())
                loss3 = self.criterion[3](output[3], labels)
                loss = loss0+loss1*alpha+loss2+loss3*beta
                
                total_loss += loss.item()
                _, predicted = torch.max(output[0], 1)
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

            avg_loss = total_loss / len(valid_loader)
            accuracy = correct / total
            lighter_accuracy = lighter_correct / lighter_total
            darker_accuracy = darker_correct / darker_total
            f1 = f1_score(all_labels, all_predictions, average='weighted')
        return avg_loss, accuracy, lighter_accuracy, darker_accuracy, f1
        

    def test(self, test_loader, alpha=1.0, beta=0.8):
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
        
        with torch.no_grad():
            for images, labels, fitz_scales in tqdm(test_loader):
                self.optimizer.zero_grad()
                images = images.to(self.device)
                labels = labels.to(self.device)
                fitz_scales = fitz_scales.to(self.device)
                
                output = self.model(images)
                loss0 = self.criterion[0](output[0], labels)
                loss1 = self.criterion[1](output[1], fitz_scales-1)
                loss2 = self.criterion[2](output[2].float(), (fitz_scales-1).long())
                loss3 = self.criterion[3](output[3], labels)
                loss = loss0+loss1*alpha+loss2+loss3*beta
                
                total_loss += loss.item()
                _, predicted = torch.max(output[0], 1)
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

            avg_loss = total_loss / len(test_loader)
            accuracy = correct / total
            lighter_accuracy = lighter_correct / lighter_total
            darker_accuracy = darker_correct / darker_total
            f1 = f1_score(all_labels, all_predictions, average='weighted')
        return avg_loss, accuracy, lighter_accuracy, darker_accuracy, f1