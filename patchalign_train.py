import numpy as np
import os
import torch
import torch.nn as nn
from models.patchalign_model import ModelPatchAlign
from transformers import AutoTokenizer
import torch.nn.functional as F
from loss.patchalign_loss import GOT_Loss, Masked_GOT_NewSinkhorn, Confusion_Loss
from fairness_metric import compute_fairness_metrics, calculate_fairness_metrics
from lr_scheduler import CosineAnnealingWarmUpRestarts
from tqdm import tqdm
import shutil
import sys
from sklearn.metrics import f1_score
import logging
import yaml
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

config = yaml.load(open("patchalign_config.yaml", "r"), Loader=yaml.FullLoader)
torch.manual_seed(40)
# torch.backends.cudnn.deterministic = False
# torch.backends.cudnn.benchmark = True

concatenated_label = ['actinic keratosis benign', 'nevus benign', 'seborrheic keratosis benign', 'basal cell carcinoma malignant', 'melanoma malignant', 'squamous cell carcinoma malignant', 'eudermic eudermic']

def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./patchalign_config.yaml', os.path.join(model_checkpoints_folder, 'patchalign_config.yaml'))
        
class PatchAlign(object):
    def __init__(self, config):
        self.config = config
        self.device = self._get_device()
        self.criterion = [nn.CrossEntropyLoss(), Confusion_Loss(), nn.CrossEntropyLoss()]
        self.got_loss = GOT_Loss()
        self.truncation = config['truncation']
        self.tokenizer = AutoTokenizer.from_pretrained(config['model']['bert_base_model'])
        # self.model = ModelPatchAlign(**self.config["model"]).to(self.device) 
        self.model = nn.DataParallel(ModelPatchAlign(**self.config["model"])).to(self.device)   
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                        self.config['learning_rate'], 
                                        weight_decay=eval(self.config['weight_decay']))    
        self.scheduler = CosineAnnealingWarmUpRestarts(self.optimizer,
                                self.config['T_0'], self.config['T_mult'], 
                                self.config['eta_max'], self.config['T_up'], 
                                self.config['gamma'])
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
            
        total_loss, correct, total = 0, 0, 0
        lighter_correct, lighter_total = 0, 0
        darker_correct, darker_total = 0, 0
        
        all_labels, all_predictions, all_fitz_types = [], [], []
        lighter_labels, lighter_predictions = [], []
        darker_labels, darker_predictions = [], []
        
        with torch.set_grad_enabled(phase=='train'):
            for images, labels, fitz_scales in tqdm(loader):
                images, labels, fitz_scales = images.to(self.device), labels.to(self.device), fitz_scales.to(self.device)
            
                if phase == 'train':
                    self.optimizer.zero_grad()
                    
                # fitz_scales 값의 범위 확인 및 강제 조정
                # print(f"Original fitz_scales: {fitz_scales}")
                # adjusted_fitz_scales = torch.clamp(fitz_scales - 3, 0, self.config['fitz_classes'] - 1).long()
                # print(f"Adjusted fitz_scales: {adjusted_fitz_scales}")    

                encoded_inputs = self.tokenizer(concatenated_label, 
                                            return_tensors="pt", 
                                            padding=True, 
                                            truncation=self.truncation)
            
                encoded_inputs = {key: value.to(self.device) for key, value in encoded_inputs.items()}
            
                output = self.model.module(images, encoded_inputs) # DataParallel 오류로 module 추가, 250226
            
                # print(output[-2].shape, output[-1].shape, output[3].shape) 
                l_got = self.got_loss(output[-2], output[-1], output[3], lamb = 0.9)
                loss0 = self.criterion[0](output[0], labels)

                # out-domain cls 시 활용
                if phase != 'test':
                    loss1 = self.criterion[1](output[1], fitz_scales-1)
                    loss2 = self.criterion[2](output[2].float(), (fitz_scales-1).long())
                else:
                    loss1 = self.criterion[1](output[1], fitz_scales-3)
                    loss2 = self.criterion[2](output[2].float(), (fitz_scales-3).long())

                # loss1 = self.criterion[1](output[1], fitz_scales-1)
                # loss2 = self.criterion[2](output[2].float(), (fitz_scales-1).long())
                loss = loss0 + loss1*0.5 + loss2 + l_got*0.7#+loss3*beta   #0.5 / 1   #(in)0.5/0.7 
                
                if phase == 'train':
                    loss.backward()
                    self.optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(output[0], 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_fitz_types.extend(fitz_scales.cpu().numpy())
                
                # lighter_mask = (fitz_scales == 1.0) | (fitz_scales == 2.0)            
                # lighter_correct += ((predicted == labels) & lighter_mask).sum().item()
                # lighter_total += lighter_mask.sum().item()
                
                # lighter_labels.extend((labels & lighter_mask).cpu().numpy())
                # lighter_predictions.extend((predicted & lighter_mask).cpu().numpy())

                # darker_mask = (fitz_scales == 3.0) | (fitz_scales == 4.0)
                # darker_correct += ((predicted == labels) & darker_mask).sum().item()
                # darker_total += darker_mask.sum().item()     
                
                # darker_labels.extend((labels & darker_mask).cpu().numpy())
                # darker_predictions.extend((predicted & darker_mask).cpu().numpy())     

            avg_loss = total_loss / len(loader)
            accuracy = correct / total
            
            # lighter_accuracy = lighter_correct / lighter_total
            # darker_accuracy = darker_correct / darker_total
            
            # f1 = f1_score(all_labels, all_predictions, average='macro')
            # lighter_f1 = f1_score(lighter_labels, lighter_predictions, average='macro')
            # darker_f1 = f1_score(darker_labels, darker_predictions, average='macro')   
            # eopp0, eopp1, eodd = compute_fairness_metrics(lighter_labels, lighter_predictions, darker_labels, darker_predictions, self.classes)
            fairness_metrics = calculate_fairness_metrics(all_predictions, all_labels, all_fitz_types, config['fitz_classes'])
            
        
            return {
                'avg_loss': avg_loss,
                'accuracy': accuracy,
                # 'lighter_accuracy': lighter_accuracy,
                # 'darker_accuracy': darker_accuracy,
                # 'f1': f1,
                # 'lighter_f1': lighter_f1,
                # 'darker_f1': darker_f1,
                # 'eopp0': eopp0,
                # 'eopp1': eopp1,
                # 'eodd': eodd,
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
