import torch
import torch.nn as nn
from models.ours_model import ModelOurs
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
import yaml
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

config = yaml.load(open("ours_config.yaml", "r"), Loader=yaml.FullLoader)
torch.manual_seed(0)

def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./ours_config.yaml', os.path.join(model_checkpoints_folder, 'ours_config.yaml'))
        
class Ours(object):
    def __init__(self, config):
        self.config = config
        self.device = self._get_device()
        self.nt_xent_criterion = NTXentLoss(self.device, config['batch_size'], **config['cl_loss'])
        self.cls_criterion = nn.CrossEntropyLoss()
        self.dist_criterion = nn.CosineSimilarity() #RBFKernel() #L2Distance() #MahalanobisDistance()  
        self.match_criterion = nn.MSELoss()  
        self.truncation = config['truncation']
        self.tokenizer = AutoTokenizer.from_pretrained(config['model']['bert_base_model'])#, do_lower_case=config['model_bert']['do_lower_case'])
        self.model = nn.DataParallel(ModelOurs(**self.config["model"])).to('cuda') # to(self.device)   
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                self.config['learning_rate'], 
                                weight_decay=eval(self.config['weight_decay']))
        self.scheduler = CosineAnnealingWarmUpRestarts(self.optimizer,
                                self.config['T_0'], self.config['T_mult'], 
                                self.config['eta_max'], self.config['T_up'], 
                                self.config['gamma'])
        self.cl1_weight, self.cl2_weight, self.match_weight, self.cls_weight = 1, 1, 0.01, 1
    
        # self.cl1_cls_model = nn.DataParallel(ModelCLCLS(**self.config["model"])).to(self.device)        
        # self.cl1_cls_optimizer = torch.optim.AdamW(self.cl1_cls_model.parameters(), 
        #                                 self.config['learning_rate']
        #                                 weight_decay=eval(self.config['weight_decay']))
        # self.cl1_cls_scheduler = CosineAnnealingWarmUpRestarts(self.cl1_cls_optimizer,
        #                                 self.config['T_0'], self.config['T_mult'], 
        #                                 self.config['eta_max'], self.config['T_up'], 
        #                                 self.config['gamma'])
        # self.cl2_model = nn.DataParallel(ModelCL(**self.config["model"])).to(self.device)                                                    
        # self.cl2_optimizer = torch.optim.AdamW(self.cl2_model.parameters(), 
        #                                 self.config['learning_rate'], 
        #                                 weight_decay=eval(self.config['weight_decay']))
        # self.cl2_scheduler = CosineAnnealingWarmUpRestarts(self.cl2_optimizer,
        #                                 self.config['T_0'], self.config['T_mult'], 
        #                                 self.config['eta_max'], self.config['T_up'], 
        #                                 self.config['gamma'])
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
        cl_loss2_sum = 0
        match_loss_sum = 0
        cls_loss_sum = 0
        cosine_similarities = []

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
                xls2 = self.tokenizer(list(phrases2), return_tensors="pt", padding=True, truncation=self.truncation)

                xis = images.cuda()
                xls1 = {key: value.cuda() for key, value in xls1.items()}
                xls2 = {key: value.cuda() for key, value in xls2.items()}
                labels = labels.cuda()
                fitz_scales = fitz_scales.cuda()

                zis1, zis2, zls1, zls2, outputs = self.model(xis, xls1, xls2)
                
                cl_loss1 = self.nt_xent_criterion(zis1, zls1)
                cl_loss2 = self.nt_xent_criterion(zis2, zls2)            
                # xis_l2 = self.dist_criterion(zis1, zis2).sum()
                # xls_l2 = self.dist_criterion(zls1, zls2).sum()
                # print(xis_l2, xls_l2)
                
                # match_loss = torch.sqrt(self.match_criterion(xis_l2, xls_l2))
                # match_loss = self.match_criterion(xis_l2, xls_l2)
                cls_loss = self.cls_criterion(outputs, labels)
                # cosine_similarity = F.cosine_similarity(zis1, zis2, dim=-1).mean().item()
                
                loss = (self.cl1_weight * cl_loss1) + (self.cl2_weight * cl_loss2) + (self.cls_weight * cls_loss)
                # loss = (self.cl1_weight * cl_loss1) + (self.cl2_weight * cl_loss2) + (self.match_weight * match_loss) + (self.cls_weight * cls_loss)

                if phase == 'train':
                    loss.backward()
                    self.optimizer.step()
                
                total_loss += loss.item()
                cl_loss1_sum += (self.cl1_weight * cl_loss1).item()
                cl_loss2_sum += (self.cl2_weight * cl_loss2).item()
                # match_loss_sum += (self.match_weight * match_loss).item()
                cls_loss_sum += (self.cls_weight * cls_loss).item()
                # cosine_similarities.append(cosine_similarity)
                
                _, predicted = torch.max(outputs.data, 1)
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

        total_loss /= len(loader)
        cl_loss1_sum /= len(loader)
        cl_loss2_sum /= len(loader)
        # match_loss_sum /= len(loader)
        cls_loss_sum /= len(loader)
        # avg_cosine_similarity = np.mean(cosine_similarities)

        accuracy = correct / total
        # lighter_accuracy = lighter_correct / lighter_total
        # darker_accuracy = darker_correct / darker_total

        # f1 = f1_score(all_labels, all_predictions, average='macro')
        # lighter_f1 = f1_score(lighter_labels, lighter_predictions, average='macro')
        # darker_f1 = f1_score(darker_labels, darker_predictions, average='macro')

        # eopp0, eopp1, eodd = compute_fairness_metrics(lighter_labels, lighter_predictions, darker_labels, darker_predictions, self.classes)
        fairness_metrics = calculate_fairness_metrics(all_predictions, all_labels, all_fitz_types, config['fitz_classes'])

        return {
            'total_loss': total_loss,
            'cl_loss1': cl_loss1_sum,
            'cl_loss2': cl_loss2_sum,
            # 'mse_loss': match_loss_sum,
            'cls_loss': cls_loss_sum,
            # 'avg_cosine_similarity': avg_cosine_similarity,
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
