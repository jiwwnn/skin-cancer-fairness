import torch
import torch.nn as nn
from models.disco_model import ModelDisCo
import torch.nn.functional as F
from loss.disco_loss import Confusion_Loss, Supervised_Contrastive_Loss
from fairness_metric import compute_fairness_metrics, calculate_fairness_metrics
from lr_scheduler import CosineAnnealingWarmUpRestarts
import numpy as np
import os
import shutil
import sys
from tqdm import tqdm
from sklearn.metrics import f1_score
import logging
import yaml

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

config = yaml.load(open("disco_config.yaml", "r"), Loader=yaml.FullLoader)
torch.manual_seed(0)

def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./disco_config.yaml', os.path.join(model_checkpoints_folder, 'disco_config.yaml'))
        
class DisCo(object):
    def __init__(self, config):
        self.config = config
        self.device = self._get_device()
        self.criterion = [
            nn.CrossEntropyLoss(), Confusion_Loss(),
            nn.CrossEntropyLoss(), Supervised_Contrastive_Loss()
        ]
        self.truncation = config['truncation']
        self.model = nn.DataParallel(ModelDisCo()).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           self.config['learning_rate'],
                                           weight_decay=eval(self.config['weight_decay']))
        self.scheduler = CosineAnnealingWarmUpRestarts(
            self.optimizer, self.config['T_0'], self.config['T_mult'], 
            self.config['eta_max'], self.config['T_up'], self.config['gamma']
        )
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

        with torch.set_grad_enabled(phase == 'train'):
            for images, labels, fitz_scales in tqdm(loader):
                images, labels, fitz_scales = images.to(self.device), labels.to(self.device).long(), fitz_scales.to(self.device)
                
                if phase == 'train':
                    self.optimizer.zero_grad()

                output = self.model(images)
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
                
                # 모델 출력의 클래스 수 확인, 20250102
                output = self.model(images)

                # 모델 출력의 클래스 수 확인
                num_classes = output[0].shape[1]

                # # Debugging: labels 범위 검증
                # print(f"Labels range: min={labels.min().item()}, max={labels.max().item()}")
                # print(f"Number of classes (num_classes): {num_classes}")

                # # Invalid labels 검증 및 수정
                # if labels.max() >= num_classes or labels.min() < 0:
                #     print(f"Invalid labels detected: max={labels.max().item()}, min={labels.min().item()}")
                #     labels = torch.clamp(labels, 0, num_classes - 1)
                
                # 이부분 오류 발생 labels가 모델 출력의 클래스 수 범위를 벗어남 -> 250112 out-domain cls 코드 제거하여 오류 수정 
                loss3 = self.criterion[3](output[3], labels)
                loss = loss0 + loss1*1 + loss2 + loss3*0.8  # 1 / 0.8

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
        
        # 20250101 수정
        # Fitzpatrick scale의 고유 값 개수를 기반으로 num_types 계산
        num_types = len(set(all_fitz_types))
        fairness_metrics = calculate_fairness_metrics(all_predictions, all_labels, all_fitz_types, num_types)
    
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
