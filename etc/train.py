import torch

# from models.resnet_clr import ResNetSimCLR
from models.model import ModelCLR
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from loss.nt_xent import NTXentLoss
from loss.l2_distance import L2DistanceLoss
import numpy as np
import os
import shutil
import sys
from tqdm import tqdm
from transformers import AdamW
from transformers import AutoTokenizer
import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys, os

apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False

torch.manual_seed(0)

def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))
        
def save_resnet_weights(model, path):
    state_dict = model.state_dict()
    resnet_keys = {k:v for k, v in state_dict.items() if 'res_features1' in k or 'res_l1_1' in k or 'res_l2_1' in k}
    torch.save(resnet_keys, path)

class SimCLR(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()
        self.writer = SummaryWriter()
        self.dataset = dataset
        self.nt_xent_criterion = NTXentLoss(self.device, config['batch_size'], **config['cl_loss'])
        self.l2_criterion = L2DistanceLoss()
        self.truncation = config['truncation']
        self.tokenizer = AutoTokenizer.from_pretrained(config['model']['bert_base_model'])#, do_lower_case=config['model_bert']['do_lower_case'])

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def train(self):
        #Dataloaders
        train_loader, valid_loader = self.dataset.get_data_loaders()

        #Model Resnet Initialize
        model = ModelCLR(**self.config["model"]).to(self.device)
        model = self._load_pre_trained_weights(model)

        optimizer = torch.optim.Adam(model.parameters(), 
                                        eval(self.config['learning_rate']), 
                                        weight_decay=eval(self.config['weight_decay']))

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                                T_max=len(train_loader), 
                                                                eta_min=0, 
                                                                last_epoch=-1)

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)


        if apex_support and self.config['fp16_precision']:
            model, optimizer = amp.initialize(model, optimizer,
                                              opt_level='O2',
                                              keep_batchnorm_fp32=True)

        #Checkpoint folder
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        print(f'Training...')

        
        for epoch_counter in range(self.config['epochs']):
            print(f'Epoch {epoch_counter}')
            epoch_train_loss = 0
            epoch_cl_loss = 0
            epoch_l2_loss = 0
            model.train()
            for xis, xls1, xls2 in tqdm(train_loader):

                optimizer.zero_grad()
                # optimizer_bert.zero_grad()

                xls1 = self.tokenizer(list(xls1), 
                                    return_tensors="pt", 
                                    padding=True, 
                                    truncation=self.truncation)
                
                xls2 = self.tokenizer(list(xls2), 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=self.truncation)
                

                xis = xis.to(self.device)
                xls1 = xls1.to(self.device)
                xls2 = xls2.to(self.device)
                
                # get the representations and the projections
                zis1, zls1, zis2, zls2 = model(xis, xls1, xls2)  # [N,C]

                # get the representations and the projections
                # zls = model_bert(xls)  # [N,C]
                # zls = xls
                # normalize projection feature vectors
                # print(zis1.size(), zis2.size())
                cl_loss = self.nt_xent_criterion(zis1, zls1, zis2, zls2)
                # cl_loss = self._step(model_res, model_bert, xis, xls, n_iter)
                # l2_criterion = torch.norm(zis1 - zis2, p=2, dim=1)
                # print(l2_criterion)
                # l2_loss = l2_criterion.mean()
                # print(l2_loss)
                l2_loss = self.l2_criterion(zis1, zis2)
                loss = cl_loss - (l2_loss * 0.00001)
                
                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)

                if apex_support and self.config['fp16_precision']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                epoch_train_loss += loss.item()
                epoch_cl_loss += cl_loss.item()
                epoch_l2_loss += l2_loss.item()
                # optimizer_bert.step()
                n_iter += 1
                
            epoch_train_loss /= len(train_loader)
            epoch_cl_loss /= len(train_loader)
            epoch_l2_loss /= len(train_loader)
            print(f'Average train loss:{epoch_train_loss:.6f} cl_loss:{epoch_cl_loss:.6f} l2_loss:{epoch_l2_loss:.6f}')
                
            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(model, valid_loader, n_iter)
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    # torch.save(model.image_encoder1, os.path.join(model_checkpoints_folder, 'image_encoder1.pth'))
                    save_resnet_weights(model, os.path.join(model_checkpoints_folder, 'image_encoder1.pth'))
                    print(f"Epoch {epoch_counter}: Best model saved with validation loss: {best_valid_loss:.6f}")
                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler.step()
            self.writer.add_scalar('cosine_lr_decay', scheduler.get_lr()[0], global_step=n_iter)

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./runs', self.config['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader, n_iter):

        # validation steps
        with torch.no_grad():
            model.eval()
            # model_bert.eval()
            valid_loss = 0.0
            counter = 0
            print(f'Validation step')
            for xis, xls1, xls2 in tqdm(valid_loader):

                xls1 = self.tokenizer(list(xls1), return_tensors="pt", padding=True, truncation=self.truncation)
                xls2 = self.tokenizer(list(xls2), return_tensors="pt", padding=True, truncation=self.truncation)

                xis = xis.to(self.device)
                xls1 = xls1.to(self.device)
                xls2 = xls2.to(self.device)

                # get the representations and the projections
                zis1, zls1, zis2, zls2 = model(xis, xls1, xls2)  # [N,C]
                cl_loss = self.nt_xent_criterion(zis1, zls1, zis2, zls2)
                l2_loss = self.l2_criterion(zis1, zis2)
                loss = cl_loss - (l2_loss * 0.00001)
                
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
            print(f'Average validation loss:{valid_loss:.6f}')
        # model.train()
        # model_bert.train()
        return valid_loss
