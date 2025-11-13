import numpy as np
import pandas as pd
import torch
import argparse
import yaml
import os
from dataloader.baseline_dataset_wrapper import TestDataSetWrapper,TrainDataSetWrapper
from sklearn.model_selection import train_test_split
from atrb_train import *
import torch.nn.functional as F
from datetime import datetime

from pytorch_grad_cam.grad_cam_plusplus import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt

import sys
sys.path.append('/dshome/ddualab/jiwon/skin_cancer_fairness/torch_grad_cam')

torch.manual_seed(0)

class ModelWrapper(torch.nn.Module):
    def __init__(self, original_model):
        super().__init__()
        # 원본 ModelATRB
        self.original_model = original_model
        # 배치마다 다르게 설정할 attribute
        self.current_attribute = None

    def forward(self, x):
        # Grad-CAM은 model(x)만 부르므로, 여기서 attribute까지 전달
        if self.current_attribute is None:
            raise ValueError("current_attribute가 설정되지 않았습니다.")
        return self.original_model(x, self.current_attribute)

    
def print_metrics(stage, metrics):
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            value_str = ", ".join(f"{v:.4f}" for v in value)  
            print(f'{stage} {key.replace("_", " ").capitalize()}: [{value_str}]', end=' ')
        else:
            print(f'{stage} {key.replace("_", " ").capitalize()}: {value:.4f}', end=' ')

def main():
    config = yaml.load(open("/dshome/ddualab/jiwon/skin_cancer_fairness/atrb_config.yaml", "r"), Loader=yaml.FullLoader)
    
    train_dataset = pd.read_csv(config['train_csv']) 
    test_dataset = pd.read_csv(config['test_csv']) 
    
    train_data, valid_data = train_test_split(train_dataset, test_size=0.25, random_state=42)     
    train_data = train_data.reset_index(drop=True)
    valid_data = valid_data.reset_index(drop=True)
    # print(len(train_data), len(valid_data), len(test_dataset))   

    train_wrapper = TrainDataSetWrapper(train_data, config['batch_size'], config['pad_img_root_dir'], **config['dataset'])
    valid_wrapper = TestDataSetWrapper(valid_data, config['batch_size'], config['pad_img_root_dir'], **config['dataset'])
    test_wrapper = TestDataSetWrapper(test_dataset, config['batch_size'], config['fitz17_img_root_dir'], **config['dataset'])

    train_loader = train_wrapper.get_data_loaders()
    valid_loader = valid_wrapper.get_data_loaders()
    test_loader = test_wrapper.get_data_loaders() 

    # train_wrapper = TrainDataSetWrapper(train_dataset, config['batch_size'], **config['dataset'])
    # test_wrapper = TestDataSetWrapper(test_dataset, config['batch_size'], **config['dataset'])
    
    # train_loader, valid_loader = train_wrapper.get_data_loaders()
    # test_loader = test_wrapper.get_data_loaders()
    
    atrb = ATRB(config)
    num_epochs = config['epochs']
    
    max_patience = 5 
    patience = 0
    best_valid_accuracy = 0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}')
        
        train_metrics = atrb.train(train_loader)
        print_metrics("Train", train_metrics)
        print('\n')

        valid_metrics = atrb.valid(valid_loader)
        print_metrics("Valid", valid_metrics)
        print('\n')

        if valid_metrics['accuracy'] > best_valid_accuracy:
            best_valid_accuracy = valid_metrics['accuracy']
            patience = 0
        else:
            patience += 1
            print(f'No improvement in validation accuracy for {patience} epochs.')

        if patience >= max_patience:
            print('Early Stopping triggered.')
            current_time = datetime.now().strftime('%m-%d_%H-%M')
            torch.save(atrb.get_model().state_dict(), os.path.join('./runs', f'atrb_model_{current_time}.pth'))
            break

    test_metrics = atrb.test(test_loader)
    print_metrics("Test", test_metrics)
    # print('\n')

    # # gradpluplus 적용 
    # raw_model = atrb.get_model()
    # actual_model = raw_model.module
    # wrapped_model = ModelWrapper(actual_model)
    # # print(wrapped_model.original_model)
    # target_layers = [wrapped_model.original_model.feature_extractor.layer4[-1]]
    # grad_cam = GradCAMPlusPlus(model=wrapped_model, target_layers=target_layers)
    
    # model_name = "atrb"  # baseline 모델들
    # output_dir = f"/dshome/ddualab/jiwon/skin_cancer_fairness/results/grad_cam_results/{model_name}"
    # os.makedirs(output_dir, exist_ok=True)
    
    # try:
    #     for idx, (inputs, labels, fitz_scales) in enumerate(test_loader):
    #         inputs = inputs.to(atrb.device)
    #         fitz_scales_one_hot = F.one_hot(fitz_scales.long() - 1, num_classes=config['fitz_classes']).float()
    #         fitz_scales_one_hot = fitz_scales_one_hot.to(atrb.device)
    #         wrapped_model.current_attribute = fitz_scales_one_hot
    #         grayscale_cam = grad_cam(inputs, targets=None)
    #         for i in range(inputs.size(0)):
    #             input_image = inputs[i].permute(1, 2, 0).detach().cpu().numpy()
    #             input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())
    #             visualization = show_cam_on_image(input_image, grayscale_cam[i, :], use_rgb=True)
    #             save_path = os.path.join(output_dir, f"grad_cam_{idx}_{i}.png")
    #             plt.imsave(save_path, visualization)
    #         break
    # finally:
    #     grad_cam.activations_and_grads.release()

if __name__ == "__main__":
    main()









# 원본 코드 

# def print_metrics(stage, metrics):
#     for key, value in metrics.items():
#         if isinstance(value, np.ndarray):
#             value_str = ", ".join(f"{v:.4f}" for v in value)  
#             print(f'{stage} {key.replace("_", " ").capitalize()}: [{value_str}]', end=' ')
#         else:
#             print(f'{stage} {key.replace("_", " ").capitalize()}: {value:.4f}', end=' ')

# def main():
#     config = yaml.load(open("atrb_config.yaml", "r"), Loader=yaml.FullLoader)
    
#     train_dataset = pd.read_csv(config['train_csv']) 
#     test_dataset = pd.read_csv(config['test_csv']) 
    
#     # train_data, valid_data = train_test_split(train_dataset, test_size=0.2, random_state=42, stratify=train_dataset[['fitzpatrick_scale', 'label']])    
#     # train_data = train_data.reset_index(drop=True)
#     # valid_data = valid_data.reset_index(drop=True)
    
#     train_wrapper = TrainDataSetWrapper(train_dataset, config['batch_size'], **config['dataset'])
#     test_wrapper = TestDataSetWrapper(test_dataset, config['batch_size'], **config['dataset'])
    
#     train_loader, valid_loader = train_wrapper.get_data_loaders()
#     test_loader = test_wrapper.get_data_loaders()
    
#     atrb = ATRB(config)
#     num_epochs = config['epochs']
    
#     max_patience = 5 
#     patience = 0
#     best_valid_accuracy = 0
    
#     for epoch in range(num_epochs):
#         print(f'Epoch {epoch}')
        
#         train_metrics = atrb.train(train_loader)
#         print_metrics("Train", train_metrics)
#         print('\n')

#         valid_metrics = atrb.valid(valid_loader)
#         print_metrics("Valid", valid_metrics)
#         print('\n')

#         if valid_metrics['accuracy'] > best_valid_accuracy:
#             best_valid_accuracy = valid_metrics['accuracy']
#             patience = 0
#         else:
#             patience += 1
#             print(f'No improvement in validation accuracy for {patience} epochs.')

#         if patience >= max_patience:
#             print('Early Stopping triggered.')
#             torch.save(atrb.get_model().state_dict(), os.path.join('./runs', 'atrb_model.pth'))
#             break

#     test_metrics = atrb.test(test_loader)
#     print_metrics("Test", test_metrics)

# if __name__ == "__main__":
#     main()  