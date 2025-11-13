import os
import torch
import numpy as np
import pandas as pd
import yaml
from resnet_train import ResNet50
from sklearn.model_selection import train_test_split
from dataloader.baseline_dataset_wrapper import TrainDataSetWrapper, TestDataSetWrapper
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
        
        self.original_model = original_model
        
        self.current_attribute = None

    def forward(self, x):
        if self.current_attribute is None:
            raise ValueError("current_attribute가 설정되지 않았습니다.")
        return self.original_model(x)

def print_metrics(stage, metrics):
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            value_str = ", ".join(f"{v:.4f}" for v in value)  
            print(f'{stage} {key.replace("_", " ").capitalize()}: [{value_str}]', end=' ')
        else:
            print(f'{stage} {key.replace("_", " ").capitalize()}: {value:.4f}', end=' ')

def main():
    config = yaml.load(open("/dshome/ddualab/jiwon/skin_cancer_fairness/resnet_config.yaml", "r"), Loader=yaml.FullLoader)
    
    train_dataset = pd.read_csv(config['train_csv']) 
    test_dataset = pd.read_csv(config['test_csv']) 
    
    train_data, valid_data = train_test_split(train_dataset, test_size=0.25, random_state=42)  
    
    train_data = train_data.reset_index(drop=True)
    valid_data = valid_data.reset_index(drop=True)
    
    train_wrapper = TrainDataSetWrapper(train_data, config['batch_size'], config['pad_img_root_dir'], **config['dataset'])
    valid_wrapper = TestDataSetWrapper(valid_data, config['batch_size'], config['pad_img_root_dir'], **config['dataset'])
    test_wrapper = TestDataSetWrapper(test_dataset, config['batch_size'], config['pad_img_root_dir'], **config['dataset'])
    # print(len(train_data), len(valid_data), len(test_dataset))  

    # train_wrapper = TrainDataSetWrapper(train_dataset, config['batch_size'], **config['dataset'])
    # test_wrapper = TestDataSetWrapper(test_dataset, config['batch_size'], **config['dataset'])
    
    train_loader = train_wrapper.get_data_loaders()
    valid_loader = valid_wrapper.get_data_loaders()
    test_loader = test_wrapper.get_data_loaders() 
    
    # train_loader, valid_loader = train_wrapper.get_data_loaders()
    # test_loader = test_wrapper.get_data_loaders()
    
    resnet = ResNet50(config)
    num_epochs = config['epochs']
    
    max_patience = 5
    patience = 0
    best_valid_accuracy = 0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}')
        
        train_metrics = resnet.train(train_loader)
        print_metrics("Train", train_metrics)
        print('\n')

        valid_metrics = resnet.valid(valid_loader)
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
            torch.save(resnet.get_model().state_dict(), os.path.join('./runs', f'resnet_model_{current_time}.pth'))
            break

    test_metrics = resnet.test(test_loader)
    print_metrics("Test", test_metrics)
    
    # raw_model = resnet.get_model()  
    # if isinstance(raw_model, torch.nn.DataParallel):
    #     actual_model = raw_model.module
    # else:
    #     actual_model = raw_model
    
    # # Grad-CAM++ 적용
    # wrapped_model = ModelWrapper(actual_model)
    # target_layers = [wrapped_model.original_model.feature_extractor[7][-1]]  # 모델 구조에 맞게 수정 필요
    # grad_cam = GradCAMPlusPlus(model=wrapped_model, target_layers=target_layers)
    
    # model_name = "resnet"
    # output_dir = f"/dshome/ddualab/jiwon/skin_cancer_fairness/results/grad_cam_results/{model_name}"
    # os.makedirs(output_dir, exist_ok=True)
    
    # try:
    #     for idx, (inputs, labels, fitz_scales) in enumerate(test_loader):
    #         if idx == 17:  
    #             break
    #         inputs = inputs.to(resnet.device)
            
    #         fitz_scales_one_hot = F.one_hot(fitz_scales.long() - 1, num_classes=config['fitz_classes']).float()
    #         fitz_scales_one_hot = fitz_scales_one_hot.to(resnet.device)
    #         wrapped_model.current_attribute = fitz_scales_one_hot
            
    #         grayscale_cam = grad_cam(inputs, targets=None)
    #         for i in range(inputs.size(0)):
    #             input_image = inputs[i].permute(1, 2, 0).detach().cpu().numpy()
    #             input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())
    #             visualization = show_cam_on_image(input_image, grayscale_cam[i, :], use_rgb=True)
                
    #             label = labels[i].item()
    #             fitz_value = fitz_scales[i].item()  
                
    #             save_path = os.path.join(output_dir, f"grad_cam_batch{idx}_img{i}_label{label}_fitz{fitz_value}.png")
    #             plt.imsave(save_path, visualization)
            
    # finally:
    #     grad_cam.activations_and_grads.release()
    


if __name__ == "__main__":
    main()