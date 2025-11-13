import os
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
from ours_train import Ours
from uuid import uuid4  
import yaml
from sklearn.model_selection import train_test_split
from dataloader.ours_dataset_wrapper import TrainDataSetWrapper, TestDataSetWrapper
import torch.nn.functional as F
from transformers import AutoTokenizer
from pytorch_grad_cam.grad_cam_plusplus import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
from datetime import datetime
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import sys
sys.path.append('/dshome/ddualab/jiwon/skin_cancer_fairness/torch_grad_cam')
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

torch.manual_seed(0)

# 원본 모델을 감싸는 wrapper class
class ModelWrapper(torch.nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.original_model = original_model
        self.current_xls1 = None
        self.current_xls2 = None

    def set_current_xls(self, xls1, xls2):
        self.current_xls1 = xls1
        self.current_xls2 = xls2

    def forward(self, x):
        if self.current_xls1 is None or self.current_xls2 is None:
            raise ValueError("current_xls1 또는 current_xls2가 설정되지 않았습니다. "
                             "set_current_xls(xls1, xls2) 메서드를 사용해 값을 설정하세요.")
        # out1, out2 = self.original_model(x, self.current_xls1, self.current_xls2)
        # return out1, out2
        # 모델 return값에 맞춰 수정, 250116
        zis1, zis2, zls1, zls2, outputs = self.original_model(x, self.current_xls1, self.current_xls2)
        return outputs 
    
def print_metrics(stage, metrics):
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            value_str = ", ".join(f"{v:.4f}" for v in value)  
            print(f'{stage} {key.replace("_", " ").capitalize()}: [{value_str}]', end=' ')
        else:
            print(f'{stage} {key.replace("_", " ").capitalize()}: {value:.4f}', end=' ')

def main():
    config = yaml.load(open("/dshome/ddualab/jiwon/skin_cancer_fairness/ours_config.yaml", "r"), Loader=yaml.FullLoader)
    
    train_dataset = pd.read_csv(config['train_csv']) 
    test_dataset = pd.read_csv(config['test_csv']) 

    train_data, valid_data = train_test_split(train_dataset, test_size=0.25, random_state=42)    
    train_data = train_data.reset_index(drop=True)
    valid_data = valid_data.reset_index(drop=True)
    # print(len(train_data), len(valid_data), len(test_dataset))
        
    # train_wrapper = TrainDataSetWrapper(train_dataset, config['batch_size'], **config['dataset'])
    # test_wrapper = TestDataSetWrapper(test_dataset, config['batch_size'], **config['dataset'])
    
    train_wrapper = TrainDataSetWrapper(train_data, config['batch_size'], config['pad_img_root_dir'], **config['dataset'])
    valid_wrapper = TestDataSetWrapper(valid_data, config['batch_size'], config['pad_img_root_dir'], **config['dataset'])
    test_wrapper = TestDataSetWrapper(test_dataset, config['batch_size'], config['pad_img_root_dir'], **config['dataset'])

    # train_loader, valid_loader = train_wrapper.get_data_loaders()
    # test_loader = test_wrapper.get_data_loaders()
    
    train_loader = train_wrapper.get_data_loaders()
    valid_loader = valid_wrapper.get_data_loaders()
    test_loader = test_wrapper.get_data_loaders() 

    ours = Ours(config)
    num_epochs = config['epochs']
    
    max_patience = 10
    patience = 0
    best_valid_accuracy = 0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}')
        
        train_metrics = ours.train(train_loader)
        print_metrics("Train", train_metrics)
        print('\n')

        valid_metrics = ours.valid(valid_loader)
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
            torch.save(ours.get_model().state_dict(), os.path.join('./runs', f'out_ours_model_{current_time}.pth'))
            break

    test_metrics = ours.test(test_loader)
    print_metrics("Test", test_metrics)
    
    # raw_model = ours.get_model()  
    # if isinstance(raw_model, torch.nn.DataParallel):
    #     actual_model = raw_model.module
    # else:
    #     actual_model = raw_model
    
    # # Grad-CAM++ 설정
    # wrapped_model = ModelWrapper(actual_model)
    # # print(wrapped_model.original_model)
    # # res_features[-1]이 AdaptiveAvgPool2d이라서 마지막 conv layer 숫자 기입, 250119
    # target_layers = [wrapped_model.original_model.res_features1[6][-1].conv3]
    # # target_layers_1 = [wrapped_model.original_model.res_features1[-1][-1].conv2]  # ResNet-50(병변)의 layer4 마지막 conv2
    # # target_layers_2 = [wrapped_model.original_model.res_features2[-1][-1].conv2]  # ResNet-50(피부)의 layer4 마지막 conv2

    # grad_cam = GradCAMPlusPlus(model=wrapped_model, target_layers=target_layers)
    # # grad_cam_1 = GradCAMPlusPlus(model=wrapped_model, target_layers=target_layers_1)
    # # grad_cam_2 = GradCAMPlusPlus(model=wrapped_model, target_layers=target_layers_2)


    # # # res_features2의 구조 확인
    # # print("=== res_features2 구조 ===")
    # # for idx, layer in enumerate(wrapped_model.original_model.res_features2):
    # #     print(f"Index: {idx}, Layer: {layer}")
        
    # # target_layers = target_layers = [wrapped_model.original_model.res_features2[7]]
    # # grad_cam = GradCAMPlusPlus(model=wrapped_model, target_layers=target_layers)
    
    # model_name = "ours"
    # output_dir = f"/dshome/ddualab/jiwon/skin_cancer_fairness/results/grad_cam_results/{model_name}"
    # os.makedirs(output_dir, exist_ok=True)
    
    # try:
    #     for idx, (images, phrases1, phrases2, labels, fitz_scales) in enumerate(test_loader):
    #         if idx == 4:  
    #             break
    #         images = images.to(ours.device)

    #         xls1 = ours.tokenizer(list(phrases1), return_tensors="pt", padding=True, truncation=True)
    #         xls1 = {k: v.cuda() for k, v in xls1.items()}  

    #         xls2 = ours.tokenizer(list(phrases2), return_tensors="pt", padding=True, truncation=True)
    #         xls2 = {k: v.cuda() for k, v in xls2.items()}
            
    #         wrapped_model.set_current_xls(xls1, xls2)

    #         # Grad-CAM++ 활성화 맵 생성
    #         grayscale_cam = grad_cam(images, targets=None)

    #         for i in range(images.size(0)):  
    #             input_image = images[i].permute(1, 2, 0).detach().cpu().numpy()
    #             input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())
    #             visualization = show_cam_on_image(input_image, grayscale_cam[i, :], use_rgb=True)
                
    #             fitz_value = fitz_scales[i].item()  
    #             save_path = os.path.join(output_dir, f"grad_cam_batch{idx}_img{i}_fitz{fitz_value}.png")
    #             plt.imsave(save_path, visualization)

    # finally:
    #     grad_cam.activations_and_grads.release()

        # if grad_cam_1 and grad_cam_1.activations_and_grads:
        #     grad_cam_1.activations_and_grads.release()
        # if grad_cam_2 and grad_cam_2.activations_and_grads:
        #     grad_cam_2.activations_and_grads.release()

if __name__ == "__main__":
    main()
            
    #         # 텍스트 토큰화
    #         xls1 = ours.tokenizer(list(phrases1), return_tensors="pt", padding=True, truncation=True)
    #         xls1 = {k: v.cuda() for k, v in xls1.items()}
    #         xls2 = ours.tokenizer(list(phrases2), return_tensors="pt", padding=True, truncation=True)
    #         xls2 = {k: v.cuda() for k, v in xls2.items()}
            
    #         wrapped_model.current_xls1 = xls1
    #         wrapped_model.current_xls2 = xls2

    #         outputs = wrapped_model(images)  # 모델에서 결과값 추출
    #         # 튜플인 경우 첫 번째 값을 선택
    #         if isinstance(outputs, tuple):
    #             outputs = outputs[0]
    
    #         preds = torch.argmax(outputs, dim=1).cpu().numpy()
    #         targets = [ClassifierOutputTarget(int(p)) for p in preds]
            
    #         grayscale_cam = grad_cam(images, targets=None)
    #         # fitz_scales_one_hot = F.one_hot(fitz_scales.long() - 1, num_classes=config['fitz_classes']).float()
    #         # fitz_scales_one_hot = fitz_scales_one_hot.to(ours.device)
    #         # wrapped_model.current_attribute = fitz_scales_one_hot
            
    #         for i in range(images.size(0)):
    #             input_image = images[i].permute(1, 2, 0).detach().cpu().numpy()
    #             input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())
    #             visualization = show_cam_on_image(input_image, grayscale_cam[i, :], use_rgb=True)
    #             save_path = os.path.join(output_dir, f"grad_cam_{idx}_{i}.png")
    #             plt.imsave(save_path, visualization)
    #         break
    # finally:
    #     grad_cam.activations_and_grads.release()
    