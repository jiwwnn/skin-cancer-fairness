import os
import torch
import numpy as np
import pandas as pd
from patchalign_train import PatchAlign
import yaml
from sklearn.model_selection import train_test_split
from dataloader.baseline_dataset_wrapper import TrainDataSetWrapper, TestDataSetWrapper
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime

from pytorch_grad_cam.grad_cam_plusplus import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image

import sys
sys.path.append('/dshome/ddualab/jiwon/skin_cancer_fairness/torch_grad_cam')

torch.manual_seed(39)
# torch.backends.cudnn.deterministic = False
# torch.backends.cudnn.benchmark = True

class ModelWrapper(torch.nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.original_model = original_model
        self.current_xls = None

    def forward(self, x):
        if self.current_xls is None:
            raise ValueError("current_xls가 설정되지 않았습니다.")
        output = self.original_model(x, self.current_xls)
        return output[0] #  model return 형식이 리스트여서 그 중 분류에 해당하는 첫번째 요소만 받아옴 250113


def print_metrics(stage, metrics):
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            value_str = ", ".join(f"{v:.4f}" for v in value)  
            print(f'{stage} {key.replace("_", " ").capitalize()}: [{value_str}]', end=' ')
        else:
            print(f'{stage} {key.replace("_", " ").capitalize()}: {value:.4f}', end=' ')

# def adjust_label_range(dataset, label_column, num_classes):
    
#     min_value = dataset[label_column].min()
#     max_value = dataset[label_column].max()
#     print(f"Original {label_column} range: {min_value} to {max_value}")

#     # Adjust values to [0, num_classes-1]
#     dataset[label_column] = dataset[label_column] - min_value
#     dataset[label_column] = dataset[label_column].clip(0, num_classes - 1)

#     print(f"Adjusted {label_column} range: {dataset[label_column].min()} to {dataset[label_column].max()}")
#     print(f"Adjusted {label_column} unique values: {dataset[label_column].unique()}")
#     return dataset

def main():
    config = yaml.load(open("/dshome/ddualab/jiwon/skin_cancer_fairness/patchalign_config.yaml", "r"), Loader=yaml.FullLoader)

    train_dataset = pd.read_csv(config['train_csv']) 
    test_dataset = pd.read_csv(config['test_csv']) 
    
    # # fitzpatrick_scale 정규화
    # train_dataset = adjust_label_range(train_dataset, label_column='fitzpatrick_scale', num_classes=config['fitz_classes'])
    # test_dataset = adjust_label_range(test_dataset, label_column='fitzpatrick_scale', num_classes=config['fitz_classes'])

    train_data, valid_data = train_test_split(train_dataset, test_size=0.25, random_state=42)   
    
    train_data = train_data.reset_index(drop=True)
    valid_data = valid_data.reset_index(drop=True)
    
    train_wrapper = TrainDataSetWrapper(train_data, config['batch_size'], config['pad_img_root_dir'], **config['dataset'])
    valid_wrapper = TestDataSetWrapper(valid_data, config['batch_size'], config['pad_img_root_dir'], **config['dataset'])
    test_wrapper = TestDataSetWrapper(test_dataset, config['batch_size'], config['fitz17_img_root_dir'], **config['dataset'])
    # print(len(train_data), len(valid_data), len(test_dataset))

    # train_wrapper = TrainDataSetWrapper(train_dataset, config['batch_size'], **config['dataset'])
    # test_wrapper = TestDataSetWrapper(test_dataset, config['batch_size'], **config['dataset'])
    
    train_loader = train_wrapper.get_data_loaders()
    valid_loader = valid_wrapper.get_data_loaders()
    test_loader = test_wrapper.get_data_loaders() 

    # train_loader, valid_loader = train_wrapper.get_data_loaders()
    # test_loader = test_wrapper.get_data_loaders()
    
    patchalign = PatchAlign(config)
    num_epochs = config['epochs']
    
    max_patience = 10
    patience = 0
    best_valid_accuracy = 0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}')
        
        train_metrics = patchalign.train(train_loader)
        print_metrics("Train", train_metrics)
        print('\n')

        valid_metrics = patchalign.valid(valid_loader)
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
            torch.save(patchalign.get_model().state_dict(), os.path.join('./runs', f'out_patchalign_model_{current_time}.pth'))
            break

    test_metrics = patchalign.test(test_loader)
    print_metrics("Test", test_metrics)
    
    # raw_model = patchalign.get_model()  
    # if isinstance(raw_model, torch.nn.DataParallel):
    #     actual_model = raw_model.module
    # else:
    #     actual_model = raw_model

    # # 활성화와 그래디언트를 2D 공간 이미지로 재구성, 250114
    # def reshape_transform(tensor, height=14, width=14):
    #     result = tensor[:, 1 :  , :].reshape(tensor.size(0),height, width, tensor.size(2))

    #     # Bring the channels to the first dimension,
    #     # like in CNNs.
    #     result = result.transpose(2, 3).transpose(1, 2)
    #     return result

    # # Grad-CAM++ 적용
    # wrapped_model = ModelWrapper(actual_model)
    # # print(wrapped_model.original_model)
    # # target_layers = [wrapped_model.original_model.feature_extractor[7][-1]]  # 모델 구조에 맞게 수정 필요
    # target_layers = [wrapped_model.original_model.feature_extractor.encoder.layer[-1].layernorm_before] # 모델 구조에 맞게 수정 완료, 250113 
    # grad_cam = GradCAMPlusPlus(model=wrapped_model, target_layers=target_layers, reshape_transform=reshape_transform) # reshape_transform 함수 전달, 250114

    # model_name = "patchalign"
    # output_dir = f"/dshome/ddualab/jiwon/skin_cancer_fairness/results/grad_cam_results/{model_name}"
    # os.makedirs(output_dir, exist_ok=True)
    
    # concatenated_label = ['actinic keratosis benign', 'nevus benign', 'seborrheic keratosis benign', 'basal cell carcinoma malignant', 'melanoma malignant', 'squamous cell carcinoma malignant', 'eudermic eudermic']

    # try:
    #     for idx, (images, labels, fitz_scales) in enumerate(test_loader):
    #         if idx == 17:
    #             break
    #         images = images.to(patchalign.device)

    #         # 텍스트 토큰화 추가 250113
    #         xls = patchalign.tokenizer(concatenated_label, return_tensors="pt", padding=True, truncation=True)
    #         xls = {k: v.cuda() for k, v in xls.items()}
            
    #         wrapped_model.current_xls = xls
    #         # outputs = wrapped_model(images)

    #         # fitz_scales_one_hot = F.one_hot(fitz_scales.long() - 1, num_classes=config['fitz_classes']).float()
    #         # fitz_scales_one_hot = fitz_scales_one_hot.to(patchalign.device)
    #         # wrapped_model.current_attribute = fitz_scales_one_hot
            
    #         grayscale_cam = grad_cam(images, targets=None)

    #         for i in range(images.size(0)):
    #             input_image = images[i].permute(1, 2, 0).detach().cpu().numpy() 
    #             input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())
    #             visualization = show_cam_on_image(input_image, grayscale_cam[i, :], use_rgb=True)
                
    #             label = labels[i].item()
    #             fitz_scale = fitz_scales[i].item() 
    #             save_path = os.path.join(output_dir, f"grad_cam_batch{idx}_img{i}_label{label}_fitz{fitz_scale}.png")
    #             plt.imsave(save_path, visualization)
    # finally:
    #     grad_cam.activations_and_grads.release()
    

if __name__ == "__main__":
    main()
    



