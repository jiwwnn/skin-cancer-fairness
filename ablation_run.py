import os 
import torch
import numpy as np
from ablation_train import CL1CLS
import yaml
from dataloader.ours_dataset_wrapper import TrainDataSetWrapper, TestDataSetWrapper
from datetime import datetime

def print_metrics(stage, metrics):
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            value_str = ", ".join(f"{v:.4f}" for v in value)  
            print(f'{stage} {key.replace("_", " ").capitalize()}: [{value_str}]', end=' ')
        else:
            print(f'{stage} {key.replace("_", " ").capitalize()}: {value:.4f}', end=' ')

def main():
    config = yaml.load(open("ablation_config.yaml", "r"), Loader=yaml.FullLoader)
    
    train_wrapper = TrainDataSetWrapper(config['batch_size'], **config['dataset'])
    test_wrapper = TestDataSetWrapper(config['batch_size'], **config['dataset'])
    
    train_loader, valid_loader = train_wrapper.get_data_loaders()
    test_loader = test_wrapper.get_data_loaders()
    
    cl1cls = CL1CLS(config)
    num_epochs = config['epochs']
    
    max_patience = 5
    patience = 0
    best_valid_accuracy = 0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}')
        
        train_metrics = cl1cls.train(train_loader)
        print_metrics("Train", train_metrics)
        print('\n')

        valid_metrics = cl1cls.valid(valid_loader)
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
            torch.save(ours.get_model().state_dict(), os.path.join('./runs', f'ablation_model_{current_time}.pth'))
            break

    test_metrics = cl1cls.test(test_loader)
    print_metrics("Test", test_metrics)

if __name__ == "__main__":
    main()
    