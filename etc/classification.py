import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from tqdm import tqdm
import os
import yaml

from dataloader.cls_dataset_wrapper import TrainClsDataSetWrapper, TestClsDataSetWrapper

class ImageClassifier:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def train_cls(self, train_cls_loader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        correct = 0 
        total = 0
        
        for images, labels in tqdm(train_cls_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            
            outputs = self.model.to(self.device)(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        avg_loss = total_loss / len(train_cls_loader)
        accuracy = correct / total
        return avg_loss, accuracy
    
    def test_cls(self, test_cls_loader, criterion):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(test_cls_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model.to(self.device)(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(test_cls_loader)
        accuracy = correct / total
        return avg_loss, accuracy                
                
                
def cls_main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = resnet50(pretrained=False).to(device)    
    model.load_state_dict(torch.load('/dshome/ddualab/jiwon/ConVIRT-pytorch/runs/Jun24_23-03-38_daintlabA/checkpoints/image_encoder1.pth'), strict=False)
    # model = torch.load('/dshome/ddualab/jiwon/ConVIRT-pytorch/runs/Jun17_22-26-59_daintlabB/checkpoints/image_encoder1.pth')
    model.fc = nn.Linear(model.fc.in_features, 2)
    classifier = ImageClassifier(model, device)
    
    config = yaml.load(open("/dshome/ddualab/jiwon/ConVIRT-pytorch/config.yaml", "r"), Loader=yaml.FullLoader)
    train_cls_wrapper = TrainClsDataSetWrapper(config['batch_size'], **config['train_cls_dataset'])
    test_cls_wrapper = TestClsDataSetWrapper(config['batch_size'], **config['test_cls_dataset'])
    
    train_cls_loader = train_cls_wrapper.get_data_loaders()
    test_cls_loader = test_cls_wrapper.get_data_loaders()
    
    optimizer = optim.Adam(classifier.model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    num_epochs = 50
    max_patience = 10
    patience = 0
    best_valid_accuracy = 0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}') 
        
        train_loss, train_accuracy = classifier.train_cls(train_cls_loader, optimizer, criterion)
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        
        valid_loss, valid_accuracy = classifier.test_cls(test_cls_loader, criterion)
        print(f'Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.4f}')

        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            patience = 0
        else:
            patience +=1
            print(f'No improvement in validation accuracy for {patience} epochs.')
        
        if patience >= max_patience:
            print('Early Stopping triggered.')
            break
        
if __name__ == '__main__':
    cls_main()        
        