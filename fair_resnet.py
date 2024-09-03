import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18, resnet50
from tqdm import tqdm
import os
import yaml

from dataloader.fair_cls_dataset_wrapper import TrainClsDataSetWrapper, TestClsDataSetWrapper
from sklearn.metrics import f1_score

torch.manual_seed(0)

class ResNet50Classifier:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def train_cls(self, train_cls_loader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        correct = 0 
        lighter_correct = 0
        darker_correct = 0
        total = 0
        lighter_total = 0
        darker_total = 0
        
        all_labels = []
        all_predictions = []
        
        for images, labels, fitz_scales in tqdm(train_cls_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            fitz_scales = fitz_scales.to(self.device)
            
            optimizer.zero_grad()
            
            outputs = self.model.to(self.device)(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            
            lighter_mask = (fitz_scales == 1.0) | (fitz_scales == 2.0)
            lighter_correct += ((predicted == labels) & lighter_mask).sum().item()
            lighter_total += lighter_mask.sum().item()

            darker_mask = (fitz_scales == 3.0) | (fitz_scales == 4.0)
            darker_correct += ((predicted == labels) & darker_mask).sum().item()
            darker_total += darker_mask.sum().item()     
            
        avg_loss = total_loss / len(train_cls_loader)
        accuracy = correct / total
        lighter_accuracy = lighter_correct / lighter_total
        darker_accuracy = darker_correct / darker_total
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        return avg_loss, accuracy, lighter_accuracy, darker_accuracy, f1
    
    def valid_cls(self, valid_cls_loader, criterion):
        self.model.eval()
        total_loss = 0
        correct = 0
        lighter_correct = 0
        darker_correct = 0
        total = 0
        lighter_total = 0
        darker_total = 0
        
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for images, labels, fitz_scales in tqdm(valid_cls_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                fitz_scales = fitz_scales.to(self.device)

                outputs = self.model.to(self.device)(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                
                lighter_mask = (fitz_scales == 1.0) | (fitz_scales == 2.0)
                lighter_correct += ((predicted == labels) & lighter_mask).sum().item()
                lighter_total += lighter_mask.sum().item()

                darker_mask = (fitz_scales == 3.0) | (fitz_scales == 4.0)
                darker_correct += ((predicted == labels) & darker_mask).sum().item()
                darker_total += darker_mask.sum().item()
                

        avg_loss = total_loss / len(valid_cls_loader)
        accuracy = correct / total
        lighter_accuracy = lighter_correct / lighter_total
        darker_accuracy = darker_correct / darker_total
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        return avg_loss, accuracy, lighter_accuracy, darker_accuracy, f1
    
    
    def test_cls(self, test_cls_loader, criterion):
        self.model.eval()
        total_loss = 0
        correct = 0
        lighter_correct = 0
        darker_correct = 0
        total = 0
        lighter_total = 0
        darker_total = 0
        
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for images, labels, fitz_scales in tqdm(test_cls_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                fitz_scales = fitz_scales.to(self.device)

                outputs = self.model.to(self.device)(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                
                lighter_mask = (fitz_scales == 1.0) | (fitz_scales == 2.0)
                lighter_correct += ((predicted == labels) & lighter_mask).sum().item()
                lighter_total += lighter_mask.sum().item()

                darker_mask = (fitz_scales == 3.0) | (fitz_scales == 4.0)
                darker_correct += ((predicted == labels) & darker_mask).sum().item()
                darker_total += darker_mask.sum().item()


        avg_loss = total_loss / len(test_cls_loader)
        accuracy = correct / total
        lighter_accuracy = lighter_correct / lighter_total
        darker_accuracy = darker_correct / darker_total
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        return avg_loss, accuracy, lighter_accuracy, darker_accuracy, f1             
                
                
def resnet50_cls_main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = resnet50(pretrained=True).to(device)    
    # model.load_state_dict(torch.load('/dshome/ddualab/jiwon/ConVIRT-pytorch/runs/Jun24_23-03-38_daintlabA/checkpoints/image_encoder1.pth'), strict=False)
    model.fc = nn.Linear(model.fc.in_features, 6)
    classifier = ResNet50Classifier(model, device)
    
    config = yaml.load(open("/dshome/ddualab/jiwon/ConVIRT-pytorch/config.yaml", "r"), Loader=yaml.FullLoader)
    train_cls_wrapper = TrainClsDataSetWrapper(config['batch_size'], **config['train_dataset'])
    test_cls_wrapper = TestClsDataSetWrapper(config['batch_size'], **config['test_dataset'])
    
    train_cls_loader, valid_cls_loader = train_cls_wrapper.get_data_loaders()
    test_cls_loader = test_cls_wrapper.get_data_loaders()
    
    optimizer = optim.Adam(classifier.model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    num_epochs = 20
    # max_patience = 10
    # patience = 0
    # best_valid_accuracy = 0
    test_performed = False 
    
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}') 
        
        train_loss, train_accuracy, lighter_train_accuracy, darker_train_accuracy, train_f1 = classifier.train_cls(train_cls_loader, optimizer, criterion)
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Lighter Train Accuracy: {lighter_train_accuracy:.4f}, Darker Train Accuracy: {darker_train_accuracy:.4f}, Train F1 Score: {train_f1:.4f}')
        
        valid_loss, valid_accuracy, lighter_valid_accuracy, darker_valid_accuracy, valid_f1 = classifier.valid_cls(valid_cls_loader, criterion)
        print(f'Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.4f}, Lighter Validation Accuracy: {lighter_valid_accuracy:.4f}, Darker Validation Accuracy: {darker_valid_accuracy:.4f}, Valid F1 Score: {valid_f1:.4f}')

        # if lighter_valid_accuracy > best_valid_accuracy:
        #     best_valid_accuracy = lighter_valid_acscuracy
        #     patience = 0
        # else:
        #     patience +=1
        #     print(f'No improvement in validation accuracy for {patience} epochs.')
        
        # if patience >= max_patience:
        #     print('Early Stopping triggered.')
        #     break

        if not test_performed and epoch == num_epochs - 1:
            test_loss, test_accuracy, lighter_test_accuracy, darker_test_accuracy, test_f1 = classifier.test_cls(test_cls_loader, criterion)
            print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Lighter Test Accuracy: {lighter_test_accuracy:.4f}, Darker Test Accuracy: {darker_test_accuracy:.4f}, Test F1 Score: {test_f1:.4f}')
        
if __name__ == '__main__':
    resnet50_cls_main()        
        
    