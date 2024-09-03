from skin_cancer_fairness.ours_train import SimCLR
import yaml
from dataloader.fair_dataset_wrapper import TrainDataSetWrapper, TestDataSetWrapper

# from dataloader.dataset import ClrDataset
# from dataloader.dataset_wrapper import SimCLRDataTransform

def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    
    train_wrapper = TrainDataSetWrapper(config['batch_size'], **config['train_dataset'])
    test_wrapper = TestDataSetWrapper(config['batch_size'], **config['test_dataset'])
    
    train_loader, valid_loader = train_wrapper.get_data_loaders()
    test_loader = test_wrapper.get_data_loaders()
    
    simclr = SimCLR(config)
    num_epochs = config['epochs']
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}')
        
        train_loss, train_cl1_cls_loss, train_cl2_loss, train_accuracy, train_lighter_accuracy, train_darker_accuracy, train_f1 = simclr.train(train_loader)
        print(f'Train Loss: {train_loss:.4f}, Train CL1 CLS Loss: {train_cl1_cls_loss:.4f}, Train CL2 Loss: {train_cl2_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train Lighter Accuracy: {train_lighter_accuracy:.4f}, Train Darker Accuracy: {train_darker_accuracy:.4f}, Train F1: {train_f1:.4f}')

        valid_loss, valid_cl1_cls_loss, valid_cl2_loss, valid_accuracy, valid_lighter_accuracy, valid_darker_accuracy, valid_f1 = simclr.validate(valid_loader)
        print(f'Valid Loss: {valid_loss:.4f}, Valid CL1 CLS Loss: {valid_cl1_cls_loss:.4f}, Valid CL2 Loss: {valid_cl2_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}, Valid Lighter Accuracy: {valid_lighter_accuracy:.4f}, Valid Darker Accuracy: {valid_darker_accuracy:.4f}, Valid F1: {valid_f1:.4f}')

        if epoch == num_epochs - 1:
            test_loss, test_cl1_cls_loss, test_cl2_loss, test_accuracy, test_lighter_accuracy, test_darker_accuracy, test_f1 = simclr.test(test_loader)
            print(f'Test Loss: {test_loss:.4f}, Test CL1 CLS Loss: {test_cl1_cls_loss:.4f}, Test CL2 Loss: {test_cl2_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test Lighter Accuracy: {test_lighter_accuracy:.4f}, Test Darker Accuracy: {test_darker_accuracy:.4f}, Test F1 Score: {test_f1:.4f}')

if __name__ == "__main__":
    main()
    
    # dataset = ClrDataset(csv_file='/dshome/ddualab/jiwon/ConVIRT-pytorch/data/lesion_txt.csv', 
    #                      img_root_dir='/dshome/ddualab/jiwon/ConVIRT-pytorch/data/skin_cancer_images',
    #                      input_shape=((224, 224, 3)),
    #                      img_path_col='img_id',
    #                      text_col='description',
    #                      text_from_files='',
    #                      text_root_dir='/dshome/ddualab/jiwon/ConVIRT-pytorch/data/lesion_txt.csv'
    #                      )
    
    # print(len(dataset))


