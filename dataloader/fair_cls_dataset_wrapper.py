import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from dataloader.fair_cls_dataset import TrainImageDataset, TestImageDataset
from sklearn.model_selection import train_test_split
# from fair_cls_dataset import TrainImageDataset, TestImageDataset


np.random.seed(0)


class TrainClsDataSetWrapper(object):
    def __init__(self, batch_size, num_workers, input_shape, s, csv_file, img_root_dir, text_root_dir, img_path_col, label_col, fitz_scale_col):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.s = s
        self.input_shape = eval(input_shape)
        self.csv_file = csv_file
        self.img_root_dir = img_root_dir
        self.text_root_dir = text_root_dir
        self.img_path_col = img_path_col
        self.label_col = label_col
        self.fitz_scale_col = fitz_scale_col
        
    def get_data_loaders(self):
        data_augment = self.get_simclr_pipeline_transform()
        train_dataset = TrainImageDataset(
            csv_file = self.csv_file,
            img_root_dir = self.img_root_dir,
            text_root_dir = self.text_root_dir, 
            img_path_col = self.img_path_col,
            label_col = self.label_col,
            fitz_scale_col = self.fitz_scale_col,
            transform = data_augment
        )
    

        train_indices, valid_indices = train_test_split(list(range(len(train_dataset))))        
        np.random.shuffle(train_indices)
        np.random.shuffle(valid_indices)
    
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler, 
                                  num_workers=self.num_workers, drop_last=True, shuffle=False)
        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler, 
                                  num_workers=self.num_workers, drop_last=True, shuffle=False)
        
        return train_loader, valid_loader
    
    
    def get_simclr_pipeline_transform(self):
        data_transforms = transforms.Compose([
            transforms.Resize((self.input_shape[0], self.input_shape[1])),
            # transforms.RandomResizedCrop(size=self.input_shape[0], scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))            
        ])
        
        return data_transforms
    
class TestClsDataSetWrapper(object):
    def __init__(self, batch_size, num_workers, input_shape, s, csv_file, img_root_dir, text_root_dir, img_path_col, label_col, fitz_scale_col):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.s = s
        self.input_shape = eval(input_shape)
        self.csv_file = csv_file
        self.img_root_dir = img_root_dir
        self.text_root_dir = text_root_dir
        self.img_path_col = img_path_col
        self.label_col = label_col
        self.fitz_scale_col = fitz_scale_col
        
    def get_data_loaders(self):
        data_augment = self.get_simclr_pipeline_transform()
        test_dataset = TestImageDataset(
            csv_file = self.csv_file,
            img_root_dir = self.img_root_dir,
            text_root_dir = self.text_root_dir, 
            img_path_col = self.img_path_col,
            label_col = self.label_col,
            fitz_scale_col = self.fitz_scale_col,
            transform = data_augment
        )
        
        
        num_test = len(test_dataset)
        indices = list(range(num_test))
        np.random.shuffle(indices)
        
        test_sampler = SubsetRandomSampler(indices)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, sampler=test_sampler, 
                                  num_workers=self.num_workers, drop_last=True, shuffle=False)
        
        return test_loader
    
    
    def get_simclr_pipeline_transform(self):
        data_transforms = transforms.Compose([
            transforms.Resize((self.input_shape[0], self.input_shape[1])),
            # transforms.RandomResizedCrop(size=self.input_shape[0], scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))            
        ])
        
        return data_transforms
    
    # def get_train_validation_data_loaders(self, train_dataset, test_dataset):
    #     num_train = len(train_dataset)
    #     indices = list(range(num_train))
    #     np.random.shuffle(indices)
        
    #     split = int(np.floor(self.valid_size * num_train))
    #     train_idx, valid_idx = indices[split:], indices[:split]
        
    #     train_sampler = SubsetRandomSampler(train_idx)
    #     valid_sampler = SubsetRandomSampler(valid_idx)
        
    #     train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler, 
    #                               num_workers=self.num_workers, drop_last=True, shuffle=False)
        
    #     valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler, 
    #                               num_workers=self.num_workers, drop_last=True)
    #     return train_loader, valid_loader
    
    
import yaml  
if __name__ == '__main__':    
    config = yaml.load(open("/dshome/ddualab/jiwon/ConVIRT-pytorch/config.yaml", "r"), Loader=yaml.FullLoader)
    # train_dataset = TrainClsDataSetWrapper(batch_size=config['batch_size'], **config['train_cls_dataset'])
    test_dataset = TestClsDataSetWrapper(batch_size=config['sbatch_size'], **config['test_cls_dataset'])
    test_loader = test_dataset.get_data_loaders()
    print(len(test_loader))