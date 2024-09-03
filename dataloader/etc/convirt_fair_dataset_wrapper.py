import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from dataloader.convirt_fair_dataset import TrainDataset, TestDataset
# from fair_dataset import TrainDataset, TestDataset
from sklearn.model_selection import train_test_split



np.random.seed(0)


class TrainDataSetWrapper(object):
    def __init__(self, batch_size, num_workers, s, csv_file, img_root_dir, input_shape, img_path_col, text_root_dir, text_col1, label_col, fitz_scale_col):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.s = s
        self.csv_file = csv_file 
        self.img_root_dir = img_root_dir
        self.input_shape = eval(input_shape)
        self.img_path_col = img_path_col
        self.text_root_dir = text_root_dir
        self.text_col1 = text_col1
        self.label_col = label_col
        self.fitz_scale_col = fitz_scale_col
        
    def get_data_loaders(self):
        data_augment = self.get_simclr_pipeline_transform()
        train_dataset = TrainDataset(
            csv_file = self.csv_file,
            img_root_dir = self.img_root_dir,
            input_shape = self.input_shape,
            img_path_col = self.img_path_col,
            text_root_dir = self.text_root_dir, 
            text_col1 = self.text_col1,           
            label_col = self.label_col,
            fitz_scale_col = self.fitz_scale_col,
            transform=SimCLRDataTransform(data_augment)
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
    
class TestDataSetWrapper(object):
    def __init__(self, batch_size, num_workers, s, csv_file, img_root_dir, input_shape, img_path_col, text_root_dir, text_col1, label_col, fitz_scale_col):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.s = s
        self.csv_file = csv_file 
        self.img_root_dir = img_root_dir
        self.input_shape = eval(input_shape)
        self.img_path_col = img_path_col
        self.text_root_dir = text_root_dir
        self.text_col1 = text_col1
        self.label_col = label_col
        self.fitz_scale_col = fitz_scale_col
        
    def get_data_loaders(self):
        data_augment = self.get_simclr_pipeline_transform()
        test_dataset = TestDataset(
            csv_file = self.csv_file,
            img_root_dir = self.img_root_dir,
            input_shape = self.input_shape,
            img_path_col = self.img_path_col,
            text_root_dir = self.text_root_dir, 
            text_col1 = self.text_col1,              
            label_col = self.label_col,
            fitz_scale_col = self.fitz_scale_col,
            transform=SimCLRDataTransform(data_augment)
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

class SimCLRDataTransform(object):
    def __init__(self, transform_image):
        self.transform_image = transform_image

    def __call__(self, sample):
        image = self.transform_image(sample['image'])
        phrase = sample['phrase1']
        # label = sample['label']
        # fitz_scale = sample['fitz_scale']

        return image, phrase

    
import yaml  
if __name__ == '__main__':    
    config = yaml.load(open("/dshome/ddualab/jiwon/ConVIRT-pytorch/config.yaml", "r"), Loader=yaml.FullLoader)
    # train_dataset = TrainClsDataSetWrapper(batch_size=config['batch_size'], **config['train_cls_dataset'])
    train_dataset = TrainDataSetWrapper(batch_size=config['batch_size'], **config['train_dataset'])
    train_loader, valid_loader = train_dataset.get_data_loaders()
    print(len(train_loader))
