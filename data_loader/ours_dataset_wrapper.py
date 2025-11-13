import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from dataloader.ours_dataset import CustomDataset
from sklearn.model_selection import train_test_split

np.random.seed(0)
torch.manual_seed(0)

class TrainDataSetWrapper(object):
    def __init__(self, df, batch_size, img_root_dir, num_workers, s, input_shape, img_path_col, text_col1, text_col2, label_col, fitz_scale_col):
        self.df = df
        # self.valid_df = valid_df
        self.batch_size = batch_size
        self.img_root_dir = img_root_dir
        self.num_workers = num_workers
        self.s = s
        self.input_shape = eval(input_shape)
        self.img_path_col = img_path_col
        self.text_col1 = text_col1
        self.text_col2 = text_col2
        self.label_col = label_col
        self.fitz_scale_col = fitz_scale_col
        
    def get_data_loaders(self):
        data_augment = self.get_simclr_pipeline_transform()
        dataset = CustomDataset(
            df = self.df,
            img_root_dir = self.img_root_dir,
            input_shape = self.input_shape,
            img_path_col = self.img_path_col,
            text_col1 = self.text_col1, 
            text_col2 = self.text_col2,              
            label_col = self.label_col,
            fitz_scale_col = self.fitz_scale_col,
            transform=SimCLRDataTransform(data_augment)
        )
        # valid_dataset = CustomDataset(
        #     df = self.valid_df,
        #     img_root_dir = self.img_root_dir,
        #     input_shape = self.input_shape,
        #     img_path_col = self.img_path_col,
        #     text_col1 = self.text_col1, 
        #     text_col2 = self.text_col2,              
        #     label_col = self.label_col,
        #     fitz_scale_col = self.fitz_scale_col,
        #     transform=SimCLRDataTransform(data_augment)
        # )
        
        # # 클래스별 샘플 개수 계산
        # class_sample_count = dataset.df[self.label_col].value_counts().sort_index()
        
        # # 클래스별 가중치 계산 (샘플 개수의 역수)
        # weight = 1. / class_sample_count
        # dataset.df['sample_weight'] = dataset.df[self.label_col].map(weight)
        
        # # 샘플별 가중치를 텐서로 변환
        # samples_weight = torch.tensor(dataset.df['sample_weight'].values).type('torch.DoubleTensor')

        # # WeightedRandomSampler 적용
        # sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

        # train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=sampler, 
        #                           num_workers=self.num_workers, drop_last=True)
        # valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False, 
        #                           num_workers=self.num_workers, drop_last=True)

        # return train_loader, valid_loader

        # train_indices, valid_indices = train_test_split(list(range(len(dataset))))        
        # np.random.shuffle(train_indices)
        # np.random.shuffle(valid_indices)
    
        # train_sampler = SubsetRandomSampler(train_indices)
        # valid_sampler = SubsetRandomSampler(valid_indices)
        
        # train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler, 
        #                           num_workers=self.num_workers, drop_last=True, shuffle=False)
        # valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler, 
        #                           num_workers=self.num_workers, drop_last=True, shuffle=False)
        
        # return train_loader, valid_loader

        # num_data = len(dataset)
        # indices = list(range(num_data))
        # np.random.shuffle(indices) # 시드로 결과 고정
        
        # sampler = SubsetRandomSampler(indices)
        loader = DataLoader(dataset, batch_size=self.batch_size,
                            num_workers=self.num_workers, drop_last=True, shuffle=True)    
        return loader
    
    
    def get_simclr_pipeline_transform(self):
        data_transforms = transforms.Compose([
            transforms.Resize((self.input_shape[0], self.input_shape[1])),
            # transforms.RandomResizedCrop(size=self.input_shape[0], scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(), # 기본값 p=0.5
            transforms.RandomVerticalFlip(),
            # transforms.RandomGrayscale(p=0.2),
            transforms.RandomRotation(15), # 추가 
            # transforms.AutoAugment(policy=v2.AutoAugmentPolicy.IMAGENET), # 추가 
            # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10) ,  # x, y 방향으로 10% 이동 # 최대 10도 밀림
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))            
        ])
        
        return data_transforms
    
class TestDataSetWrapper(object):
    def __init__(self, df, batch_size, img_root_dir, num_workers, s, input_shape, img_path_col, text_col1, text_col2, label_col, fitz_scale_col):
        self.df = df
        self.batch_size = batch_size
        self.img_root_dir = img_root_dir
        self.num_workers = num_workers
        self.s = s
        self.input_shape = eval(input_shape)
        self.img_path_col = img_path_col
        self.text_col1 = text_col1
        self.text_col2 = text_col2
        self.label_col = label_col
        self.fitz_scale_col = fitz_scale_col
        
    def get_data_loaders(self):
        data_augment = self.get_simclr_pipeline_transform()
        dataset = CustomDataset(
            df = self.df,
            img_root_dir = self.img_root_dir,
            input_shape = self.input_shape,
            img_path_col = self.img_path_col,
            text_col1 = self.text_col1, 
            text_col2 = self.text_col2,              
            label_col = self.label_col,
            fitz_scale_col = self.fitz_scale_col,
            transform=SimCLRDataTransform(data_augment)
        )
        
        # num_data = len(dataset)
        # indices = list(range(num_data))
        # np.random.shuffle(indices) # 시드로 결과 고정
        
        # sampler = SubsetRandomSampler(indices)
        loader = DataLoader(dataset, batch_size=self.batch_size, 
                                  num_workers=self.num_workers, drop_last=True, shuffle=False)    
        return loader
    
    def get_simclr_pipeline_transform(self):
        data_transforms = transforms.Compose([
            transforms.Resize((self.input_shape[0], self.input_shape[1])),
            # transforms.RandomResizedCrop(size=self.input_shape[0], scale=(0.8, 1.0)),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.RandomRotation(15), # 추가
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
        phrase1 = sample['phrase1']
        phrase2 = sample['phrase2']
        label = sample['label']
        fitz_scale = sample['fitz_scale']

        return image, phrase1, phrase2, label, fitz_scale

    
import yaml  
if __name__ == '__main__':    
    config = yaml.load(open("/dshome/ddualab/jiwon/ConVIRT-pytorch/config.yaml", "r"), Loader=yaml.FullLoader)
    # train_dataset = TrainClsDataSetWrapper(batch_size=config['batch_size'], **config['train_cls_dataset'])
    train_dataset = TrainDataSetWrapper(batch_size=config['batch_size'], **config['train_dataset'])
    train_loader= train_dataset.get_data_loaders()
    print(len(train_loader))
