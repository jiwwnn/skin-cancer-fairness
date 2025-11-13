import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from dataloader.baseline_dataset import CustomDataset
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import torch

np.random.seed(0)
torch.manual_seed(0)

class TrainDataSetWrapper(object):
    def __init__(self, df, batch_size, img_root_dir, num_workers, input_shape, s, img_path_col, label_col, fitz_scale_col):
        self.df = df
        # self.valid_df = valid_df
        self.batch_size = batch_size
        self.img_root_dir = img_root_dir
        self.num_workers = num_workers
        self.s = s
        self.input_shape = eval(input_shape)
        self.img_path_col = img_path_col
        self.label_col = label_col
        self.fitz_scale_col = fitz_scale_col
        
    def get_data_loaders(self):
        data_augment = self.get_simclr_pipeline_transform()
        dataset = CustomDataset(
            df = self.df,
            img_root_dir = self.img_root_dir,
            img_path_col = self.img_path_col,
            label_col = self.label_col,
            fitz_scale_col = self.fitz_scale_col,
            transform = data_augment
        )
        # valid_dataset = CustomDataset(
        #     df = self.valid_df,
        #     img_root_dir = self.img_root_dir,
        #     img_path_col = self.img_path_col,
        #     label_col = self.label_col,
        #     fitz_scale_col = self.fitz_scale_col,
        #     transform = data_augment
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
    
        # train_indices, valid_indices = list(range(len(train_dataset))), list(range(len(valid_dataset)))
    
        # train_indices, valid_indices = train_test_split(list(range(len(train_dataset))))        
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
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15), # 추가 
            # transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))            
        ])
        
        return data_transforms

class TrainDataSetWrapper_RESM:
    def __init__(self, df, batch_size, img_root_dir, num_workers, input_shape, s, img_path_col, label_col, fitz_scale_col):
        self.df = df
        # self.valid_df = valid_df
        self.batch_size = batch_size
        self.img_root_dir = img_root_dir
        self.num_workers = num_workers
        self.s = s
        self.input_shape = eval(input_shape)
        self.img_path_col = img_path_col
        self.label_col = label_col
        self.fitz_scale_col = fitz_scale_col

    def get_data_loaders(self):
       
        data_augment = self.get_simclr_pipeline_transform()
        dataset = CustomDataset(
            df=self.df, 
            img_root_dir=self.img_root_dir,
            img_path_col=self.img_path_col,
            label_col=self.label_col,
            fitz_scale_col=self.fitz_scale_col,
            transform=data_augment
        )

        # train_dataset = CustomDataset(
        #     df=self.train_df, 
        #     img_root_dir=self.img_root_dir,
        #     img_path_col=self.img_path_col,
        #     label_col=self.label_col,
        #     fitz_scale_col=self.fitz_scale_col,
        #     transform=data_augment
        # )

        # valid_dataset = CustomDataset(
        #     df=self.valid_df, 
        #     img_root_dir=self.img_root_dir,
        #     img_path_col=self.img_path_col,
        #     label_col=self.label_col,
        #     fitz_scale_col=self.fitz_scale_col,
        #     transform=data_augment
        # )

        # 샘플 가중치 계산 (수정 -> fitzpatrick과 label 조합)
        # 가중치는 1 / 조합 개수로 정의
        # class_sample_count = train_dataset.df.groupby([self.fitz_scale_col, self.label_col]).size()
        # weight = 1. / class_sample_count
        
        # train_dataset.df['sample_weight'] = train_dataset.df.apply(
        #     lambda x: weight.loc[(x[self.fitz_scale_col], x[self.label_col])], axis=1
        # )
        samples_weight = torch.tensor(dataset.df['sample_weight'].values).type('torch.DoubleTensor')
        
        # WeightedRandomSampler 적용
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

        loader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler, 
                                  num_workers=self.num_workers, drop_last=True, shuffle=False)
        # valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, 
        #                           num_workers=self.num_workers, drop_last=True, shuffle=False)

        return loader

    def get_simclr_pipeline_transform(self):
        data_transforms = transforms.Compose([
            transforms.Resize((self.input_shape[0], self.input_shape[1])),
            transforms.RandomRotation(15), # 추가 
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        return data_transforms

class TrainDataSetWrapper_REWT:
    def __init__(self, df, batch_size, img_root_dir, num_workers, s, input_shape, img_path_col, label_col, fitz_scale_col):
        self.df = df
        # self.valid_df = valid_df
        self.batch_size = batch_size
        self.img_root_dir = img_root_dir
        self.num_workers = num_workers
        self.s = s
        self.input_shape = eval(input_shape)
        self.img_path_col = img_path_col
        self.label_col = label_col
        self.fitz_scale_col = fitz_scale_col

    def get_data_loaders(self):
        data_augment = self.get_simclr_pipeline_transform()
        dataset = CustomDataset(
            df = self.df,
            img_root_dir=self.img_root_dir,
            img_path_col=self.img_path_col,
            label_col=self.label_col,
            fitz_scale_col=self.fitz_scale_col,
            transform=data_augment
        )
        # valid_dataset = CustomDataset(
        #     df = self.valid_df,
        #     img_root_dir=self.img_root_dir,
        #     img_path_col=self.img_path_col,
        #     label_col=self.label_col,
        #     fitz_scale_col=self.fitz_scale_col,
        #     transform=data_augment
        # )

        # total_samples = len(train_dataset.df) # 총 데이터 개수
        # fitz_count = train_dataset.df[self.fitz_scale_col].value_counts() # # Fitzpatrick별 개수
        # label_count = train_dataset.df[self.label_col].value_counts() # # Label별 개수
        
        # # 각 Fitz & Label 조합별 개수
        # def calculate_weight(row):
        #     fitz = row[self.fitz_scale_col]
        #     label = row[self.label_col]

        #     # 조합 개수 계산
        #     joint_count = len(train_dataset.df[
        #         (train_dataset.df[self.fitz_scale_col] == fitz) &
        #         (train_dataset.df[self.label_col] == label)
        #     ])

        #     # 가중치 계산
        #     if joint_count > 0:
        #         return (fitz_count[fitz] * label_count[label]) / (joint_count * total_samples)
        #     else:
        #         return 0  # 조합이 없으면 -> 0반환 
        
        # # 샘플별 가중치 계산
        # train_dataset.df['sample_weight'] = train_dataset.df.apply(calculate_weight, axis=1)
        
        samples_weight = torch.tensor(dataset.df['sample_weight'].values).type('torch.DoubleTensor')

        sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

        loader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler, 
                                  num_workers=self.num_workers, drop_last=True, shuffle=False)
        # valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, 
        #                           num_workers=self.num_workers, drop_last=True, shuffle=False)

        return loader

    def get_simclr_pipeline_transform(self):
        data_transforms = transforms.Compose([
            transforms.Resize((self.input_shape[0], self.input_shape[1])),
            transforms.RandomRotation(15), # 추가 
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        return data_transforms

# class TrainDataSetWrapper_ATRB:
#     def __init__(self, batch_size, num_workers, s, input_shape, csv_file, img_root_dir, text_root_dir, img_path_col, label_col, fitz_scale_col):
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.s = s
#         self.input_shape = eval(input_shape)
#         self.csv_file = csv_file
#         self.img_root_dir = img_root_dir
#         self.text_root_dir = text_root_dir  
#         self.img_path_col = img_path_col
#         self.label_col = label_col
#         self.fitz_scale_col = fitz_scale_col

#     def get_data_loaders(self):
        
#         data_augment = self.get_simclr_pipeline_transform()
#         train_dataset = TrainDataset(
#             csv_file=self.csv_file,
#             img_root_dir=self.img_root_dir,
#             text_root_dir=self.text_root_dir,  
#             img_path_col=self.img_path_col,
#             label_col=self.label_col,
#             fitz_scale_col=self.fitz_scale_col,
#             transform=data_augment
#         )

#         # 클래스별 샘플 개수 계산
#         class_sample_count = train_dataset.df[self.label_col].value_counts().sort_index()
        
#         # 클래스별 가중치 계산 (샘플 개수의 역수)
#         weight = 1. / class_sample_count
#         train_dataset.df['sample_weight'] = train_dataset.df[self.label_col].map(weight)
        
#         # 샘플별 가중치를 텐서로 변환
#         samples_weight = torch.tensor(train_dataset.df['sample_weight'].values).type('torch.DoubleTensor')

#         # WeightedRandomSampler 적용
#         sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

#         train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=sampler, 
#                                   num_workers=self.num_workers, drop_last=True)
#         valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, 
#                                   num_workers=self.num_workers, drop_last=True)

#         return train_loader, valid_loader

#     def get_simclr_pipeline_transform(self):
#         # 데이터 증강 파이프라인 설정
#         data_transforms = transforms.Compose([
#             transforms.Resize((self.input_shape[0], self.input_shape[1])),  # 이미지 크기 조정
#             transforms.RandomHorizontalFlip(),  # 랜덤 가로 뒤집기
#             transforms.RandomVerticalFlip(),  # 랜덤 세로 뒤집기
#             transforms.ToTensor(),  # 텐서로 변환
#             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 정규화
#         ])
#         return data_transforms

class TestDataSetWrapper(object):
    def __init__(self, df, batch_size, img_root_dir, num_workers, input_shape, s, img_path_col, label_col, fitz_scale_col):
        self.df = df
        self.batch_size = batch_size
        self.img_root_dir = img_root_dir
        self.num_workers = num_workers
        self.s = s
        self.input_shape = eval(input_shape)
        self.img_path_col = img_path_col
        self.label_col = label_col
        self.fitz_scale_col = fitz_scale_col
        
    def get_data_loaders(self):
        data_augment = self.get_simclr_pipeline_transform()
        dataset = CustomDataset(
            df = self.df, 
            img_root_dir = self.img_root_dir,
            img_path_col = self.img_path_col,
            label_col = self.label_col,
            fitz_scale_col = self.fitz_scale_col,
            transform = data_augment
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
            # transforms.RandomRotation(15), # 추가 
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
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
    test_dataset = TestDataSetWrapper(batch_size=config['batch_size'], **config['test_cls_dataset'])
    test_loader = test_dataset.get_data_loaders()
    print(len(test_loader))