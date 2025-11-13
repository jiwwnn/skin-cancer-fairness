from convirt_cl_train import SimCLR
import yaml
from dataloader.convirt_fair_dataset_wrapper import TrainDataSetWrapper

# from dataloader.dataset import ClrDataset
# from dataloader.dataset_wrapper import SimCLRDataTransform

def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    
    dataset = TrainDataSetWrapper(config['batch_size'], **config['train_dataset'])

    simclr = SimCLR(dataset, config)
    simclr.train()


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


