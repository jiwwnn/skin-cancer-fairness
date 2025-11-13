from train import SimCLR
import yaml
from skin_cancer_fairness.dataloader.multimodal_dataset_wrapper import DataSetWrapper

# from dataloader.dataset import ClrDataset
# from dataloader.dataset_wrapper import SimCLRDataTransform

def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    
    dataset = DataSetWrapper(config['batch_size'], **config['dataset'])

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


