"""
Reference for BERT Sentence Embeddings method

@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "http://arxiv.org/abs/1908.10084",

"""

import torch.nn as nn
import torch.nn.functional as F
# import torchvision.models as models
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
import torch
from transformers import AutoModel
import yaml

config = yaml.load(open("ours_config.yaml", "r"), Loader=yaml.FullLoader)

class ModelOurs(nn.Module):
    def __init__(self, res_base_model, bert_base_model, out_dim, freeze_layers, do_lower_case):
        super(ModelOurs, self).__init__()
        # init BERT
        self.bert_model = self._get_bert_basemodel(bert_base_model, freeze_layers)
        
        # projection MLP for BERT model
        self.bert_l1 = nn.Linear(768, 768) # 768 is the size of the BERT embeddings
        self.bert_l2 = nn.Linear(768, out_dim) # 768 is the size of the BERT embeddings

        # init ResNet for two encoders
        self.resnet_dict = {"resnet18": resnet18(weights=ResNet18_Weights.IMAGENET1K_V1), 
                            "resnet50": resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)} 
        
        resnet1 = self._get_res_basemodel(res_base_model)
        resnet2 = self._get_res_basemodel(res_base_model)
        
        num_ftrs1 = resnet1.fc.in_features
        num_ftrs2 = resnet2.fc.in_features
        
        self.res_features1 = nn.Sequential(*list(resnet1.children())[:-1])
        self.res_features2 = nn.Sequential(*list(resnet2.children())[:-1])
        
        # projection MLP for ResNet encoders
        self.res_l1_1 = nn.Linear(num_ftrs1, num_ftrs1)
        self.res_l2_1 = nn.Linear(num_ftrs1, out_dim)
        
        self.res_l1_2 = nn.Linear(num_ftrs2, num_ftrs2)
        self.res_l2_2 = nn.Linear(num_ftrs2, out_dim)
        
        # fully connected layers for classification
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, config['classes'])
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def _get_res_basemodel(self, res_model_name):
        try:
            res_model = self.resnet_dict[res_model_name]
            print("Image feature extractor:", res_model_name)
            return res_model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def _get_bert_basemodel(self, bert_model_name, freeze_layers):
        try:
            model = AutoModel.from_pretrained(bert_model_name)
            print("Text feature extractor:", bert_model_name)
        except:
            raise ("Invalid model name. Check the config file and pass a BERT model from transformers library")
        
        if freeze_layers is not None:
            for layer_idx in freeze_layers:
                for param in list(model.encoder.layer[layer_idx].parameters()):
                    param.requires_grad = False
        return model

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def image_encoder_1(self, xis):
        h = self.res_features1(xis)
        h = torch.flatten(h, start_dim=1)
        # h = h.squeeze() # 20250204 코드 수정

        x = self.res_l1_1(h)
        x = F.relu(x)
        x = self.res_l2_1(x)
        
        return x

    def image_encoder_2(self, xis):
        h = self.res_features2(xis)
        h = torch.flatten(h, start_dim=1)
        # h = h.squeeze() # 20250204 코드 수정
        
        x = self.res_l1_2(h)
        x = F.relu(x)
        x = self.res_l2_2(x)
        
        return x

    def text_encoder(self, encoded_inputs):
        outputs = self.bert_model(**encoded_inputs)
        
        with torch.no_grad():
            sentence_embeddings = self.mean_pooling(outputs, encoded_inputs['attention_mask']).float()
            x = self.bert_l1(sentence_embeddings)
            x = F.relu(x)
            out_emb = self.bert_l2(x)

        return out_emb

    def forward(self, xis, xls1, xls2):
        zis1 = self.image_encoder_1(xis)
        zis2 = self.image_encoder_2(xis)
        
        zls1 = self.text_encoder(xls1)
        zls2 = self.text_encoder(xls2)
                
        x = zis1.view(zis1.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        outputs = self.softmax(x)
        # outputs = x
        
        return zis1, zis2, zls1, zls2, outputs

    