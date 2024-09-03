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
import torchvision.models as models
import torch
from transformers import AutoModel

# Create the BertClassfier class
class ModelCLR(nn.Module):
    def __init__(self, res_base_model, bert_base_model, out_dim, freeze_layers, do_lower_case):
        super(ModelCLR, self).__init__()
        #init BERT
        self.bert_model1 = self._get_bert_basemodel(bert_base_model,freeze_layers)
        self.bert_model2 = self._get_bert_basemodel(bert_base_model,freeze_layers)
        
        # projection MLP for BERT model
        self.bert_l1_1 = nn.Linear(768, 768) #768 is the size of the BERT embbedings
        self.bert_l2_1 = nn.Linear(768, out_dim) #768 is the size of the BERT embbedings
        
        self.bert_l1_2 = nn.Linear(768, 768) 
        self.bert_l2_2 = nn.Linear(768, out_dim)    

        # init Resnet
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
                            "resnet50": models.resnet50(pretrained=False)}
        resnet1 = self._get_res_basemodel(res_base_model)
        resnet2 = self._get_res_basemodel(res_base_model)
        num_ftrs1 = resnet1.fc.in_features
        num_ftrs2 = resnet2.fc.in_features
        
        self.res_features1 = nn.Sequential(*list(resnet1.children())[:-1])
        self.res_features2 = nn.Sequential(*list(resnet2.children())[:-1])
                
        # projection MLP for ResNet Model
        self.res_l1_1 = nn.Linear(num_ftrs1, num_ftrs1)
        self.res_l2_1 = nn.Linear(num_ftrs1, out_dim)
        
        self.res_l1_2 = nn.Linear(num_ftrs2, num_ftrs2)
        self.res_l2_2 = nn.Linear(num_ftrs2, out_dim)        

    def _get_res_basemodel(self, res_model_name):
        try:
            res_model = self.resnet_dict[res_model_name]
            print("Image feature extractor:", res_model_name)
            return res_model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def _get_bert_basemodel(self, bert_model_name, freeze_layers):
        try:
            model = AutoModel.from_pretrained(bert_model_name)#, return_dict=True)
            print("Image feature extractor:", bert_model_name)
        except:
            raise ("Invalid model name. Check the config file and pass a BERT model from transformers lybrary")

        if freeze_layers is not None:
            for layer_idx in freeze_layers:
                for param in list(model.encoder.layer[layer_idx].parameters()):
                    param.requires_grad = False
        return model

    
    def mean_pooling(self, model_output, attention_mask):
        """
        Mean Pooling - Take attention mask into account for correct averaging
        Reference: https://www.sbert.net/docs/usage/computing_sentence_embeddings.html
        """
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def image_encoder1(self, xis):
        h = self.res_features1(xis)
        h = h.squeeze()

        x = self.res_l1_1(h)
        x = F.relu(x)
        x = self.res_l2_1(x)

        return h, x

    def text_encoder1(self, encoded_inputs):
        """
        Obter os inputs e em seguida extrair os hidden layers e fazer a media de todos os tokens
        Fontes:
        - https://github.com/BramVanroy/bert-for-inference/blob/master/introduction-to-bert.ipynb
        - Nils Reimers, Iryna Gurevych. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
        https://www.sbert.net
        """
        outputs = self.bert_model1(**encoded_inputs)
        
        with torch.no_grad():
            sentence_embeddings = self.mean_pooling(outputs, encoded_inputs['attention_mask']).float()
            x = self.bert_l1_1(sentence_embeddings)
            x = F.relu(x)
            out_emb = self.bert_l2_1(x)

        return out_emb
    
    def image_encoder2(self, xis):
        h = self.res_features2(xis)
        h = h.squeeze()
        x = self.res_l1_2(h)
        x = F.relu(x)
        x = self.res_l2_2(x)
        return h, x
    

    def text_encoder2(self, encoded_inputs):
        outputs = self.bert_model2(**encoded_inputs)
        with torch.no_grad():
            sentence_embeddings = self.mean_pooling(outputs, encoded_inputs['attention_mask']).float()
            x = self.bert_l1_2(sentence_embeddings)
            x = F.relu(x)
            out_emb = self.bert_l2_2(x)
        return out_emb

    def forward(self, xis, encoded_inputs1, encoded_inputs2):
        h1, zis1 = self.image_encoder1(xis)
        zls1 = self.text_encoder1(encoded_inputs1)
        h2, zis2 = self.image_encoder2(xis)
        zls2 = self.text_encoder2(encoded_inputs2)
        return zis1, zls1, zis2, zls2
    

    