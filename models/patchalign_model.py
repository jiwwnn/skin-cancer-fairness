import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig, AutoModel
import yaml

# torch.backends.cudnn.deterministic = False
# torch.backends.cudnn.benchmark = True
config = yaml.load(open("patchalign_config.yaml", "r"), Loader=yaml.FullLoader)
    
class ModelPatchAlign(torch.nn.Module):
    def __init__(self, bert_base_model, out_dim, freeze_layers, do_lower_case) :
        super(ModelPatchAlign, self).__init__()
            
        self.feature_extractor = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.vit_features = nn.Sequential(*list(self.feature_extractor.children())[:-1])
            
        self.a = 197
        self.b = 7
        
        self.activation = torch.nn.ReLU()
        # branch 1
        self.branch_1 = nn.Sequential(
            nn.Linear(self.feature_extractor.config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128, config['classes']),
        )
        # branch 2
        self.branch_2 = nn.Linear(self.feature_extractor.config.hidden_size, config['fitz_classes'])

        self.mask = nn.Sequential(
        # nn.Linear(1000, 128),
        nn.Linear(197 * 768, 128),
        nn.ReLU(),
        nn.Linear(128, self.a * self.b),
        nn.Sigmoid()
        )
        # self.mask = BinaryMatrixGenerator(197 * 768, 128, a = self.a, b = self.b)
        
        self.bert_model = self._get_bert_basemodel(bert_base_model,freeze_layers)
        self.bert_l1 = nn.Linear(768, out_dim) #768 is the size of the BERT embbedings
    
        
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


    def text_encoder(self, encoded_inputs):
        """
        Obter os inputs e em seguida extrair os hidden layers e fazer a media de todos os tokens
        Fontes:
        - https://github.com/BramVanroy/bert-for-inference/blob/master/introduction-to-bert.ipynb
        - Nils Reimers, Iryna Gurevych. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
        https://www.sbert.net
        """
        outputs = self.bert_model(**encoded_inputs) 
        
        with torch.no_grad():
            sentence_embeddings = self.mean_pooling(outputs, encoded_inputs['attention_mask']).float()
            x = self.bert_l1(sentence_embeddings)
            out_emb = F.relu(x)
        return out_emb
        
        
    def forward(self, x, encoded_inputs):
        feature_map = self.feature_extractor(x)  # (bs, bottle_neck)
        feature_map = feature_map.last_hidden_state
        
        x = feature_map
        x = x.view(x.size(0), -1)
        # print(x.shape)
        out_mask = self.mask(x)
        # print("Out mask shape:", out_mask.shape) 
        
        x = torch.sum(out_mask.view(-1, 197, 7), dim=2).unsqueeze(dim=2)
        x = torch.cat([x,]*768,dim=2)

        out_1 = self.branch_1(feature_map * x)
        out_2 = self.branch_2(feature_map * x)
        #out_4 = self.project_head(feature_map)
        # detach feature map and pass though branch 2 again
        feature_map_detach = feature_map.detach()
        out_3 = self.branch_2(feature_map_detach)
        #print(feature_map.shape)
    
        text_emb = self.text_encoder(encoded_inputs)
        text_emb = torch.cat([text_emb.unsqueeze(0)]*32)

        return [torch.mean(out_1, dim=1), torch.mean(out_2, dim=1), torch.mean(out_3, dim=1), out_mask.view(-1, 197, 7), feature_map*x, text_emb]
        # return [out_1, out_2, out_3]
 
# class BinaryMatrixGenerator(nn.Module):
#     def __init__(self, input_size, hidden_size, a , b):
#         super(BinaryMatrixGenerator, self).__init__()
#         self.layer1 = nn.Linear(input_size, hidden_size)
#         self.a = a
#         self.b = b

#         self.layer2 = nn.Linear(hidden_size, self.a * self.b)  # Adjust the output layer size
#         self.sigmoid = nn.Sigmoid()
        
#     def forward(self, x):
#         x = x.view(x.size(0), -1)
#         x = torch.relu(self.layer1(x))
#         x = torch.relu(self.layer2(x))
#         M = torch.round(torch.nn.functional.normalize(x.view(-1, self.a, self.b)))+1e-12
#         return M
    
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import timm  
# from transformers import AutoModel
# import yaml

# config = yaml.load(open("patchalign_config.yaml", "r"), Loader=yaml.FullLoader)

# class ModelPatchAlign(torch.nn.Module):
#     def __init__(self, bert_base_model, out_dim, freeze_layers, do_lower_case):
#         super(ModelPatchAlign, self).__init__()
        
#         # Initialize ViT from timm
#         self.feature_extractor = timm.create_model('vit_base_patch16_224.mae', pretrained=True)
#         self.hidden_size = 768  # vit_base_patch16_224.mae's hidden size
        
#         self.a = 197
#         self.b = 7

#         self.activation = nn.ReLU()

#         # Branch 1
#         self.branch_1 = nn.Sequential(
#             nn.Linear(self.hidden_size, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, config['classes']),
#         )

#         # Branch 2
#         self.branch_2 = nn.Linear(self.hidden_size, config['fitz_classes'])

#         # Mask generation
#         self.mask = nn.Sequential(
#             nn.Linear(197 * self.hidden_size, 128),
#             nn.ReLU(),
#             nn.Linear(128, self.a * self.b),
#             nn.Sigmoid()
#         )

#         # Text encoder (BERT)
#         self.bert_model = self._get_bert_basemodel(bert_base_model, freeze_layers)
#         self.bert_l1 = nn.Linear(768, out_dim)

#     def _get_bert_basemodel(self, bert_model_name, freeze_layers):
#         try:
#             model = AutoModel.from_pretrained(bert_model_name)
#             print(f"Loaded BERT model: {bert_model_name}")
#         except Exception as e:
#             raise ValueError(f"Failed to load BERT model: {e}")

#         if freeze_layers is not None:
#             for layer_idx in freeze_layers:
#                 for param in model.encoder.layer[layer_idx].parameters():
#                     param.requires_grad = False

#         return model

#     def mean_pooling(self, model_output, attention_mask):
#         """
#         Perform mean pooling on the token embeddings.
#         """
#         token_embeddings = model_output.last_hidden_state
#         input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
#         sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
#         sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
#         return sum_embeddings / sum_mask

#     def text_encoder(self, encoded_inputs):
#         """
#         Process text inputs with BERT model and project to output space.
#         """
#         bert_output = self.bert_model(**encoded_inputs)
#         sentence_embeddings = self.mean_pooling(bert_output, encoded_inputs['attention_mask'])
#         projected = self.bert_l1(sentence_embeddings)
#         return F.relu(projected)

#     def forward(self, x, encoded_inputs):
#         # Process images through ViT (timm model)
#         feature_map = self.feature_extractor.forward_features(x)  # Use `forward_features` for timm ViT models
#         feature_map = feature_map.view(-1, 197, self.hidden_size)  # Reshape to (batch_size, 197, hidden_size)

#         # Generate mask
#         x_flat = feature_map.view(feature_map.size(0), -1)
#         out_mask = self.mask(x_flat)
#         mask = out_mask.view(-1, 197, 7).sum(dim=2).unsqueeze(-1)
#         mask_expanded = mask.repeat(1, 1, self.hidden_size)

#         # Apply mask to feature map
#         masked_features = feature_map * mask_expanded

#         # Pass through branches
#         out_1 = self.branch_1(masked_features.mean(dim=1))
#         out_2 = self.branch_2(masked_features.mean(dim=1))

#         # Detach features and recompute branch 2 for unmasked features
#         detached_features = feature_map.detach()
#         out_3 = self.branch_2(detached_features.mean(dim=1))

#         # Process text inputs
#         text_features = self.text_encoder(encoded_inputs)

#         return [
#             torch.mean(out_1, dim=1),
#             torch.mean(out_2, dim=1),
#             torch.mean(out_3, dim=1),
#             out_mask.view(-1, 197, 7),
#             feature_map * mask_expanded,
#             text_features,
#         ]









