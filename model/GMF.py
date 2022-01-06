import torch.nn as nn
import torch

class GMF(nn.Module):
    def __init__(self,
                 num_users:int,
                 num_items:int,
                 num_factor:int=8,
                 use_pretrain: bool = False,
                 use_NeuMF:bool = False,
                 pretrained_GMF=None
                 ):
        super(GMF,self).__init__()

        self.pretrained_GMF = pretrained_GMF
        self.num_users = num_users
        self.num_items = num_items
        self.use_pretrain = use_pretrain
        self.use_NeuMF = use_NeuMF

        self.pretrained_GMF = pretrained_GMF

        self.user_embedding = nn.Embedding(num_users,num_factor)
        self.item_embedding = nn.Embedding(num_items,num_factor)
        if not self.use_NeuMF:
            self.predict_layer = nn.Linear(num_factor,1)
            self.Sigmoid = nn.Sigmoid()
        if use_pretrain:
            self._load_pretrained_model()
        else:
            self._init_weight()

    def _init_weight(self):
        if not self.use_pretrain:
            nn.init.normal_(self.user_embedding.weight,std=1e-2)
            nn.init.normal_(self.item_embedding.weight,std=1e-2)
        if not self.use_NeuMF:
            nn.init.normal_(self.predict_layer.weight,std=1e-2)

    def _load_pretrained_model(self):
        self.user_embedding.weight.data.copy_(
            self.pretrained_GMF.user_embedding.weight)
        self.item_embedding.weight.data.copy_(
            self.pretrained_GMF.item_embedding.weight)


    def forward(self,users,items):
        embedding_elementwise = self.user_embedding(users) * self.item_embedding(items)
        output = embedding_elementwise
        if not self.use_NeuMF:
            output = self.predict_layer(embedding_elementwise)
            output = self.Sigmoid(output)
            output = output.view(-1)
        else:
            output = self.predict_layer(output)

        return output

