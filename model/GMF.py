import torch.nn as nn
import torch

class GMF(nn.Module):
    def __init__(self,
                 num_users:int,
                 num_items:int,
                 num_factor:int=8,
                 use_pretrain: bool = False,
                 ):
        super(GMF,self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.use_pretrain = use_pretrain

        self.user_embedding = nn.Embedding(num_users,num_factor)
        self.item_embedding = nn.Embedding(num_items,num_factor)
        self.predict_layer = nn.Sequential(nn.Linear(num_factor,1),nn.Sigmoid())
        self._init_weight()

    def _init_weight(self):
        if self.use_pretrain == False :
            nn.init.normal_(self.user_embedding.weight,std=1e-2)
            nn.init.normal_(self.item_embedding.weight,std=1e-2)
        for layer in self.predict_layer:
            if isinstance(layer,nn.Linear):
                nn.init.normal_(layer.weight,std=1e-2)
                layer.bias.data.zero_()

    def forward(self,users,items):
        embedding_elementwise = self.user_embedding(users) * self.item_embedding(items)
        output = self.predict_layer(embedding_elementwise)
        return output

    def __call__(self,*args):
        return self.forward(*args)

