import torch.nn as nn
import torch
# import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self,
                 num_users:int,
                 num_items:int,
                 num_factor:int=8,
                 layer=None,
                 use_pretrain: bool = False,
                 use_NeuMF:bool = False,
                 pretrained_MLP=None
                 ):
        super(MLP,self).__init__()

        if layer is None:
            layer = [64,32,16]

        self.pretrained_model = pretrained_MLP
        self.num_users = num_users
        self.num_items = num_items
        self.use_pretrian = use_pretrain
        self.user_embedding = nn.Embedding(num_users,layer[0]//2)
        self.item_embedding = nn.Embedding(num_items,layer[0]//2)
        self.use_NeuMF = use_NeuMF
        MLP_layers=[]

        for idx,factor in enumerate(layer):
            # ith MLP layer (layer[i],layer[i]//2) -> #(i+1)th MLP layer (layer[i+1],layer[i+1]//2)
            # ex) (64,32) -> (32,16) -> (16,8)
            MLP_layers.append(nn.Linear(factor, factor // 2))
            MLP_layers.append(nn.ReLU())

        # unpacking layers in to torch.nn.Sequential
        self.MLP_model = nn.Sequential(*MLP_layers)

        if not use_NeuMF:
            self.predict_layer = nn.Sequential(nn.Linear(num_factor, 1), nn.Sigmoid())
        if use_pretrain:
            self._load_pretrained_model()
        else:
            self._init_weight()

    def _init_weight(self):
        if not self.use_pretrian:
            nn.init.normal_(self.user_embedding.weight,std=1e-2)
            nn.init.normal_(self.item_embedding.weight,std=1e-2)
            for layer in self.MLP_model:
                if isinstance(layer,nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)

    def _load_pretrained_model(self):
        self.user_embedding.weight.data.copy_(
            self.pretrained_MLP.user_embedding.weight)
        self.item_embedding.weight.data.copy_(
            self.pretrained_MLP.item_embedding.weight)
        for (layer, pretrained_layer) in zip(self.MLP_model,self.pretrained_model):
            if isinstance(layer,nn.Linear) and isinstance(pretrained_layer,nn.Linear):
                layer.weight.data.copy_(pretrained_layer.weight)
                layer.bias.data.copy_(pretrained_layer.bias)

    def forward(self,user,item):
        embed_user = self.user_embedding(user)
        embed_item = self.item_embedding(item)
        embed_input = torch.cat((embed_user,embed_item),dim=-1)
        output = self.MLP_model(embed_input)
        if not self.use_NeuMF:
            output = self.predict_layer(output)
            output = output.view(-1)
        return output

    def __call__(self,*args):
        return self.forward(*args)