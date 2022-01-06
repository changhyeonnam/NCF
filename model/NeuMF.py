import torch.nn as nn
import torch
from model.GMF import GMF
from model.MLP import MLP

class NeuMF(nn.Module):
    def __init__(self,
                 num_users: int,
                 num_items: int,
                 num_factor: int = 8,
                 use_pretrain: bool = False,
                 layer=None, # layer for MLP
                 pretrained_GMF =None,
                 pretrained_MLP =None
                 ):
        super(NeuMF,self).__init__()
        self.use_pretrain = use_pretrain

        self.pretrained_GMF = pretrained_GMF
        self.pretrained_MLP = pretrained_MLP

        # layer for MLP
        if layer is None:
            layer = [64,32, 16]

        if self.use_pretrain:
            self._load_pretrain_model()
        else:
            self.GMF=GMF(num_users,num_items,num_factor,use_pretrain=use_pretrain,use_NeuMF=True)
            self.MLP=MLP(num_users,num_items,num_factor,layer,use_pretrain=use_pretrain,use_NeuMF=True)

        self.predict_layer= nn.Linear(num_factor*2,1)
        self.Sigmoid = nn.Sigmoid()

        if not self.use_pretrain:
            nn.init.normal_(self.predict_layer.weight,std=1e-2)

    def _load_pretrain_model(self):
        predict_weight = torch.cat([
            self.pretrained_GMF.predict_layer.weight,
            self.pretrained_MLP.predict_layer.weight], dim=1)
        precit_bias = self.pretrained_GMF.predict_layer.bias + \
                      self.pretrained_MLP.predict_layer.bias
        self.predict_layer.weight.data.copy_(0.5 * predict_weight)
        self.predict_layer.bias.data.copy_(0.5 * precit_bias)

    def forward(self,user,item):
        before_last_layer_output = torch.cat((self.GMF(user,item),self.MLP(user,item)),dim=-1)
        output = self.predict_layer(before_last_layer_output)
        output = self.Sigmoid(output)
        return output.view(-1)