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
                 ):
        super(NeuMF,self).__init__()
        self.use_pretrain = use_pretrain
        # layer for MLP
        if layer is None:
            layer = [32,16, 8]

        if use_pretrain == True:
            # not implemented
            pass
        else:
            self.GMF=GMF(num_users,num_items,num_factor,use_pretrain=use_pretrain,notuseNeuMF=False)
            self.MLP=MLP(num_users,num_items,num_factor,layer,use_pretrain=use_pretrain,notuseNeuMF=False)
        self.predict_layer=nn.Sequential(nn.Linear(num_factor*2,1),nn.Sigmoid())

        #self._init_weight()

    def _init_weight(self):

        nn.init.kaiming_uniform_(self.predict_layer.weight,
                                 a=0, nonlinearity='sigmoid')
        # self.predict_layer.bias.data.zero_()

    def forward(self,user,item):
       # print(user.shape)
       # print(item.shape)
       # print(self.GMF(user,item).shape)
       # print(self.MLP(user,item).shape)
       # print(f'NeMF <- GMF {self.GMF(user,item).shape}, MLP{self.MLP(user,item).shape}')
        before_last_layer_output = torch.cat((self.GMF(user,item),self.MLP(user,item)),dim=-1)
        # print(f'NeuMF before_last_layer_output:{before_last_layer_output.shape}')

        output = self.predict_layer(before_last_layer_output)
        # print(f'NeuMF output shape:{output.view(-1).shape}, not view:{output.shape}')

        return output.view(-1)
