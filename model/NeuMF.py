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
            layer = [64,32, 16]

        if use_pretrain == True:
            # not implemented
            pass
        else:
<<<<<<< HEAD
            self.GMF=GMF(num_users,num_items,num_factor,use_pretrain=use_pretrain,use_NeuMF=True)
            self.MLP=MLP(num_users,num_items,num_factor,layer,use_pretrain=use_pretrain,use_NeuMF=True)
=======
            self.GMF=GMF(num_users,num_items,num_factor,use_pretrain=use_pretrain,notuseNeuMF=False)
            self.MLP=MLP(num_users,num_items,num_factor,layer,use_pretrain=use_pretrain,notuseNeuMF=False)
>>>>>>> 400fe117ba803f42cd8620893e3ee4638b74da36
        self.predict_layer=nn.Sequential(nn.Linear(num_factor*2,1),nn.Sigmoid())

    def forward(self,user,item):
<<<<<<< HEAD
=======
       # print(user.shape)
       # print(item.shape)
       # print(self.GMF(user,item).shape)
       # print(self.MLP(user,item).shape)
       # print(f'NeMF <- GMF {self.GMF(user,item).shape}, MLP{self.MLP(user,item).shape}')
>>>>>>> 400fe117ba803f42cd8620893e3ee4638b74da36
        before_last_layer_output = torch.cat((self.GMF(user,item),self.MLP(user,item)),dim=-1)
        output = self.predict_layer(before_last_layer_output)

        return output.view(-1)