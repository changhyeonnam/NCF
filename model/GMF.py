import torch.nn as nn
import torch

class GMF(nn.Module):
    def __init__(self,
                 num_users:int,
                 num_items:int,
                 num_factor:int=8,
                 use_pretrain: bool = False,
<<<<<<< HEAD
                 use_NeuMF:bool = False
=======
                 notuseNeuMF:bool = False
>>>>>>> 400fe117ba803f42cd8620893e3ee4638b74da36
                 ):
        super(GMF,self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.use_pretrain = use_pretrain
<<<<<<< HEAD
        self.use_NeuMF = use_NeuMF

        self.user_embedding = nn.Embedding(num_users,num_factor)
        self.item_embedding = nn.Embedding(num_items,num_factor)
        if not self.use_NeuMF:
=======
        self.notuseNeuMF = notuseNeuMF
        self.user_embedding = nn.Embedding(num_users,num_factor)
        self.item_embedding = nn.Embedding(num_items,num_factor)
        if notuseNeuMF:
>>>>>>> 400fe117ba803f42cd8620893e3ee4638b74da36
            self.predict_layer = nn.Sequential(nn.Linear(num_factor,1),nn.Sigmoid())
        self._init_weight()

    def _init_weight(self):
        if not self.use_pretrain:
            nn.init.normal_(self.user_embedding.weight,std=1e-2)
            nn.init.normal_(self.item_embedding.weight,std=1e-2)
        if not self.use_NeuMF:
            for layer in self.predict_layer:
                if isinstance(layer,nn.Linear):
                    nn.init.normal_(layer.weight,std=1e-2)

    def forward(self,users,items):
        embedding_elementwise = self.user_embedding(users) * self.item_embedding(items)
<<<<<<< HEAD
=======
        if self.notuseNeuMF:
            output = self.predict_layer(embedding_elementwise)
            output = output.view(-1)
        else:
            output = embedding_elementwise
        # print(f'GMF output shape:{output.view(-1).shape}, not view:{output.shape}')
>>>>>>> 400fe117ba803f42cd8620893e3ee4638b74da36

        if not self.use_NeuMF:
            output = self.predict_layer(embedding_elementwise)
            output = output.view(-1)
        else:
            output = embedding_elementwise
        return output

