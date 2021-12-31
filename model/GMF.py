import torch.nn as nn
import torch

class GMF(nn.Module):
    def __init__(self,
                 num_users:int,
                 num_items:int,
                 num_factor:int=8):
        super(GMF,self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.user_embedding = nn.Embedding(num_users,num_factor)
        self.item_embedding = nn.Embedding(num_items,num_factor)
        self.predict_layer = nn.Sequential(nn.Linear(num_factor,1),nn.Sigmoid())

    def forward(self,users,items):
        embedding_elementwise = self.user_embedding(users) * self.item_embedding(items)
        output = self.predict_layer(embedding_elementwise)
        return output

    def __call__(self,*args):
        return self.forward(*args)

