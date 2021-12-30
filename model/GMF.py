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

    def forward(self,users,items):
        result = torch.bmm(self.user_embedding(users),torch.transpose(self.item_embedding(items),1,2))
        return result

    def __call__(self,*args):
        return self.forward(*args)

