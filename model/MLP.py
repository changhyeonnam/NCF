import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self,
                 num_users:int,
                 num_items:int,
                 layer=None,):
        super(MLP,self).__init__()

        if layer is None:
            layer = [16, 8]
        self.num_users = num_users
        self.num_items = num_items
        self.user_embedding = nn.Embedding(num_users,layer[0]//2)
        self.item_embedding = nn.Embedding(num_items,layer[0]//2)
        MLP_layers=[]
        for idx,num_factor in enumerate(layer):
            # ith MLP layer (layer[i],layer[i]//2) -> #(i+1)th MLP layer (layer[i+1],layer[i+1]//2)
            # ex) (32,16) -> (16,8) -> (8,4)
            MLP_layers.append(nn.Linear(num_factor,num_factor//2))
            MLP_layers.append(nn.ReLU())
        # unpacking layers in to torch.nn.Sequential
        self.MLP_model = nn.Sequential(*MLP_layers)
        self.predict_layer = nn.Linear(layer[len(layer)-1]//2,1)

    def forward(self,user,item):
        embed_user = self.user_embedding(user)
        embed_item = self.item_embedding(item)
        # dim=-1 means torch.cat(((2,3),(2,3)),-1) => (2,6) ((4,3))
        embed_input = torch.cat((embed_user,embed_item),dim=-1)
        output = torch.sigmoid(self.predict_layer(self.MLP_model(embed_input)))

        return output


