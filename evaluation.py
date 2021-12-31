import torch
import torch.nn as nn
import numpy as np

class Test():
    def __init__(self,model:torch.nn.Module,
                 dataloader:torch.utils.data.dataloader,
                 criterion:torch.nn,
                 device,
                 top_k,
                 ):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.device = device
        self.top_k = top_k
    def hit(self,item,pred_items):
        if item in pred_item:
            return 1
        else:
            return 0

    def ndcg(self,item,pred_items):
        if item in pred_items:
            idx = pred_items.index(item)
            return np.reciprocal(np.log2(index+2))
        return 0

    def metrics(self):
        HR, NDCG = [], []
        device = self.device
        model = self.model
        top_k  = self.top_k
        for user, item, target in self.dataloader:
            user, item, target = user.to(device), item.to(device), target.to(device)
            pred = model(user,item)
            _,idx = torch.topk(pred,top_k)
            recommends = torch.take(item,idx).numpy().tolist()

            item = item[0].item()
            HR.append(self.hit(item,recommends))
            NDCG.append(self.ndcg(item,recommends))
        return np.mean(HR), np.mean(NDCG)
