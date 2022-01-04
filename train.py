import torch
import numpy as np
from evaluation import metrics
class Train():
    def __init__(self,model:torch.nn.Module
                 ,optimizer:torch.optim,
                 epochs:int,
                 dataloader:torch.utils.data.dataloader,
                 criterion:torch.nn,
                 test_obj,
                 device='cpu',
                 print_cost=True):
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.dataloader = dataloader
        self.criterion = criterion
        self.device = device
        self.print_cost = print_cost
        self.test = test_obj

    def train(self):
        model = self.model
        optimizer = self.optimizer
        total_epochs = self.epochs
        dataloader = self.dataloader
        criterion = self.criterion
        total_batch = len(dataloader)
        loss = []
        device = self.device
        test = self.test

        for epochs in range(0,total_epochs):
            avg_cost = 0
            for user,item,target in dataloader:
                user,item,target=user.to(device),item.to(device),target.float().to(device)
                optimizer.zero_grad()
                #print(f'user:{user.shape}, item:{item.shape}')
                pred = model(user, item)
               # print(f'target:{target.shape}')
               # print(f'pred:{pred.shape}')
                cost = criterion(pred,target)
                cost.backward()
                optimizer.step()
                avg_cost += cost.item() / total_batch
            if self.print_cost:
                print(f'Epoch: {(epochs + 1):04}, {criterion._get_name()}= {avg_cost:.9f}')
                HR, NDCG = metrics(model,test,10,device)
                print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

            loss.append(avg_cost)

        if self.print_cost:
            print('Learning finished')
        return loss
