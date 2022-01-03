import torch
from evaluation import Test

class Train():
    def __init__(self,model:torch.nn.Module
                 ,optimizer:torch.optim,
                 epochs:int,
                 dataloader:torch.utils.data.dataloader,
                 criterion:torch.nn,
                 device='cpu',
                 print_cost=True):
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.dataloader = dataloader
        self.criterion = criterion
        self.device = device
        self.print_cost = print_cost

    def train(self):
        model = self.model
        optimizer = self.optimizer
        total_epochs = self.epochs
        dataloader = self.dataloader
        criterion = self.criterion
        total_batch = len(dataloader)
        loss = []
        device = self.device


        test_dataset = MovieLens(root=root_path, train=False, ng_ratio=99)
        test = Test(model=model,
                    criterion=criterion,
                    dataloader=dataloader_test,
                    device=device,
                    top_k=args.topk, )

        for epochs in range(0,total_epochs):
            avg_cost = 0
            for user,item,target in dataloader:
                user,item,target=user.to(device),item.to(device),target.to(device)
                optimizer.zero_grad()
                val = model(user, item)
                pred = torch.flatten(val,start_dim=1).to(device)
                cost = criterion(pred,target)
                cost.backward()
                optimizer.step()
                avg_cost += cost.item() / total_batch
            if self.print_cost:
                print(f'Epoch:, {(epochs + 1):04}, {criterion._get_name()}=, {avg_cost:.9f}')
                HR, NDCG = test.metrics()
                print(f'NDCG:{NDCG}, HR:{HR}')

            loss.append(avg_cost)

        if self.print_cost:
            print('Learning finished')
        return loss