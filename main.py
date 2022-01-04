import torch
import argparse
import inspect
from torch.utils.data import DataLoader
import torch.optim as optim
# import matplotlib.pyplot as plt
from utils import MovieLens
from model.MLP import MLP
from model.GMF import GMF
from model.NeuMF import NeuMF
from train import Train
from evaluation import metrics
import os
import numpy as np

device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')
if device == 'cuda':
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())


parser=argparse.ArgumentParser(description="Run selected model")
parser.add_argument('-e','--epoch',type=int,default=1,help="Number of epochs")
parser.add_argument('-b','--batch',type=int,default=256,help="Batch size")
parser.add_argument('-tb','--test_batch',type=int,default=100,help="test Batch size")
parser.add_argument('-l','--layer',type=None,default=[32,16,8],help='MLP layer factor list')
parser.add_argument('-f','--factor',type=int,default=8,help='choose number of predictive factor')
parser.add_argument('-m','--model',type=str,default='NeuMF',help='select among the following model,[MLP, GMF, NeuMF]')
parser.add_argument('-s','--size',type=str,default='small',help='Size of File')
parser.add_argument('-lr','--lr',type=float,default=1e-3,help='learning rate')
parser.add_argument('-dl','--download',type=str,default='False',help='Download or not')
parser.add_argument('-p','--use_pretrain',type=str,default='False',help='use pretrained model or not')
parser.add_argument('-k','--topk',type=int,default=10,help='choose top@k for NDCG@k, HR@k')
args = parser.parse_args()

# print selected model
print(f'model name: {args.model}')

# argparse doesn't supprot boolean type
if args.download=='True':
    download = True
else:
    download = False
if args.use_pretrain=='True':
    use_pretrain=True
else:
    use_pretrain=False

# root path for dataset
root_path='data'

# load train,test dataset
train_dataset = MovieLens(root=root_path,train=True,ng_ratio=4)
test_dataset = MovieLens(root=root_path,train=False,ng_ratio=99)
print('data loaded!')

# load number of nunique user Id, item Id
max_num_users,max_num_items = train_dataset.get_numberof_users_items()

# dataloader for train_dataset
dataloader_train= DataLoader(dataset=train_dataset,
                        batch_size=args.batch,
                        shuffle=True,
                        num_workers=0,
                        )

# dataloader for test_dataset
dataloader_test = DataLoader(dataset=test_dataset,
                             batch_size=args.test_batch,
                             shuffle=False,
                             num_workers=0,
                             drop_last=True
                             )
# select model among these models ['MLP', 'GMF', 'NeuMF']
if args.model=='MLP':
    model = MLP(num_users=args.batch*max_num_users,
                num_items=args.batch*max_num_items,
                num_factor=args.factor,
                layer=args.layer,
                use_pretrain=args.use_pretrain)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

elif args.model=='GMF':
    model = GMF(num_users=args.batch*max_num_users,
                num_items=args.batch*max_num_items,
                num_factor=args.factor,
                use_pretrain=args.use_pretrain)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

elif args.model=='NeuMF':
    model = NeuMF(num_users=args.batch*max_num_users,
                  num_items=args.batch*max_num_items,
                  num_factor=args.factor,
                  use_pretrain=use_pretrain,
                  layer=args.layer)
    if use_pretrain :
        optimizer =  optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(model.parameters(),lr=args.lr)

#if torch.cuda.device_count() >1:
#    print("Multi gpu", torch.cuda.device_count())
#    model = torch.nn.DataParallel(model)

model.to(device)

# objective function is log loss (Cross-entropy loss)
criterion = torch.nn.BCELoss()

if __name__=='__main__' :



    train = Train(model=model,
                  optimizer=optimizer,
                  criterion=criterion,
                  epochs=args.epoch,
                  test_obj=dataloader_test,
                  dataloader=dataloader_train,
                  device=device,
                  print_cost=True,)
    train.train()
    pretrained_model_path ='pretrain'
    # if not use_pretrain:
    #     if not os.path.exists(pretrained_model_path):
    #         os.makedirs(pretrained_model_path)
    #     model_save_path = os.path.join(pretrained_model_path,args.model+'.pth')
    #     if args.model=='MLP' or 'GMF' :
    #         torch.save(model.state_dict(),model_save_path)


    HR,NDCG = metrics(model,test_loader=dataloader_test,top_k=args.topk,device=device)
    print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

