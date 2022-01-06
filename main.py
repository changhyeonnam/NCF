import pandas as pd
import torch
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import MovieLens,Download
from model.MLP import MLP
from model.GMF import GMF
from model.NeuMF import NeuMF
from train import Train
from evaluation import metrics
import os
import numpy as np
import time
from torchsummary import summary

# print device info
device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')
print('device:',device)

# print gpu info
if device == 'cuda':
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())

parser=argparse.ArgumentParser(description="Run selected model")
parser.add_argument('-e','--epoch',type=int,default=1,help="Number of epochs")
parser.add_argument('-b','--batch',type=int,default=256,help="Batch size")
parser.add_argument('-l','--layer', nargs='+',type=list,default=[64,32,16],help='MLP layer factor list')
parser.add_argument('-f','--factor',type=int,default=8,help='choose number of predictive factor')
parser.add_argument('-m','--model',type=str,default='NeuMF',help='select among the following model,[MLP, GMF, NeuMF]')
parser.add_argument('-lr', '--lr', default=1e-3, type=float,help='learning rate for optimizer')
parser.add_argument('-dl','--download',type=str,default='True',help='Download or not')
parser.add_argument('-pr','--use_pretrain',type=str,default='False',help='use pretrained model or not')
parser.add_argument('-k','--topk',type=int,default=10,help='choose top@k for NDCG@k, HR@k')
parser.add_argument('-fi','--file_size',type=str,default='100k',help='choose file size, [100k,1m,10m,20m]')
args = parser.parse_args()

# print selected model
print(f'model name: {args.model}')

# argparse doesn't supprot boolean type
use_downlaod = True if args.download=='True' else False
use_pretrain = True if args.use_pretrain=='True' else False

# root path for dataset
root_path='dataset'+args.file_size

print(args.file_size)
# load dataframe
dataset = Download(root=root_path,file_size=args.file_size,download=use_downlaod)
total_dataframe, train_dataframe, test_dataframe = dataset.split_train_test()

# make torch.utils.data.Data object
train_set = MovieLens(df=train_dataframe,total_df=total_dataframe,ng_ratio=4)
test_set = MovieLens(df=test_dataframe,total_df=total_dataframe,ng_ratio=99)

# get number of unique userID, unique  movieID
max_num_users,max_num_items = total_dataframe['userId'].nunique(), total_dataframe['movieId'].nunique()

print('data loaded!')

# dataloader for train_dataset
dataloader_train= DataLoader(dataset=train_set,
                        batch_size=args.batch,
                        shuffle=True,
                        num_workers=0,
                        )

# dataloader for test_dataset
dataloader_test = DataLoader(dataset=test_set,
                             batch_size=100,
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
                use_pretrain=use_pretrain,
                use_NeuMF=False)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

elif args.model=='GMF':
    model = GMF(num_users=args.batch*max_num_users,
                num_items=args.batch*max_num_items,
                num_factor=args.factor,
                use_pretrain=use_pretrain,
                use_NeuMF=False)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

elif args.model=='NeuMF':
    model = NeuMF(num_users=args.batch*max_num_users,
                  num_items=args.batch*max_num_items,
                  num_factor=args.factor,
                  use_pretrain=use_pretrain,
                  layer=args.layer)
    if not use_pretrain :
        optimizer =  optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(model.parameters(),lr=args.lr)

# for parallel GPU
# if torch.cuda.device_count() >1:
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
    
    # measuring time
    start = time.time()

    train.train()
    pretrained_model_path ='pretrain'
    end = time.time()
    print(f'training time:{end-start:.5f}')
    HR,NDCG = metrics(model,test_loader=dataloader_test,top_k=args.topk,device=device)
    print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

