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
from parser import args

# check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

# print GPU information
if torch.cuda.is_available():
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())



# print selected model
print(f'model name: {args.model}')

# argparse doesn't supprot boolean type
use_downlaod = True if args.download=='True' else False
use_pretrain = True if args.use_pretrain=='True' else False
save_model = True if args.save_model == 'True' else False

pretrain_dir = 'pretrain'
if not os.path.isdir(pretrain_dir):
    os.makedirs(pretrain_dir)

# root path for dataset
root_path='dataset'+args.file_size

print(f'file size:{args.file_size}')

# load dataframe
dataset = Download(root=root_path,file_size=args.file_size,download=use_downlaod)
total_dataframe, train_dataframe, test_dataframe = dataset.split_train_test()

# make torch.utils.data.Data object
train_set = MovieLens(df=train_dataframe,total_df=total_dataframe,ng_ratio=4)
test_set = MovieLens(df=test_dataframe,total_df=total_dataframe,ng_ratio=99)

# get number of unique userID, unique  movieID
max_num_users,max_num_items = total_dataframe['userId'].max()+1, total_dataframe['movieId'].max()+1

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
    model = MLP(num_users=max_num_users,
                num_items=max_num_items,
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
    if use_pretrain:
        GMF_dir = os.path.join(pretrain_dir,'GMF.pth')
        MLP_dir = os.path.join(pretrain_dir,'MLP.pth')
        pretrained_GMF = torch.load(GMF_dir)
        pretrained_MLP = torch.load(MLP_dir)

        for param in pretrained_GMF.parameters():
            param.requires_grad = False

        for param in pretrained_MLP.parameters():
            param.requires_grad = False
    else:
        pretrained_GMF = None
        pretrained_MLP = None

    model = NeuMF(num_users=args.batch*max_num_users,
                  num_items=args.batch*max_num_items,
                  num_factor=args.factor,
                  use_pretrain=use_pretrain,
                  layer=args.layer,
                  pretrained_GMF=pretrained_GMF,
                  pretrained_MLP=pretrained_MLP)
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

    if save_model:
        pretrain_model_dir = os.path.join(pretrain_dir,args.model+'.pth')
        torch.save(model,pretrain_model_dir)


    end = time.time()
    print(f'training time:{end-start:.5f}')
    HR,NDCG = metrics(model,test_loader=dataloader_test,top_k=args.topk,device=device)
    print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

