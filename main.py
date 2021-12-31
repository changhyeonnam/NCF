import torch
import argparse
import inspect
from torch.utils.data import DataLoader
import torch.optim as optim
# import matplotlib.pyplot as plt
from utils import MovieLens
from model.MLP import MLP
from model.GMF import GMF
from train import Train
from evaluation import Test
import os

device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')
if device == 'cuda':
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())


parser=argparse.ArgumentParser(description="Run selected model")
parser.add_argument('-e','--epoch',type=int,default=1,help="Number of epochs")
parser.add_argument('-b','--batch',type=int,default=32,help="Batch size")
parser.add_argument('-l','--layer',type=None,default=[32,16,8],help='MLP layer factor list')
parser.add_argument('-m','--model',type=str,default='GMF',help='MLP, GMF, NeuMF')
parser.add_argument('-s','--size',type=str,default='small',help='Size of File')
parser.add_argument('-lr','--lr',type=float,default=1e-3,help='learning rate')
parser.add_argument('-dl','--download',type=str,default='False',help='Download or not')
args = parser.parse_args()

# print selected model
print(f'model name: {args.model}')

# argparse doesn't supprot boolean type
if args.download=='True':
    download = True
else:
    download = False

# root path for dataset
root_path ='dataset'
# load train,test dataset
train_dataset = MovieLens(root=root_path,file_size=args.size,train=True,download=download)
test_dataset = MovieLens(root=root_path,file_size=args.size,train=False,download=False)

# load number of nunique user Id, item Id
max_num_users,max_num_items = train_dataset.get_numberof_users_items()

# dataloader for train_dataset
dataloader_train= DataLoader(dataset=train_dataset,
                        batch_size=32,
                        shuffle=True,
                        num_workers=0,
                        )
# dataloader for test_dataset
dataloader_test = DataLoader(dataset=test_dataset,
                             batch_size=32,
                             shuffle=True,
                             num_workers=0,
                             drop_last=True
                             )
if args.model=='MLP':
    model = MLP(num_usermax_num_users,num_items=max_num_items,layer=args.layer)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

elif args.model=='GMF':
    model = GMF(num_users=args.batch*max_num_users, num_items=args.batch*max_num_items)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

elif args.model=='NCF':
    model = NCF()
    optimizer = optim.SGD(model.parameters(),lr=args.lr)

#if torch.cuda.device_count() >1:
#    print("Multi gpu", torch.cuda.device_count())
#    model = torch.nn.DataParallel(model)
model.to(device)

optimizer = optim.Adam(model.parameters(),lr=args.lr)
criterion = torch.nn.BCELoss()

if __name__=='__main__' :
    train = Train(model=model,
                  optimizer=optimizer,
                  criterion=criterion,
                  epochs=args.epoch,
                  dataloader=dataloader_train,
                  device=device,
                  print_cost=True)
    train.train()
    pretrained_model_path ='pretrain'
    # if not os.path.exists(pretrained_model_path):
    #     os.makedirs(pretrained_model_path)
    # model_save_path = os.path.join(pretrained_model_path,args.model+'.pth')
    # if args.model=='MLP' or 'GMF' :
    #     torch.save(model.state_dict(),model_save_path)

    test = Test(model=model,
                criterion=criterion,
                dataloader=dataloader_test,
                device=device,
                top_k=10,)
    HR,NDCG = test.metrics()
    print(f'NDCG:{NDCG}, HR:{HR}')

