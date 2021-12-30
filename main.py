import torch
import argparse
import inspect
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import MovieLens
from model.MLP import MLP
from model.GMF import GMF
from train import Train
from evaluation import Test

device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')
if device == 'cuda':
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())


parser=argparse.ArgumentParser(description="Run selected model")
parser.add_argument('-e','--epoch',type=int,default=1,help="Number of epochs")
parser.add_argument('-b','--batch',type=int,default=32,help="Batch size")
parser.add_argument('-l','--layer',type=None,default=[32,16,8],help='MLP layer factor list')
parser.add_argument('-m','--model',type=str,default='MLP',help='MLP, GMF, NeuMF')
parser.add_argument('-s','--size',type=str,default='small',help='Size of File')
parser.add_argument('-lr','--lr',type=float,default=1e-3,help='learning rate')
parser.add_argument('-dl','--download',type=str,default='False',help='Download or not')
args = parser.parse_args()

print(f'model name: {args.model}')
if args.download=='True':
    download = True
else:
    download = False

root_path ='dataset'
train_dataset = MovieLens(root=root_path,file_size=args.size,train=True,download=download)
test_dataset = MovieLens(root=root_path,file_size=args.size,train=False,download=False)
max_num_users,max_num_items = train_dataset.num_user,train_dataset.num_item
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
                             )
if args.model=='MLP':
    model = MLP(num_users=args.batch*max_num_users,num_items=args.batch*max_num_items,layer=args.layer)
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
    test = Test(model=model,
                criterion=criterion,
                dataloader=dataloader_test,
                device=device,
                print_cost=True)
    costs = train.train()
    plt.plot(range(0, args.epochs), costs)
    plt.xlabel('epoch')
    plt.ylabel('RMSE')
    now = time.localtime()
    time_now = f"{now.tm_hour:02d}:{now.tm_min:02d}:{now.tm_sec:02d} "
    fig_file = f"loss_curve_epochs_{args.epochs}_batch_{args.batch}_size_{args.size}_lr_{args.lr}_factor_{args.factor}.png"
    if os.path.isfile(fig_file):
        os.remove(fig_file)
    plt.savefig(fig_file)
    test.test()
