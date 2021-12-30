import zipfile
from torch.utils.data import Dataset
import torch
import os
import pandas  as pd
import numpy as np
from zipfile import ZipFile
import requests
from sklearn.model_selection  import train_test_split
import random
class MovieLens(Dataset):
    def __init__(self,
                 root:str='dataset',
                 file_size:str='small',
                 download:bool=False,
                 train:bool=True,
                 )->None:
        '''
        :param root: dir for download and train,test.
        :param file_size: large of small. if size if large then it will load(download) 20M dataset. if small then, it will load(download) 100K dataset.
        :param train: if true, then it will load data from train dataset dir else it will load test data
        :param download: if true, it will down load from url.
        '''
        super(MovieLens, self).__init__()
        self.root = root
        self.train = train
        if file_size =='large':
            self.file_dir='ml-latest'
        else:
            self.file_dir = 'ml-latest-'+file_size

        if download:
            self._download_movielens()
            self.df = self._read_ratings_csv()
            self.df = self._preprocess()
            self._train_test_split()
            print(len(self.df))
        else:
            # don't need to download. data from url already exists.
            self.df = self._read_ratings_csv()
            self.df = self._preprocess()
        self.num_user,self.num_item = self.get_numberof_users_items()
        self.data, self.target = self._load_data()


    def get_numberof_users_items(self) -> tuple:
        '''
        :return:  df["userId"].nunique(),  df["movieId"].nunique()
        '''
        df = self.df
        return df["userId"].nunique(), df["movieId"].nunique()

    def _load_data(self):
        '''
        if Train=True, load data from dir which is consist of train dataset
        else (Test case) load data from dir which is consist of test dataset
        :return:
        '''
        data_file = f"{'train' if self.train else 'test'}-dataset-movieLens/dataset/dataset.csv"
        data = pd.read_csv(os.path.join(self.root, data_file))
        label_file = f"{'train' if self.train else 'test'}-dataset-movieLens/label/label.csv"
        targets = pd.read_csv(os.path.join(self.root, label_file))
        print(f"loading {'train' if self.train else 'test'} file Complete!")
        return data, targets

    def __len__(self) -> int:
        '''
        get lenght of data
        :return: len(data)
        '''
        return len(self.data)


    def __getitem__(self, index):
        '''
        transform userId[index], item[inedx] to Tensor.
        and return to Datalaoder object.
        :param index: idex for dataset.
        :return: user,item,rating
        '''
        user = torch.LongTensor([self.data.userId.values[index]])
        item = torch.LongTensor([self.data.movieId.values[index]])
        target = torch.FloatTensor([self.target.rating.values[index]])
        return user,item,target

    def _download_movielens(self) -> None:
        '''
        Download dataset from url, if there is no root dir, then mkdir root dir.
        After downloading, it wil be extracted
        :return: None
        '''
        file = self.file_dir+'.zip'
        url = ("http://files.grouplens.org/datasets/movielens/"+file)
        req = requests.get(url, stream=True)
        print('Downloading MovieLens dataset')
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        with open(os.path.join(self.root, file), mode='wb') as fd:
            for chunk in req.iter_content(chunk_size=None):
                fd.write(chunk)
        with ZipFile(os.path.join(self.root, file), "r") as zip:
            # Extract files
            print("Extracting all the files now...")
            zip.extractall(path=self.root)
            print("Downloading Complete!")

    def _read_ratings_csv(self) -> pd.DataFrame:
        '''
        at first, check if file exists. if it doesn't then call _download().
        it will read ratings.csv, and transform to dataframe.
        it will drop columns=['timestamp'].
        :return:
        '''
        file = self.file_dir+'.zip'
        print("Reading file")
        zipfile = os.path.join(self.root,file)
        if not os.path.isfile(zipfile):
            self._download_movielens()
        fname = os.path.join(self.root, self.file_dir, 'ratings.csv')
        df = pd.read_csv(fname, sep=',').drop(columns=['timestamp'])
        print("Reading Complete!")
        return df

    def _preprocess(self) :
        '''
        sampling one positive feedback per four negative feedback
        :return: dataframe
        '''
        df = self.df.copy()
        users, items, labels = [], [], []
        user_item_set = set(zip(df['userId'], df['movieId']))
        all_movieIds = df['movieId'].unique()
        # negative feedback dataset ratio
        negative_ratio = 4
        for u, i in user_item_set:
            # positive instance
            users.append(u)
            items.append(i)
            labels.append(1)
            # negative instance
            for i in range(negative_ratio):
                # first item random choice
                negative_item = np.random.choice(all_movieIds)
                # check if item and user has interaction, if true then set new value from random
                while (u, negative_item) in user_item_set:
                    negative_item = np.random.choice(all_movieIds)
                users.append(u)
                items.append(negative_item)
                labels.append(0)
        df = pd.DataFrame(list(zip(users, items, labels)), columns=['userId', 'movieId', 'rating'])
        return df




    def _train_test_split(self) -> None:
        '''
        split dataset in to train and test dataset.
        and then for each dataset, split user&item(movie) and target(ratings).
        Save dataset in to corresponding directory.
        :return: None
        '''
        df = self.df.copy()
        print('Spliting Traingset & Testset')
        # Since MovieLens dataset is user-based dataset, I used Stratified k-fold.
        train, test,dummy_1,dummy_2 = train_test_split(df,df['userId'],test_size=0.2,stratify=df['userId']) # should add stratify
        train_dataset_dir = os.path.join(self.root, 'train-dataset-movieLens', 'dataset')
        train_label_dir = os.path.join(self.root, 'train-dataset-movieLens', 'label')
        test_dataset_dir = os.path.join(self.root, 'test-dataset-movieLens', 'dataset')
        test_label_dir = os.path.join(self.root, 'test-dataset-movieLens', 'label')
        dir_list = [train_dataset_dir, train_label_dir, test_dataset_dir, test_label_dir]
        for dir in dir_list:
            if not os.path.exists(dir):
                os.makedirs(dir)
        train_dataset, train_label = train.iloc[:, :-1], train.iloc[:, [-1]]
        test_dataset, test_label = test.iloc[:, :-1], test.iloc[:, [-1]]
        dataset = [train_dataset, train_label, test_dataset, test_label]
        data_dir_dict = {}
        for i in range(0, 4):
            data_dir_dict[dir_list[i]] = dataset[i]
        for i, (dir, df) in enumerate(data_dir_dict.items()):
            if i % 2 == 0:
                df.to_csv(dir + '/dataset.csv')
            else:
                df.to_csv(dir + '/label.csv')
        print('Spliting Complete!')
