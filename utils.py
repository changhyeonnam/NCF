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




class Download_read_csv():
    def __init__(self, root, filename, filetype, download):
        self.root = root
        self.filename = filename
        self.filetype = filetype
        self.download = download
        if self.download:
            self.download_movielens()
            self.ratings = self.read_ratings_csv()
        else:
            self.ratings = self.read_ratings_csv()

    def download_movielens(self) -> None:  # "-> None" 리턴값을 나타내는 것
        root_path = self.root  # 'dataset'
        file_name = self.filename
        file_type = self.filetype
        url = "http://files.grouplens.org/datasets/movielens/" + file_name + file_type

        # Downloading the file by sending the request to the URL
        req = requests.get(url)

        # 최초 directory 생성
        if not os.path.exists(root_path):
            os.makedirs(root_path)

        with open(os.path.join(root_path, file_name + file_type), 'wb') as output_file:
            output_file.write(req.content)

        # extracting the zip file contents
        zipfile = ZipFile(BytesIO(req.content))
        zipfile.extractall(path=root_path)

        print('Dataset Download Complete')

    def read_ratings_csv(self):
        ratings = pd.read_csv(self.root + '/' + self.filename + '/' + 'ratings.csv')
        ratings = ratings.drop("timestamp", axis=1)
        ratings = sklearn.utils.shuffle(ratings)
        print("ratings.csv Read Complete")
        return ratings

    def data_processing(self):
        train_ratings = self.ratings.copy()
        test_ratings = pd.DataFrame()

        for i in self.ratings['userId'].unique().tolist():
            for j in self.ratings['userId'].index:
                if (self.ratings.iloc[j, 0] == i):
                    test_ratings = pd.concat([test_ratings, pd.DataFrame(self.ratings.iloc[j, :]).T], axis=0)
                    train_ratings = train_ratings.drop(j)
                    break
        # explicit feedback -> implicit feedback
        train_ratings.loc[:, 'rating'] = 1
        test_ratings.loc[:, 'rating'] = 1

        train_ratings = train_ratings.astype(int)
        test_ratings = test_ratings.astype(int)

        '''wrong train test split
        x = self.ratings.copy()
        y = self.ratings['userId']
        # stratified sampling
        train_ratings, test_ratings, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)
        '''
        print(len(train_ratings),len(test_ratings),len(self.ratings))
        return train_ratings, test_ratings,self.ratings



class MovieLens(Dataset):
    def __init__(self,
                 dataframe: pd.DataFrame,
                 total_dataframe: pd.DataFrame,
                 root:str='data',
                 train:bool=True,
                 ng_ratio:int=10,
                 filesize:str='large',
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
        self.ng_ratio = ng_ratio

        self.filesize=filesize
        if self.filesize=='small':
            self.df = dataframe
            self.total_df = total_dataframe


        # self._data_label_split()
        self.users, self.items, self.labels = self._preprocess()

    def get_numberof_users_items(self) -> tuple:
        '''
        :return:  df["userId"].nunique(),  df["movieId"].nunique()
        '''
        df = pd.read_csv(os.path.join(self.root, 'ml-1m.total.rating'))
        return df["userId"].max(), df["movieId"].max()

    # def _load_data(self):
    #     '''
    #     if Train=True, load data from dir which is consist of train dataset
    #     else (Test case) load data from dir which is consist of test dataset
    #     :return:
    #     '''
    #     data_file = f"ml-1m.{'train' if self.train else 'test'}.data.rating"
    #     data = pd.read_csv(os.path.join(self.root, data_file))
    #     label_file = f"m1-1m.{'train' if self.train else 'test'}.label.rating"
    #     targets = pd.read_csv(os.path.join(self.root, label_file))
    #     print(f"loading {'train' if self.train else 'test'} file Complete!")
    #     return data, targets

    def __len__(self) -> int:
        '''
        get lenght of data
        :return: len(data)
        '''
        return len(self.users)


    def __getitem__(self, index):
        '''
        transform userId[index], item[inedx] to Tensor.
        and return to Datalaoder object.
        :param index: idex for dataset.
        :return: user,item,rating
        '''
        return self.users[index], self.items[index], self.labels[index]

    # def _download_movielens(self) -> None:
    #     '''
    #     Download dataset from url, if there is no root dir, then mkdir root dir.
    #     After downloading, it wil be extracted
    #     :return: None
    #     '''
    #     file = self.file_dir+'.zip'
    #     url = ("http://files.grouplens.org/datasets/movielens/"+file)
    #     req = requests.get(url, stream=True)
    #     print('Downloading MovieLens dataset')
    #     if not os.path.exists(self.root):
    #         os.makedirs(self.root)
    #     with open(os.path.join(self.root, file), mode='wb') as fd:
    #         for chunk in req.iter_content(chunk_size=None):
    #             fd.write(chunk)
    #     with ZipFile(os.path.join(self.root, file), "r") as zip:
    #         # Extract files
    #         print("Extracting all the files now...")
    #         zip.extractall(path=self.root)
    #         print("Downloading Complete!")

    # def _read_ratings_csv(self) -> pd.DataFrame:
    #     '''
    #     at first, check if file exists. if it doesn't then call _download().
    #     it will read ratings.csv, and transform to dataframe.
    #     it will drop columns=['timestamp'].
    #     :return:
    #     '''
    #     file = self.file_dir+'.zip'
    #     print("Reading file")
    #     zipfile = os.path.join(self.root,file)
    #     if not os.path.isfile(zipfile):
    #         self._download_movielens()
    #     fname = os.path.join(self.root, self.file_dir, 'ratings.csv')
    #     df = pd.read_csv(fname, sep=',').drop(columns=['timestamp'])
    #     print("Reading Complete!")
    #     return df

    def _preprocess(self) :
        '''
        sampling one positive feedback per four negative feedback
        :return: dataframe
        '''
        if self.filesize== 'small':
            total_df = self.total_df
            df = self.df
            users, items, labels = [], [], []
            user_item_set = set(zip(df['userId'], df['movieId']))
            total_user_item_set = set(zip(total_df['userId'],total_df['movieId']))
            all_movieIds = total_df['movieId'].unique()

        else:
            dataframe_file = f"ml-1m.{'train' if self.train else 'test'}.rating"
            df_dir = os.path.join(self.root,dataframe_file)
            df = pd.read_csv(df_dir,sep=',')
            total_df = pd.read_csv(os.path.join(self.root,'ml-1m.train.rating'))
            users, items, labels = [], [], []
            user_item_set = set(zip(df['userId'], df['movieId']))
            total_user_item_set = set(zip(total_df['userId'],total_df['movieId']))
            all_movieIds = total_df['movieId'].unique()

        # negative feedback dataset ratio
        negative_ratio = self.ng_ratio
        for u, i in user_item_set:
            # positive instance
            users.append(u)
            items.append(i)
            labels.append(1.0)

            #visited check
            visited=[]
            visited.append(i)
            # negative instance
            for i in range(negative_ratio):
                # first item random choice
                negative_item = np.random.choice(all_movieIds)
                # check if item and user has interaction, if true then set new value from random
                while ((u, negative_item) in total_user_item_set) :
                    negative_item = np.random.choice(all_movieIds)
                users.append(u)
                items.append(negative_item)
                # visited.append(negative_item)
                labels.append(0.0)

        return torch.tensor(users), torch.tensor(items), torch.tensor(labels)


    # def _data_label_split(self) ->None:
    #     '''
    #     this function divde in to(user,movie) and (rating)
    #     :return: None
    #     '''
    #     dataframe_file = f"ml-1m.{'train' if self.train else 'test'}.rating"
    #     df_dir = os.path.join(self.root,dataframe_file)
    #     df = pd.read_csv(df_dir,sep=',')
    #     df = self._preprocess(df)
    #     data_file = f"ml-1m.{'train' if self.train else 'test'}.data.rating"
    #     label_file = f"m1-1m.{'train' if self.train else 'test'}.label.rating"
    #     dataset_dir = os.path.join(self.root, data_file)
    #     label_dir = os.path.join(self.root, label_file)
    #     dataset, label = df.iloc[:,:-1], df.iloc[:,[-1]]
    #     dataset.to_csv(dataset_dir)
    #     label.to_csv(label_dir)
    # def _train_test_split(self) -> None:
    #     '''
    #     this function is called when downloading dataset.
    #     split dataset in to train and test dataset.
    #     :return: None
    #     '''
    #     df = self.df.copy()
    #     print('Spliting Traingset & Testset')
    #     # Since MovieLens dataset is user-based dataset, I used Stratified k-fold.
    #     train, test,dummy_1,dummy_2 = train_test_split(df,df['userId'],test_size=0.2,stratify=df['userId']) # should add stratify
    #     train_dir = os.path.join(self.root, 'train-dataset-movieLens.csv')
    #     test_dir = os.path.join(self.root, 'test-dataset-movieLens.csv')
    #     train.to_csv(train_dir)
    #     test.to_csv(test_dir)
    #     print('Spliting Complete!')

