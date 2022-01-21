## NCFML (Neural Collaborative Filtering with MovieLens in torch)

### Dataset
This repository is about Neural Collaborative Filtering with MovieLens in torch.
If there is interaction between user and item, then target value will be 1. (which means Implict Feedback)
So if there is rating value between user and movie, then target value is 1, otherwise 0. 
For negative sampling, ratio between positive feedback and negative feedback is 1:4 in trainset, and 1:99 in testset. (these ratios are same as author's code [@hexiangnan](https://github.com/hexiangnan/neural_collaborative_filtering))

You can use 100k, 1m, 10m, 20m dataset by using parser parameter `--file_size`.

### Pakage installation

```python
conda install --file requirements.txt

```

### Neural Collaborative Filtering model directory tree

```python
.
├── README.md
├── evaluation.py
├── main.py
├── model
│   ├── GMF.py
│   ├── MLP.py
│   └── NeuMF.py
├── pretrain
├── train.py
└── utils.py
```

### Quick start
```python
python main.py --epoch 30 --batch 256 --factor 8 --model NeuMF --topk 10 --file_size 100k --layer 64 32 16 --download True --use_pretrain False

```

### Development enviroment

- OS: Max OS X
- IDE: pycharm
- GPU: NVIDIA RTX A6000


### Neural Collaborative Filtering Result

| movielens 100K | Best HR@10 | NDCG@10 | Runtime | epoch | preditivie factor | batch_size | layer for MLP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GMF | 0.815 | 0.552 | 2m 40sec | 20 | 8 | 256 | X |
| MLP | 0.803 | 0.567 | 17m 42sec | 20 | 8 | 256 | [64,32,16] |
| NeuMF (without pre-training) | 0.828 | 0.574 | 21m 19sec | 20 | 8 | 256 | [64,32,16] |
| NeuMF (pretrained) | 0.980 | 0.702 | 4m 42sec | 20 | 8 | 256 | X |


| movielens-1M | Best HR@10 | NDCG@10 | Runtime | epoch | preditivie factor | batch size | layer for MLP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GMF | 0.761 | 0.493 | 26m 45sec | 20 | 8 | 256 | X |
| MLP | 0.785 | 0.526 | 1h 47m 24sec | 20 | 8 | 256 | [64,32,16] |
| NeuMF (without pre-training) | 0.796 | 0.538 | 1h 46m 20sec | 20 | 8 | 256 | [64,32,16] |
| NeuMF (pretrained) | 0.851 | 0.854 | 41m 22sec | 20 | 8 | 256 | X |

### Example of command line

- save GMF
  ```python
  python main.py --epoch 30 --batch 256 --factor 8 --model GMF --topk 10
  --file_size 100k --layer 64 32 16 --download True --save True

  ```
- save MLP

  ```python
  python main.py --epoch 30 --batch 256 --factor 8 --model MLP --topk 10
  --file_size 100k --layer 64 32 16 --download False --save True

  ```
- use pre-trained model
  ```python
  python main.py --epoch 30 --batch 256 --factor 8 --model NeuMF  --topk 10
  --file_size 100k --layer 64 32 16 --download False --use_pretrain True
  ```



reference : [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031)

review written in korean : [Review](https://changhyeonnam.github.io/2021/12/28/Neural_Collaborative_Filtering.html)

