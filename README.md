# MARLClassification

[Multi-Agent Image Classification via Reinforcement Learning](https://arxiv.org/abs/1905.04835)

_Hossein K. Mousavi, Mohammadreza Nazari, Martin Takáč, Nader Motee_ - 2019

## TODO
- new benchmarks
- docstring

## Description
/!\ old benchmarks /!\

Results reproduction of the above article : 98% on MNIST.

Extend to other image data NWPU-RESISC45 :

| | Loss | Train | Eval |
| --- | --- | :---: | :---: |
| | | prec, rec | prec, rec |
| Epoch 0 | -0.2638 | 6%, 9% | 12%, 14% |
| Epoch 10 | -0.1813 | 55%, 56% | 49%, 48% |
| Epoch 17 | -0.1613 | 63%, 64% | 54%, 53% |

## Installation
```bash
$ cd MARLClassification
$ # create and activate your virtual env
$ python -m venv venv
$ ./venv/bin/activate
$ # install requirements and marl_classification
$ pip install -r requirements.txt
$ pip install .
```

You may download datasets with bash scripts in `res` folder.
## Usage
To run training :
```bash
$ # train on MNIST
$ python -m marl_classification -a 3 --step 5 --cuda --run-id train_mnist train --action [[1,0],[-1,0],[0,1],[0,-1]] --img-size 28 --nb-class 10 -d 2 --f 4 --ft-extr mnist --nb 48 --na 48 --nm 16 --nd 8 --nlb 64 --nla 64 --batch-size 32 --lr 1e-3 --nb-epoch 20 --nr 4 --eps 1. --eps-dec 0.99995 -o ./out/mnist
$ # train on NWPU-RESISC45
$ python -m marl_classification -a 16 --step 16 --cuda --run-id train_resisc45 train --action [[1,0],[-1,0],[0,1],[0,-1]] --ft-extr resisc45 --batch-size 8 --nb-class 45 --img-size 256 -d 2 --nb 256 --na 256 --nd 8 --f 10 --nm 64 --nlb 384 --nla 384 --nb-epoch 50 --nr 4 --learning-rate 1e-4 --eps 1.0 --eps-dec 0.99995 -o ./out/resisc45
$ # train on Knee MRI
$ python -m marl_classification -a 20 --step 15 --cuda --run-id test_train_knee train --action [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]] --ft-extr kneemri --batch-size 3 --nb-class 3 --img-size 320 -d 3 --nb 1536 --na 1536 --nd 16 --f 10 --nm 256 --nlb 2048 --nla 2048 --nb-epoch 50 --nr 1 --learning-rate 2e-5 --eps 0. --eps-dec 1. -o ./out/knee_test
```
