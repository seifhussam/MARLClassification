# MARLClassification

David Castanho Terroso, 202308694
João Longras, 
Pedro Azevedo, 201905966
Seifeldin Mostafa, 202403076

In this repository, the work done in the curricular unit of "Topics in Intelligent Systems" of the Master's in Artificial Intelligence. The work developed is based on the paper [Multi-Agent Image Classification via Reinforcement Learning](https://arxiv.org/abs/1905.04835) and code here presented is inspired by [this repository](https://github.com/Ipsedo/MARLClassification).

There are two branches in this repository: [main](https://github.com/seifhussam/MARLClassification), where most of the experiments were done and inspired in the original repository, and [original_experiment](https://github.com/seifhussam/MARLClassification/tree/original_experiment), where experiments more faithful to the selected article were conducted. Below, it is possible to see the main findings of this project.

## Main Findings
### Communication
The experiments regarding communication were done in the [original_experiment](https://github.com/seifhussam/MARLClassification/tree/original_experiment) branch. The main differences between [original_experiment](https://github.com/seifhussam/MARLClassification/tree/original_experiment) and [main](https://github.com/seifhussam/MARLClassification) were the communication module implementation: the message was encoded by each agent at the time it was being sent, but was not being decoded. In this branch, that is fixed and allows the training to be done with partial communication, as it was in the original repository, with full communication, as it is done in the original article, and with no communication at all. From these three experiments, described in [this folder](https://github.com/seifhussam/MARLClassification/tree/original_experiment/docs/experiments), the following was concluded:

## Installation
```bash
$ cd /path/to/MARLClassification
$ # create and activate your virtual env
$ python -m venv venv
$ ./venv/bin/activate
$ # install requirements
$ pip install -r requirements.txt
$ # REQUIRES KAGGLE API KEY: download datasets using sh scripts in resources folder, ex : MNIST
$ ./resources/download_mnist.sh
```

You may download datasets with bash scripts in `res` folder.
## Usage
To run training :
```bash
$ cd /path/to/MARLClassification
$ # train on MNIST
$ python -m marl_classification -a 3 --step 5 --cuda --run-id train_mnist train --action [[1,0],[-1,0],[0,1],[0,-1]] --img-size 28 --nb-class 10 -d 2 --f 6 --ft-extr mnist --nb 64 --na 64 --nm 16 --nd 8 --nlb 96 --nla 96 --batch-size 32 --lr 1e-3 --nb-epoch 40 -o ./out/mnist_actor_critic
$ # train on NWPU-RESISC45
$ python -m marl_classification -a 16 --step 16 --cuda --run-id train_resisc45 train --action [[1,0],[-1,0],[0,1],[0,-1]] --ft-extr resisc45 --batch-size 8 --nb-class 45 --img-size 256 -d 2 --nb 256 --na 256 --nd 16 --f 12 --nm 64 --nlb 384 --nla 384 --nb-epoch 50 --lr 1e-4 -o ./out/resisc45_actor_critic
$ # train on AID
$ python -m marl_classification -a 16 --step 16 --cuda --run-id train_aid train --action [[3,0],[-3,0],[0,3],[0,-3]] --ft-extr aid --batch-size 8 --nb-class 30 --img-size 600 -d 2 --nb 256 --na 256 --nd 16 --f 24 --nm 64 --nlb 320 --nla 320 --nb-epoch 50 --lr 1e-4 -o ./out/aid_actor_critic
```

## Reference

[1]: https://arxiv.org/abs/1905.04835, _Hossein K. Mousavi, Mohammadreza Nazari, Martin Takáč, Nader Motee_ - 2019
