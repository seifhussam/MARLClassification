# MARLClassification

David Castanho Terroso, 202308694
João Longras, 
Pedro Azevedo, 201905966
Seifeldin Mostafa, 202403076

In this repository, the work done in the curricular unit of "Topics in Intelligent Systems" of the Master's in Artificial Intelligence. The work developed is based on the paper [Multi-Agent Image Classification via Reinforcement Learning](https://arxiv.org/abs/1905.04835) [1] and code here presented is inspired by [this repository](https://github.com/Ipsedo/MARLClassification). <br>
There are two branches in this repository: [main](https://github.com/seifhussam/MARLClassification), where most of the experiments were done and inspired in the original repository, and [original_experiment](https://github.com/seifhussam/MARLClassification/tree/original_experiment), where experiments more faithful to the selected article were conducted. Below, it is possible to see the main findings of this project. <br>
To further understand the organization of this repository, please check the documentation.md file in each branch: [main](https://github.com/seifhussam/MARLClassification/blob/main/documentation.md) and [original_experiment](https://github.com/seifhussam/MARLClassification/blob/original_experiment/documentation.md).

## Main Findings
### Communication
The experiments regarding communication were done in the [original_experiment](https://github.com/seifhussam/MARLClassification/tree/original_experiment) branch. The main differences between [original_experiment](https://github.com/seifhussam/MARLClassification/tree/original_experiment) and [main](https://github.com/seifhussam/MARLClassification) were the communication module implementation: the message was encoded by each agent at the time it was being sent, but was not being decoded. In this branch, that is fixed and allows the training to be done with partial communication, as it was in the original repository, with full communication, as it is done in the original article, and with no communication at all. From these three experiments, described in the following was concluded:
- With three agents, five steps, and a field of view with side six, the communication does not present a major role in the classification task. This may be explained by the large area covered by each agent in this experiment. In cases where a smaller field of view or smaller number of steps, communication could present a more important function. The difference in accuracy for this specific case is a 2% improvement in the experiment where communication is used. For more informations regarding this experiment, check [this document](https://github.com/seifhussam/MARLClassification/blob/original_experiment/docs/experiments/communication_experiment.md);
- The decoding of the message does not affect the communication, as reflected in the metrics measured in evaluation, as explained in [this document](https://github.com/seifhussam/MARLClassification/blob/original_experiment/docs/experiments/base_vs_original_experiment.md).

### Number of neurons
To improve the similarity between the experiment and the paper, the number of neurons in the layers was changed to match the ones described in the paper. More information about this experiment can be checked in [this document](https://github.com/seifhussam/MARLClassification/blob/original_experiment/docs/experiments/number_of_neurons_experiment.md). This experiment was done using the fully implemented communication module. The changes in the model's performance were insignificant and no correlation between the number of neurons experimented and the model's performance can be drawn. <br>
However, to further test whether this change, combined with the fully implemented communication module, the number of agents, steps, and field of view was increased. While the training duration was significantly increased, the results were also the best obtained for this dataset. Due to the computational burden, no more experiments were done to test further improvements or close performances. More information about this experiment can be checked in [this document](https://github.com/seifhussam/MARLClassification/blob/original_experiment/docs/experiments/original_with_max_agents_steps_fov.md)


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
To run training:
```bash
$ cd /path/to/MARLClassification
$ # train on MNIST
$ python -m marl_classification -a 3 --step 5 --cuda --run-id train_mnist train --action [[1,0],[-1,0],[0,1],[0,-1]] --img-size 28 --nb-class 10 -d 2 --f 6 --ft-extr mnist --nb 64 --na 64 --nm 16 --nd 8 --nlb 96 --nla 96 --batch-size 32 --lr 1e-3 --nb-epoch 40 -o ./out/mnist_actor_critic
```

## Reference
[1]
H. K. Mousavi, M. Nazari, M. Takáč, and N. Motee, "Multi-agent image classification via reinforcement learning", in 2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2019, pp. 5020-5027.