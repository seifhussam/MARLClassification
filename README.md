# MARLClassification

- David Castanho Terroso, 202308694
- João Longras, 202108780
- Pedro Azevedo, 201905966
- Seifeldin Mostafa, 202403076

In this repository, the work done in the curricular unit of "Topics in Intelligent Systems" of the Master's in Artificial Intelligence. The work developed is based on the paper [Multi-Agent Image Classification via Reinforcement Learning](https://arxiv.org/abs/1905.04835) [1] and code here presented is inspired by [this repository](https://github.com/Ipsedo/MARLClassification). <br>
There are two branches in this repository: [main](https://github.com/seifhussam/MARLClassification), where most of the experiments were done and inspired in the original repository, and [original_experiment](https://github.com/seifhussam/MARLClassification/tree/original_experiment), where experiments more faithful to the selected article were conducted. Below, it is possible to see the main findings of this project. <br>
To further understand the organization of this repository, please check the documentation.md file in each branch: [main](https://github.com/seifhussam/MARLClassification/blob/main/documentation.md) and [original_experiment](https://github.com/seifhussam/MARLClassification/blob/original_experiment/documentation.md).

## Main Findings

### Communication

The experiments regarding communication were done in the [original_experiment](https://github.com/seifhussam/MARLClassification/tree/original_experiment) branch. The main differences between [original_experiment](https://github.com/seifhussam/MARLClassification/tree/original_experiment) and [main](https://github.com/seifhussam/MARLClassification) were the communication module implementation: the message was encoded by each agent at the time it was being sent, but was not being decoded. In this branch, that is fixed and allows the training to be done with partial communication, as it was in the original repository, with full communication, as it is done in the original article, and with no communication at all. From these three experiments, described in the following was concluded:

- With three agents, five steps, and a field of view with side six, the communication does not present a major role in the classification task. This may be explained by the large area covered by each agent in this experiment. In cases where a smaller field of view or smaller number of steps, communication could present a more important function. The difference in accuracy for this specific case is a 2% improvement in the experiment where communication is used. For more informations regarding this experiment, check [this document](https://github.com/seifhussam/MARLClassification/blob/original_experiment/docs/experiments/communication_experiment.md);
- The decoding of the message does not affect the communication, as reflected in the metrics measured in evaluation, as explained in [this document](https://github.com/seifhussam/MARLClassification/blob/original_experiment/docs/experiments/base_vs_original_experiment.md).

### Message and batch size, and number of neurons

To improve the similarity between the experiment and the paper, the number of neurons in the layers, the message and the batch size was changed to match the ones described in the paper. More information about this experiment can be checked in [this document](https://github.com/seifhussam/MARLClassification/blob/original_experiment/docs/experiments/number_of_neurons_experiment.md). This experiment was done using the fully implemented communication module. The changes in the model's performance were insignificant and no correlation between the number of neurons, and the message and batch size experimented and the model's performance can be drawn. <br>
However, to further test whether this change, combined with the fully implemented communication module, the number of agents, steps, and field of view was increased. While the training duration was significantly increased, the results were also the best obtained for this dataset. Due to the computational burden, no more experiments were done to test further improvements or close performances. More information about this experiment can be checked in [this document](https://github.com/seifhussam/MARLClassification/blob/original_experiment/docs/experiments/original_with_max_agents_steps_fov.md)

### Number of steps

The experiments regarding the number of steps were done in the main branch. The purpose of the experiment is to understand the impact of the number of steps taken by the agents on the model's performance. The experiment was conducted with different number of steps [1,3,5,6,7,8,9]. The results show that using 7 steps resulted in the highest evaluation precision and recall. This suggests that the agents need a sufficient number of steps to explore the environment and make informed decisions and contribute to the classification task.

Experiment findings could be found in [this document](./docs/experiments/number_of_steps_experiment.md)

### Number of agents

The experiments regarding the number of agents were conducted to evaluate their impact on the classification task's performance. Different configurations of agents were tested, ranging from 3 to 6 agents. Increasing the number of agents generally improved the model's performance due to better coverage of the visual information. However, as the number of agents exceeded a certain threshold, the performance gain plateaued, likely due to communication bottlenecks or redundant information sharing. A balanced number of agents is crucial to maximize classification accuracy while avoiding computational inefficiencies.

Experiment findings could be found in [this document](./docs/experiments/experiments_number_of_agents.md)

### Agent Actions

The experiments involving agent actions highlighted the importance of flexibility in movement configurations. Configurations including diagonal actions resulted in better pathfinding capabilities, improving precision and recall compared to basic movements. This indicates that expanding the action space allows agents to navigate the environment more effectively, contributing to better overall performance in classification tasks.

Experiment findings could be found in [this document](./docs/experiments/actions_expreriment.md)

### CIFAR-10 Dataset

The experiments regarding the CIFAR-10 dataset were done in the main branch. The purpose of the experiment was to explore a more complex dataset than MNIST. Within this experiment, the number of agents were varied within this list [3, 6, 10, 12, 18]. However, after the number of agents exceeded 10, the number of steps decreased due to high computational cost.

The results show that using a low number of steps and a high number of agents has a positive impact on the performance of the MARL model on the CIFAR-10 dataset. Each agent only had partial information about the image, and the communication module helped the agents to share information and make better decisions.

Experiment findings could be found in [this document](./docs/experiments/ciphar_10_experiment.md)

### Observation Window Size

The experiments regarding the observation window size were done in the main branchbecause these experiments do not require changing this code only the parameters. This experiment aims to analyze the behavior of the model when the size of the agents observation window changes. There where two types of experiments were carried out, one where each size (4, 8, 16, 27) was tested three times to introduce robustness to the results, and another where only one test was carried out for each size (6, 7, 10) with the intention of increasing the range of possible sizes to study and complete the results. previous.

The results demonstrate that from size 10 the accuracy remains relatively constant at values ​​above 95%, in the interval of 4 (the smallest possible value) and 27 (the highest possible value) the accuracy increases logarithmically tending towards 100% where at size 10 the growth is insignificant. We can also see an almost opposite behavior for the "actor loss".

For the mnist dataset, a size between 8 and 10 is recommended to obtain a good result without significantly increasing computational power.

Experiment findings could be found in [this document](./docs/experiments/window_size_experoiment.md)

## Installation

```bash
cd /path/to/MARLClassification
# create and activate your virtual env
python -m venv venv
./venv/bin/activate
# install requirements
pip install -r requirements.txt
# REQUIRES KAGGLE API KEY: download datasets using sh scripts in resources folder, ex : MNIST
./resources/download_mnist.sh
```

You may download datasets with bash scripts in `res` folder.

## Usage

To run training:

```bash
cd /path/to/MARLClassification
# train on MNIST
python -m marl_classification -a 3 --step 5 --cuda --run-id train_mnist train --action [[1,0],[-1,0],[0,1],[0,-1]] --img-size 28 --nb-class 10 -d 2 --f 6 --ft-extr mnist --nb 64 --na 64 --nm 16 --nd 8 --nlb 96 --nla 96 --batch-size 32 --lr 1e-3 --nb-epoch 40 -o ./out/mnist_actor_critic
```

## Reference

[1] H. K. Mousavi, M. Nazari, M. Takáč, and N. Motee, "Multi-agent image classification via reinforcement learning", in 2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2019, pp. 5020-5027.
