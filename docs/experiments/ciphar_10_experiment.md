# MARL Experiment Documentation

## Experiment Overview

- **Cifar 10 Experiment:**  
  This experiment aims to train a Multi-Agent Reinforcement Learning (MARL) model on the Cifar 10 dataset. 


- **Date:**  
  *28/11/2024*

- **What changed from Base Experiment:**
  - This experiment uses the Cifar 10 dataset instead of the MNIST dataset.
  - Agent can only do 7 steps for each episode.
  - Agent can only move diagonally.
  - Number of agents set are 3, 6, 10.


---

## Experimental Setup

### 1. Hyperparameters

| Hyperparameter | Value                           | Description                                                          |
| -------------- | ------------------------------- | -------------------------------------------------------------------- |
| `-a`           | `3`                             | *Number of agents.*                                                  |
| `--step`       | `5`                             | *Number of steps.*                                                   |
| `--action`     | `"[[1,0],[-1,0],[0,1],[0,-1]]"` | *Possible steps for each agent.*                                     |
| `--img-size`   | `28`                            | *Image Size.*                                                        |
| `--nb-class`   | `10`                            | *Number of possible classes in the dataset.*                         |
| `-d`           | `2`                             | *State dimension (e.g. 2D).*                                         |
| `--f`          | `6`                             | *Observation window size.*                                           |
| `--ft-extr`    | `mnist`                         | *Feature extractor (e.g. CNN for mnist).*                            |
| `--nb`         | `64`                            | *Hidden size for belief in Long Short-Term Memory (LSTM).*           |
| `--na`         | `64`                            | *Hidden size for Action in Long Short-Term Memory (LSTM).*           |
| `--nm`         | `16`                            | *Message size for Neural Networks.*                                  |
| `--nd`         | `8`                             | *State Hidden Size.*                                                 |
| `--nlb`        | `96`                            | *Network internal hidden size for linear projections (belief unit).* |
| `--nla`        | `96`                            | *Network internal hidden size for linear projections (action unit).* |
| `--batch-size` | `32`                            | *Batch Size.*                                                        |
| `--lr`         | `1e-3`                          | *This is the learning rate.*                                         |
| `--nb-epoch`   | `50`                            | *This is the number of Epochs.*                                      |

Running command:
for 3 agents
```bash
# add running command here
python -m marl_classification -a 3 --step 7 --run-id train_cifar_10__3_agents__7_steps train --action "[[1,1],[-1,-1],[-1,1],[1,-1]]" --img-size 32 --nb-class 10 -d 2 --f 8 --ft-extr cifar_10 --nb 64 --na 64 --nm 16 --nd 8 --nlb 96 --nla 96 --batch-size 32 --lr 1e-3 --nb-epoch 50 -o ./out/cifar_actor_critic
```

for 6 agents
```bash
# add running command here
python -m marl_classification -a 6 --step 7 --run-id train_cifar_10__6_agents__7_steps train --action "[[1,1],[-1,-1],[-1,1],[1,-1]]" --img-size 32 --nb-class 10 -d 2 --f 8 --ft-extr cifar_10 --nb 64 --na 64 --nm 16 --nd 8 --nlb 96 --nla 96 --batch-size 32 --lr 1e-3 --nb-epoch 50 -o ./out/cifar_actor_critic_6_agents

```

for 10 agents
```bash
# add running command here
python -m marl_classification -a 10 --step 7 --run-id train_cifar_10__10_agents__7_steps train --action "[[1,1 ],[-1,-1],[-1,1],[1,-1]]" --img-size 32 --nb-class 10 -d 2 --f 8 --ft-extr cifar_10 --nb 64 --na 64 --nm 16 --nd 8 --nlb 96 --nla 96 --batch-size 32 --lr 1e-3 --nb-epoch 50 -o ./out/cifar_actor_critic_10_agents
```
---

## Results

```bash
# add results here
```


### 1. Performance Summary

- **Best Episode:**  
  *Provide details on the best performing episode (e.g., episode number, reward achieved).*

- **Average Reward:**  
  *Summarize the average reward over episodes.*

- **Convergence:**  
  *Describe whether or not the model converged, if applicable.*

### 3. Graphs and Plots

- *Attach training curves or evaluation graphs (e.g., reward over time, loss curve, exploration rate over time).*

---

## Discussion

### 1. Key Observations

- *Summarize key findings from the experiment.*
- *Mention any peculiar observations or deviations.*

### 2. Issues Encountered

- *List any issues encountered during the experiment (e.g., convergence issues, hardware limitations).*

### 3. Future Improvements

- *Suggest possible changes or improvements for future experiments (e.g., adjusting hyperparameters, changing environment, modifying network architecture).*

---

## Conclusion

- *Summarize the overall performance and findings from this experiment.*
- *Mention any final takeaways.*

---

## Appendix

### 1. Code

- *Include a link or a snippet of the code used for replicating the experiment results if necessary.*