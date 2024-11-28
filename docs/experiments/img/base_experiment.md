# Baseline Experiment Documentation

## Experiment Overview

- **Experiment Name:**  
  Impact of Movement Variations on MARL Agent Performance.

- **Date:**  
  *[06/11/2024]*

- **What changed from Base Experiment:**  
  Three different movement configurations were tested:
  1. **Basic movements:** `[1,0], [-1,0], [0,1], [0,-1]`.
  2. **Diagonal movements only:** `[1,1], [-1,-1], [1,-1], [-1,1]`.
  3. **Combined movements:** `[1,0], [-1,0], [0,1], [0,-1], [1,1], [-1,-1], [1,-1], [-1,1]`.
---

## Experimental Setup

### Hyperparameters

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

1. **Basic Movements:** `[1,0], [-1,0], [0,1], [0,-1]`.  

```bash
python -m marl_classification -a 3 --step 5 --run-id train_mnist train --action "[[1,0],[-1,0],[0,1],[0,-1]]" --img-size 28 --nb-class 10 -d 2 --f 6 --ft-extr mnist --nb 64 --na 64 --nm 16 --nd 8 --nlb 96 --nla 96 --batch-size 32 --lr 1e-3 --nb-epoch 50 -o ./out/mnist_actor_critic
```
```bash

2. **Diagonal Movements Only:** `[1,1], [-1,-1], [1,-1], [-1,1]`.  

python -m marl_classification -a 3 --step 5 --run-id train_mnist train --action "[[1,1],[-1,-1],[1,-1],[-1,1]]" --img-size 28 --nb-class 10 -d 2 --f 6 --ft-extr mnist --nb 64 --na 64 --nm 16 --nd 8 --nlb 96 --nla 96 --batch-size 32 --lr 1e-3 --nb-epoch 50 -o ./out/mnist_actor_critic
```

3. **Combined Movements:** `[1,0], [-1,0], [0,1], [0,-1], [1,1], [-1,-1], [1,-1], [-1,1]`.

```bash
python -m marl_classification -a 3 --step 5 --run-id train_mnist train --action "[[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,-1],[1,-1],[-1,1]]" --img-size 28 --nb-class 10 -d 2 --f 6 --ft-extr mnist --nb 64 --na 64 --nm 16 --nd 8 --nlb 96 --nla 96 --batch-size 32 --lr 1e-3 --nb-epoch 50 -o ./out/mnist_actor_critic
```


## Results

| Metric                    | Basic Movements      | Diagonal Movements Only | Combined Movements     |
|---------------------------|----------------------|--------------------------|------------------------|
| **Train Precision**       | `0.798`             | `0.857`                  | `0.858`                |
| **Train Recall**          | `0.796`             | `0.857`                  | `0.858`                |
| **Train Loss (Actor)**    | `1.9623`            | `1.399`                  | `1.399`                |
| **Train Loss (Critic)**   | `1.4823`            | `1.491`                  | `1.527`                |
| **Error (Train)**         | `0.5723`            | `0.477`                  | `0.368`                |
| **Eval Precision**        | `0.8088`            | `0.85`                   | `0.846`                |
| **Eval Recall**           | `0.803`             | `0.845`                  | `0.84`                 |
| **Eval Loss (Actor)**     | `1.817`             | `2.89`                   | `2.89`                 |
| **Eval Loss (Critic)**    | `1.547`             | `0.987`                  | `-1.547`               |



### 1. Performance Summary

- **Best Episode:**  
  *Provide details on the best performing episode (e.g., episode number, reward achieved).*

- **Average Reward:**  
  *Summarize the average reward over episodes.*

- **Convergence:**  
  *Describe whether or not the model converged, if applicable.*

### 3. Graphs and Plots

![baseline experiments metrics](./img/base_experiment_metrics.png)

---

## Discussion

### 1. Key Observations
NA

### 2. Issues Encountered

- Message Reciver is not applied to the model.

### 3. Future Improvements
NA

---

## Conclusion

- *Summarize the overall performance and findings from this experiment.*
- *Mention any final takeaways.*

---

## Appendix

### 1. Code

- Results could be replicated on the following commit:
```bash
git checkout 4d924926e9fc1a05a4457fc905ed4018b554aa87
```