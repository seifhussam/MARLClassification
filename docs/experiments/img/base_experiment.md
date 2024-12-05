# Baseline Experiment Documentation

## Experiment Overview

- **Experiment Name:**  
  This is the baseline experiment, running the code as is with default hyperparameters.

- **Date:**  
  *[08/10/2024]*

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

```bash
python -m marl_classification -a 3 --step 5 --run-id train_mnist train --action "[[1,0],[-1,0],[0,1],[0,-1]]" --img-size 28 --nb-class 10 -d 2 --f 6 --ft-extr mnist --nb 64 --na 64 --nm 16 --nd 8 --nlb 96 --nla 96 --batch-size 32 --lr 1e-3 --nb-epoch 50 -o ./out/mnist_actor_critic
```


---

## Results


```bash
Epoch 49 - Train, train_prec = 0.827, train_rec = 0.826, c_loss = 1.5211, a_loss = 1.5761, error = 0.5015, path = -0.9313:
Epoch 49 - Eval, eval_prec = 0.8172, eval_rec = 0.8145
```

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