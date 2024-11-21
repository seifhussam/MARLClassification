# MARL Experiment Documentation

## Experiment Overview

- **Communication Experiment:**  
  *Test the impact communication has on the model performance.*

- **Date:**  
  *[20/11/2024]*

- **What changed from Original Experiment:**
  *The communication between agents is turned off in this experiment.*

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
| `--nm`         | `12`                            | *Message size for Neural Networks.*                                  |
| `--nd`         | `8`                             | *State Hidden Size.*                                                 |
| `--nlb`        | `64`                            | *Network internal hidden size for linear projections (belief unit).* |
| `--nla`        | `64`                            | *Network internal hidden size for linear projections (action unit).* |
| `--batch-size` | `64`                            | *Batch Size.*                                                        |
| `--lr`         | `1e-3`                          | *This is the learning rate.*                                         |
| `--nb-epoch`   | `50`                            | *This is the number of Epochs.*                                      |
| `--msg`        | `none`                          | *This defines how the communication between agents is done*          |

Running command:
```bash
python -m marl_classification -a 3 --step 5 --run-id train_mnist train --action "[[1,0],[-1,0],[0,1],[0,-1]]" --img-size 28 --nb-class 10 -d 2 --f 6 --ft-extr mnist --nb 64 --na 64 --nm 16 --nd 8 --nlb 96 --nla 96 --batch-size 32 --lr 1e-3 --nb-epoch 50 -o ./out/mnist_actor_critic --msg "none"
```

---

## Results

```bash
Epoch 49 - Train, train_prec = 0.8017, train_rec = 0.8008, c_loss = 1.585, a_loss = 2.0849, error = 0.583, path = -0.8311:
Epoch 49 - Eval, eval_prec = 0.8337, eval_rec = 0.8262
```

### 3. Graphs and Plots

![alt text](communication_experiment.png)

---

## Discussion

### 1. Key Observations

- *The classification accuracy is not as good as it is with communication, as expected. The decrease in evaluation performance is around one percent. However, the decrease in training is much more significant, around 5%.*

### 2. Issues Encountered

- *None*

### 3. Future Improvements

- *There are no future improvements regarding this experiment. The results are better with communication turned on and therefore, to obtain better results, it must be kept that way. To observe a more significant change, the number of steps per episode should be reduced.*

---

## Conclusion

- *The impact no communication has on the framework evaluation accuracy is around 1%. On training, it is around 5%.*

---