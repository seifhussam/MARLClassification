# MARL Experiment Documentation

## Experiment Overview

- **Base vs Original Experiment:**  
  *Test the difference between the Base and Original implementations.*

- **Date:**  
  *[05/11/2024]*

- **What changed from Base Experiment:**
  *The hidden size, message size and batch size were changed to meet with the values used in the paper*

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

Running command:
```bash
python -m marl_classification -a 3 --step 5 --run-id original_experiment train --action "[[1,0],[-1,0],[0,1],[0,-1]]" --img-size 28 --nb-class 10 -d 2 --f 6 --ft-extr mnist --nb 64 --na 64 --nm 12 --nd 64 --nlb 64 --nla 64 --batch-size 64 --lr 1e-3 --nb-epoch 50 -o ./out/mnist_original
```

---

## Results

```bash
Epoch 49 - Train, train_prec = 0.843, train_rec = 0.842, c_loss = 1.500, a_loss = 1.482, error = 0.498, path = -1.009:
Epoch 49 - Eval, eval_prec = 0.8429, eval_rec = 0.8467
```

### 3. Graphs and Plots

![alt text](img/number_of_neurons.png.png)

---

## Discussion

### 1. Key Observations

- *Overall, the evaluation curves are similar, except on the last epochs, the results are better, around 0.5% and more consistently than in the other tests.*

### 2. Issues Encountered

- *None*

### 3. Future Improvements

- *Since the values obtained were similar to the obtained with the values suggested by the repository's author, this values should be considered the default ones, to present closer similarity with the original paper.*

---

## Conclusion

- *Besides not showing much improvement, the values used are faithful to the original article implementation and therefore should be considered the default.*

---