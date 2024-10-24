# MARL Experiment Documentation

## Experiment Overview

- **Original Experiment:**  
  *This experiment aims to reproduce the experiment with the original hyperparameters present in the paper*

- **Date:**  
  *[DD/10/2024]*

- **What changed from Original Experiment:**
  *Hyperparameters*

- **What is expected from Original Experiment:**
  *An accuracy close to 88%, as obtained in the paper*

---

## Experimental Setup

### 1. Hyperparameters

| Hyperparameter | Value                           | Description                                                          |
| -------------- | ------------------------------- | -------------------------------------------------------------------- |
| `-a`           | `2`                             | *Number of agents.*                                                  |
| `--step`       | `9`                             | *Number of steps (Time Horizon, in the paper).*                                                   |
| `--action`     | `"[[1,0],[-1,0],[0,1],[0,-1]]"` | *Possible steps for each agent.*                                     |
| `--img-size`   | `28`                            | *Image Size.*                                                        |
| `--nb-class`   | `10`                            | *Number of possible classes in the dataset.*                         |
| `-d`           | `2`                             | *State dimension (e.g. 2D).*                                         |
| `--f`          | `2`                             | *Observation window size.*                                           |
| `--ft-extr`    | `mnist`                         | *Feature extractor (e.g. CNN for mnist).*                            |
| `--nb`         | `64`                            | *Hidden size for belief in Long Short-Term Memory (LSTM).*           |
| `--na`         | `64`                            | *Hidden size for Action in Long Short-Term Memory (LSTM).*           |
| `--nm`         | `12`                            | *Message size for Neural Networks.*                                  |
| `--nd`         | `64`                             | *State Hidden Size.*                                                 |
| `--nlb`        | `64`                            | *Network internal hidden size for linear projections (belief unit).* |
| `--nla`        | `64`                            | *Network internal hidden size for linear projections (action unit).* |
| `--batch-size` | `64`                            | *Batch Size.*                                                        |
| `--lr`         | `1e-3`                          | *This is the learning rate.*                                         |
| `--nb-epoch`   | `50`                            | *This is the number of Epochs.*                                      |

Running command:
```bash
python -m marl_classification -a 2 --step 9 --run-id original_experiment train --action "[[1,0],[-1,0],[0,1],[0,-1]]" --img-size 28 --nb-class 10 -d 2 --f 2 --ft-extr mnist --nb 64 --na 64 --nm 12 --nd 64 --nlb 64 --nla 64 --batch-size 64 --lr 1e-3 --nb-epoch 50 -o ./out/mnist_original
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

- *There is no information in the paper regarding the learning rate, which was kept the same.*

### 3. Future Improvements

- *Study starting point. Every single aspect here tested must be changed in further stuies.*

---

## Conclusion

- *Summarize the overall performance and findings from this experiment.*
- *Mention any final takeaways.*

---

## Appendix

### 1. Code

- *Include a link or a snippet of the code used for replicating the experiment results if necessary.*