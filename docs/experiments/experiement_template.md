# MARL Experiment Documentation

## Experiment Overview

- **Experiment Name:**  
  *Describe the purpose or goal of this experiment.*

- **Date:**  
  *[DD/MM//YYYY]*

- **What changed from Base Experiment:**
  *Describe the changes made to the base experiment (e.g., hyperparameters, network architecture).*

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
```bash
# add running command here
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