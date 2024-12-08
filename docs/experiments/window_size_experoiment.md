# MARL Experiment Documentation

## Experiment Overview

- **Experiment Name:**  Observation window size

  Test the observation window size within the range [4, 8, 16, 27] also was done limited studi i the range [6, 7, 10]. The intencion is to obsorve the efect that changing the window size in the resolts of the espirement.

- **Date:**  
  *[16/11//2024]*

- **What changed from Base Experiment:**
  The only changes introduced in this experiment is the size of the observation window. The size of the observation window is varied folowing this matrix [4,4,4,8,8,8,16,16,16,27,27,27,6,7,10]. the number of epoch is chage in same of the experoiments from 40 to 25. everything else remains the same.

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

# f == 8

python -m marl_classification -a 3 --step 5 --cuda --run-id train_mnist train --action "[[1,0],[-1,0],[0,1],[0,-1]]" --img-size 28 --nb-class 10 -d 2 --f 8 --ft-extr mnist --nb 64 --na 64 --nm 16 --nd 8 --nlb 96 --nla 96 --batch-size 32 --lr 1e-3 --nb-epoch 25 -o ./out/mnist_actor_critic

python -m marl_classification -a 3 --step 5 --cuda --run-id train_mnist train --action "[[1,0],[-1,0],[0,1],[0,-1]]" --img-size 28 --nb-class 10 -d 2 --f 8 --ft-extr mnist --nb 64 --na 64 --nm 16 --nd 8 --nlb 96 --nla 96 --batch-size 32 --lr 1e-3 --nb-epoch 25 -o ./out/mnist_actor_critic

python -m marl_classification -a 3 --step 5 --cuda --run-id train_mnist train --action "[[1,0],[-1,0],[0,1],[0,-1]]" --img-size 28 --nb-class 10 -d 2 --f 8 --ft-extr mnist --nb 64 --na 64 --nm 16 --nd 8 --nlb 96 --nla 96 --batch-size 32 --lr 1e-3 --nb-epoch 25 -o ./out/mnist_actor_critic

```

---

## Results

### 1. Performance Summary

| ![number_of_steps_experiment__performance_summary](.\img\window_size_experoiment\window_size_experoiment_performance_summary1.png)![number_of_steps_experiment__performance_summary](.\img\window_size_experoiment\window_size_experoiment_performance_summary2.png)|
| :-----------------------------------------------------------------------------------------------------------: |
|           fig 1 & 2 : *This table shows the performance summary for different observation window size [4,8,8,8,16,16,27,27,27,6,7,10].*           |

- **Best Episode:**  
  Bests resolts in evaluation of the predictions is with window size of 27,  (resolts are in the intreval of [99,6 to 98,7]% ), but we can have simular resolts with window size of 16 (resolts are in the intreval of [98,9 to 98,6]% ) and the resolts of window size of 8 are above 90%.

- **Convergence:**  
  *Describe whether or not the model converged, if applicable.*

### 3. Graphs and Plots

| ![number_of_steps_experiment__performance_summary](.\img\window_size_experoiment\window_size_experoiment_eval_recs.png)![number_of_steps_experiment__performance_summary](.\img\window_size_experoiment\window_size_experoiment_eval_pecs_scatter_Plot.png)|
| :-----------------------------------------------------------------------------------------------------------: |
|           fig 3 & 4 : *This graphic shows evaluation of the predictions is for different observation window size [4,4,4,8,8,8,16,16,16,27,27,27,6,7,10].* |

| ![number_of_steps_experiment__performance_summary](.\img\window_size_experoiment\window_size_experoiment_actor_loss.png)![number_of_steps_experiment__performance_summary](.\img\window_size_experoiment\window_size_experoiment_actor_loss_scatter_Plot.png)|
| :-----------------------------------------------------------------------------------------------------------: |
|           fig 5 & 6 : *This graphic shows actor loss is for different observation window size [4,4,4,8,8,8,16,16,16,27,27,27,6,7,10].* |

| ![number_of_steps_experiment__performance_summary](.\img\window_size_experoiment\window_size_experoiment_error.png)|
| :-----------------------------------------------------------------------------------------------------------: |
|           fig 7 : *This graphic shows the error for different observation window size [4,4,4,8,8,8,16,16,16,27,27,27,6,7,10].* |

---

## Discussion

### 1. Key Observations

fig 3 and 4:

- From observation window size of 8 up words the performance is more than 90%.
- The observation window size of 16 and 27 does not show significant difference in performance.
- The observation window size of 6 and 7 does show significant improvement to performance when compared to observation window size of 4.
- The observation window size of 6 and 7 shows a good evaluation of the predictions performance.
- The observation window size of 10 shows evaluation of the predictions higher than 95%.
- The observation window size of 6, 7, 8, 10 shows a linear progression in the evaluation of the predictions.
- overall the observation window size shows a logarithmic progression that from values higher that 10 the preformace aperes constant. 

fig 5 and 6:

- The observation window size relation to the actor shows progression of log(1/x).
- the actor loss results shows a values with higher variety when use lower values in observation window size, exemple observation window size 4.

fig 7:

- This graphic amperes identical to the graphic of the fig 5, we can observe that in relation with the observation window size the actor loss and the error are close realtor.
- The observations of the fig 7 are the same that yhe Observations of the fig 5 and 6.

### 2. Issues Encountered

- No major issues were encountered during the experiment, however as the  observation window size increased, the training time increased.

### 3. Future Improvements

- Going forward, it would be interesting to investigate the performance of increased observation window size on a dataset with larger images, does the optimal observation window size be approximate 1/3 of the final image to give optimal results?

---

## Conclusion

- The results of this experiment show that the nobservation window size values has a significant impact on the model's performance. In this experiment, we observed that using observation window size of 8 to 10 resulted in the highest evaluation precision and recall.
- This experiment presents data that proposes that observation window size 4 may not be enough to obtain sufficient information to make meaningful predictions. We can also observe that after the observation window size range of 8 to 10, the actors have all the information they need to carry out predictions making there movement unnecessary.
- We can also observe that the images have a size of 28 and the optimal values ​​for observation window size are a range of 8 to 10 one third of the total image.

---