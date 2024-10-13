# MARL Classification <!-- omit in toc -->

Table of Content

- [Folder Structure](#folder-structure)
- [**marl\_classification/__main__.py**:](#marl_classificationmainpy)
  - [Main Arguments](#main-arguments)
  - [Train Arguments](#train-arguments)
  - [Test Arguments](#test-arguments)
  - [Infer Arguments](#infer-arguments)
- [**marl\_classification/train.py**:](#marl_classificationtrainpy)
- [**marl\_classification/metrics.py**:](#marl_classificationmetricspy)
- [**marl\_classification/options.py**:](#marl_classificationoptionspy)
- [**marl\_classification/\_\_init\_\_.py**:](#marl_classification__init__py)
- [**marl\_classification/eval.py**:](#marl_classificationevalpy)


---

## Folder Structure

```bash
marl_classification
├── __init__.py # just another init file
├── __main__.py # entry point for training, testing and inference scripts
├── core
│   ├── __init__.py # exports some classes and functions from core
│   ├── agent.py # defines the Agent class
│   ├── episode.py # defines episode and detailed_episode function (episode_retry is not used)
│   ├── observation.py # defines the Observation function
│   └── transition.py # defines the transition function
├── data # data processing scripts
│   ├── __init__.py
│   ├── dataset.py # define image loaders for different datasets
│   └── transforms.py # defines img transformations (not used)
├── eval.py # defines the evaluation script, is triggered from __main__.py
├── infer.py # defines the inference script, is triggered from __main__.py
├── metrics.py # defines the Loss and Confussion Meter classes
├── networks
│   ├── __init__.py # exports some classes and functions from networks
│   ├── ft_extractor.py # defines the feature extractor classes for different datasets
│   ├── message.py # defines the MessageSender and MessageReceiver classes
│   ├── models.py # defines the ModelsWrapper class
│   ├── policy.py # defines the Policy class and Critic class
│   ├── prediction.py # defines the Prediction class
│   └── recurrent.py # defines the LSTMCellWrapper
├── options.py # defines the options for the training, testing and inference scripts
└── train.py # defines the main training script, is triggered from __main__.py
```

---



## **marl_classification/__main__.py**:
This file is the entry point for the training, testing and inference scripts. It uses the argparse library to parse the command line arguments and call the appropriate function.


To inspect the usage of the script, run the following command:

```bash
python -m marl_classification --help
```

```bash
usage: Multi agent reinforcement learning for image classification - Main [-h] --run-id RUN_ID [-a AGENTS] [--step STEP] [--cuda] {train,test,infer} ...

positional arguments:
  {train,test,infer}

options:
  -h, --help            show this help message and exit
  --run-id RUN_ID       MLFlow run id
  -a AGENTS, --agents AGENTS
                        Number of agents
  --step STEP           Step number of RL episode
  --cuda                Train NNs with CUDA
```

to explore more about the subparsers, run the following command for `train`:

```bash
python -m marl_classification train --help
```

same for `test` and `infer`.


### Main Arguments

- "--run_id": run identifier
- "--agents"/"-a": number of agents
- "--step": number of steps
- "--cuda": trains using CUDA/GPU

### Train Arguments

- "--action": possible steps for each agent
- "--img-size": image size considering them squared
- "--nb_class": number of possible classes in the dataset
- "--dim"/"-d": state dimension (e.g. 2D)
- "--f": window size
- "--ft-extr": feature extractor (e.g. CNN for mnist)
- "--nb": hidden size for action in Long Short-Term Memory (LSTM)
- "--na": hidden size for belief LTSM
- "--nm": message size for Neural Networks
- "--nd": state hidden size
- "--nlb": neuronal internal hidden size for linear projections (action unit)
- "--res-folder": path to the dataset
- "-o"/"output_dir": output directory and models per epoch
- "--batch-size": image batch size for training and eval
- "--lr"/"--learning-rate": learning rate
- "--gamma": discount factor
- "--nb-epoch": number of training epochs
- "--freeze": modules to freeze during training

### Test Arguments

- "--batch-size": batch size for training and eval
- "--dataset-path": dataset path
- "--img-size": image size
- "--json-path": JSON path for multi agent metadata
- "--state-dict-path": ModelsWrapper state dict path
- "-o"/"--output-dir": directory for the outputs

### Infer Arguments

- "--images": path of images used for inference
- "--state-dict-path": ModelsWrapper state dict path
- "--class2idx": Class to index JSON file
- "-o"/"--output-image-dir": directory for the outputs


## **marl_classification/train.py**:

- Used when the main subparser is train
- Declare the model's and output's path
- Call the dataset constructor
- Create the Models Wrapped using the inputs from the train command
- Initiate the agents
- Initiate the training model
- Initiate the RL model
- Create train and test dataset
- Load the data accordingly
- Declare the losses/errors
- Train the model in the number of epochs declared
- Calculates all the losses and errors
- Estimate the global scores
- Stores the information in the mlflow log
- Show the progress in the console
- Save the model information

## **marl_classification/metrics.py**:

- Can be used to format metrics
- Defines the class Meter that stores a list of processed results, which can be limited by a window_size argument
- Defines the class ConfusionMeter which includes functions to calculates metrics such as confusion matrix, precision, and recall. It can also save an image of the confusion matrix

## **marl_classification/options.py**:
- Organizes the options from the run line into MainOptions, TrainOptions, EvalOptions and InferOptions NamedTuple classes

## **marl_classification/\_\_init__.py**:
- Does nothing

## **marl_classification/eval.py**:
- Code for the evaluation process (while training)
- Reads the evaluation options decided by the user
- Evaluates the model on the test dataset
- Loads the agents and the data
- Calls the function that creates the confusion matrix
- Estimates precision, precision mean, recall and recall mean
