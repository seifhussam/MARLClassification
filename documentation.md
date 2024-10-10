## **marl_classification/__main__.py**:
- Beggining of the code
- Receives arguments regarding the code run

### Main subparser

- Must be either "train", "test" or "infer"

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