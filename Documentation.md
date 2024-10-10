**marl_classification/__main__.py**:
- Beggining of the code
- Receives arguments regarding the code run

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

### Main subparser

- Must be either "train", "test" or "infer"