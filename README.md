# Learning size-adaptive molecular substructures for drug-drug interaction prediction by substructure-aware graph neural network

## Requirements  

## Step-by-step running:  
- First, cd SA-DDI/drugbank, and run data_preprocessing.py using  
  `python data_preprocessing.py -d drugbank -o all`  

  Running data_preprocessing.py convert the raw data into graph format.

- Second, run train.py using 
  `python train.py --fold 0 --save_model` 

  to train SA-DDI. The training record can be found in save/ folder.

  Explanation of parameters

  - --fold: k-th fold, from 0 to 4
  - --save_model: whether save model or not
  - --lr: learning rate, default =  5e-4
  - --batch_size: default = 512
