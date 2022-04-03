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

  - --n_iter: number of iterations
  - --fold: [0, 1, 2]
  - --epochs: number of epochs
  - --weight_decay: weight decay
  - --batch_size: batch size
  - --save_model: whether save model or not, for example, 'python train.py' will not save the model and 'python train.py --save_model' will save the model.
  - --lr: learn rate

