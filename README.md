# Postprandial Hyperglycemia Hypoglycemia Prediction
Codes for our work on postprandial hyperglycemia and hypoglycemia prediction.

## Dependencies
This code has been tested using the following environment. 
```
$ conda create --name hyperhypo python=3.8
$ source activate hyperhypo
(hyperhypo)$ conda install pytorch=1.10.2
(hyperhypo)$ pip install numpy matplotlib pandas xml pyyaml scikit-learn
```
This repository is friendly to CPU based training and evaluation.

## Data preparation
To replicate our work, one should prepare the raw OhioT1DM dataset and unzip our pretrained model `pretrained.zip`.
The following structure is required.
```
pretrained/
data/                         
+-- ohiot1dm/                           request from http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html
|   +-- OhioT1DM-training/
|   +-- OhioT1DM-testing/
|   +-- OhioT1DM-2-training/
|   +-- OhioT1DM-2-testing/
|   +-- preprocess_ohiot1dm.py
|   +-- preprocess_utils_ohiot1dm.py
```

## 1. Data Preprocessing
Run our script to preprocess the raw data to `.csv` to directory `./data/ohiot1dm/preprocessed/`.
```
(hyperhypo)$ python ./data/ohiot1dm/preprocess_ohiot1dm.py
```
## 2. Training
Run the following command to train from scratch.
```
(hyperhypo)$ python -m src.train
```

## 3. Evaluation
To replicate our paper results using our pretrained model, run the following command.
```
(hyperhypo)$ python -m src.eval
```
