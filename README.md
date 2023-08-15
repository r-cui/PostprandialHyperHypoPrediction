# Postprandial Hyperglycemia Hypoglycemia Prediction
This is the official repository of the paper "Jointly Predicting Postprandial Hypoglycemia and Hyperglycemia Using Continuous Glucose Monitoring Data in Type 1 Diabetes" in EMBC2023.

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
Please request the OhioT1DM dataset from the original producer Ohio University via http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html.

Then make the the following structure in the workspace.
```
pretrained/
data/                         
+-- ohiot1dm/                           
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
## 2. Replicating Paper Results
Run the following command to load our model weights and validate the paper scores.
```
(hyperhypo)$ python -m src.eval
```

## 3. Train from Scratch
Run the following command to train from scratch. This may produce scores different to the paper due to randomness in your local environment.
```
(hyperhypo)$ python -m src.train
```

## Citation
```
@inproceedings{cui2023jointly,
    title     = {Jointly Predicting Postprandial Hypoglycemia and Hyperglycemia Using Continuous Glucose Monitoring Data in Type 1 Diabetes},
    author    = {Cui, Ran and Nolan, Christopher J and Daskalaki, Elena and Suominen, Hanna},
    booktitle = {2023 45th Annual International Conference of the IEEE Engineering in Medicine \& Biology Society (EMBC)},
    year      = {2023}
}
```
