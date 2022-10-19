# Machine-Learning-Resample/Ensemble

## Goal 

The goal of this project is to predict credit risk using data you'd typically see from peer-to-peer lending services. I have used 2 csv's with loan stats and lending data to make my predictions. 

## Resampling and Ensemble learning models

This project includes 2 Jupyter lab notebooks that include several machine learning models. The credit_risk_ensemble.ipynb explores the Random Forest classifier and the Ensemble classifier models while the credit_risk_resampling.ipynb explores Logistic Regression, SMOTE, SMOTEEN, and CLuster Centroid models. The metrics I have used to evaluate each model include: A balanced accuracy score, confusion matrix, and classification report imbalanced. 


## Software and imports needed to run these notebooks which are written in Python3

* import numpy as np
* import pandas as pd
* from pathlib import Path
* from collections import Counter
* from matplotlib import pyplot as plt
* from sklearn.metrics import balanced_accuracy_score
* from sklearn.metrics import confusion_matrix
* from imblearn.metrics import classification_report_imbalanced
* from sklearn.preprocessing import LabelEncoder, StandardScaler
* from sklearn.model_selection import train_test_split
* from sklearn.ensemble import RandomForestClassifier
* from imblearn.ensemble import EasyEnsembleClassifier
* from sklearn.linear_model import LogisticRegression
* from imblearn.over_sampling import RandomOverSampler, SMOTE
* from imblearn.under_sampling import ClusterCentroids
* from imblearn.combine import SMOTEENN


## Conclusion

The credit_risk_resampling.ipynb proved to have better predictive models for the data used. I would not currently use the credit_risk_ensemble.ipynb in production due to its poor predictions. In consideration of improving the Ensemble models, I would start with reducing the least important features.   
