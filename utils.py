import pandas as pd

from sklearn.metrics import roc_auc_score, roc_curve

def accuracy_score(y_test,y_pred):
    roc_auc = roc_auc_score(y_test, y_pred)

    return roc_auc

from datetime import datetime

def export_pred(y_pred):
    dir = './_sub_csv'
    current_time = datetime.now().strftime("%m%d_%H%M%S")
    file = f'{dir}/submission_{current_time}.csv'

    pd.DataFrame({'ID': range(1, len(y_pred) + 1), 'Prediction': y_pred}).to_csv(f"{file}",index=None)

from sklearn.preprocessing import LabelEncoder

def ecode_label(X_train,X_test):
    data=pd.concat([X_train,X_test])
    num_train = X_train.shape[0]
    encoder=LabelEncoder()
    for col in data.select_dtypes('object').columns: 
        data[col] = encoder.fit_transform(data[col])

    X_train= data[:num_train]
    X_test=data[num_train:]

    return X_train, X_test

from sklearn.preprocessing import OneHotEncoder

def ecode_onehot(X_train,X_test):
    data=pd.concat([X_train,X_test])
    num_train = X_train.shape[0]

    data = pd.get_dummies(data)

    X_train= data[:num_train]
    X_test=data[num_train:]

    return X_train, X_test

import os
from dotenv import load_dotenv
import numpy as np

def load_train():
    load_dotenv()

    DATA = os.getenv('DATA')
    train_csv = os.getenv('train_csv')
    label = os.getenv('label')

    df = pd.read_csv(f"{DATA}/{train_csv}")

    y_train=df[label]
    X_train=df.drop(label,axis=1)

    return X_train, y_train

def load_test():
    load_dotenv()

    DATA = os.getenv('DATA')
    test_csv = os.getenv('test_csv')

    X_test = pd.read_csv(f"{DATA}/{test_csv}").drop(columns='ID')

    return X_test