import numpy as np
import pandas as pd
from typing import Tuple
from matplotlib import pyplot as plt


def get_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # load the data
    train_df = pd.read_csv('data/mnist_train.csv')
    test_df = pd.read_csv('data/mnist_test.csv')

    X_train = train_df.drop('label', axis=1).values
    y_train = train_df['label'].values

    X_test = test_df.drop('label', axis=1).values
    y_test = test_df['label'].values

    return X_train, X_test, y_train, y_test


def normalize(X_train, X_test) -> Tuple[np.ndarray, np.ndarray]:
    X_train = -1+((2.*X_train)/255)
    X_test = -1+((2.*X_test)/255)
    return X_train,X_test


def plot_metrics(metrics) -> None:
    metricsDF=pd.DataFrame(metrics,columns=["K","Accuracy","Precesion","Recall","F1 score"])
    metricsDF.plot(y=[1,2,3,4],x=0,kind="bar",figsize=(10, 6))
    