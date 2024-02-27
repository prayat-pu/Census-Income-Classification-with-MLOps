import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from model import train_model,compute_model_metrics,inference
from data import process_data
from sklearn.model_selection import train_test_split
import pickle

@pytest.fixture(scope='module')
def data():
    data_path = '../data/census.csv'

    df = pd.read_csv(data_path).drop_duplicates()
    train, test = train_test_split(df, test_size=0.20)

    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    ]

    X_train, y_train, encoder, lb = get_train_data(train,cat_features)

    X_test, y_test = get_test_data(test,cat_features,encoder,lb)


    return X_train, y_train, X_test, y_test

def get_train_data(train,cat_features):

    X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
    )

    return X_train, y_train, encoder, lb

def get_test_data(test,cat_features,encoder,lb):

    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features, label="salary", 
        training=False,encoder=encoder,lb=lb
    )

    return X_test,y_test


def test_train_model(data):
    X_train, y_train, X_test, y_test = data

    model = train_model(X_train,y_train)
    assert type(model)==RandomForestClassifier


def test_compute_model_metrics(data):


    X_train, y_train, X_test, y_test = data

    model = train_model(X_train,y_train)

    y_pred = model.predict(X_test)

    precision, recall, fbeta = compute_model_metrics(y_test,y_pred)

    
    assert precision.dtype == np.float64
    assert recall.dtype == np.float64
    assert fbeta.dtype == np.float64




def test_inference_output(data):
    X_train, y_train, X_test, y_test = data

    model = train_model(X_train,y_train)

    y_pred = model.predict(X_test)

    assert isinstance(y_pred, np.ndarray)