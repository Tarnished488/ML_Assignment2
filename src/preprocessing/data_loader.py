import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = r"C:\Users\yqy08\Desktop\数据挖掘和机器学习\Assignment\Assignment 2\comp-3027-j-assignment-2-bdic-2026"


def load_labeled_data(data_dir=None):
    """Load labeled training data (features + labels)."""
    if data_dir is None:
        data_dir = DATA_DIR
    X = pd.read_csv(f"{data_dir}/train_labeled_features.csv")
    y = pd.read_csv(f"{data_dir}/train_labeled_labels.csv").values.ravel()
    return X.values, y


def load_unlabeled_data(data_dir=None):
    """Load unlabeled training features."""
    if data_dir is None:
        data_dir = DATA_DIR
    X = pd.read_csv(f"{data_dir}/train_unlabeled_features.csv")
    return X.values


def load_test_data(data_dir=None):
    """Load test features (with Id column)."""
    if data_dir is None:
        data_dir = DATA_DIR
    df = pd.read_csv(f"{data_dir}/test_features.csv")
    ids = df["Id"].values
    X = df.drop(columns=["Id"]).values
    return ids, X


def get_train_val_split(val_size=0.2, random_state=42, data_dir=None):
    """Split labeled data into train/val sets and apply normalization."""
    X, y = load_labeled_data(data_dir)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    return X_train, X_val, y_train, y_val, scaler


def load_all(data_dir=None):
    """Load all data: labeled, unlabeled, and test."""
    X_labeled, y_labeled = load_labeled_data(data_dir)
    X_unlabeled = load_unlabeled_data(data_dir)
    ids_test, X_test = load_test_data(data_dir)

    scaler = StandardScaler()
    X_labeled = scaler.fit_transform(X_labeled)
    X_unlabeled = scaler.transform(X_unlabeled)
    X_test = scaler.transform(X_test)

    return X_labeled, y_labeled, X_unlabeled, ids_test, X_test, scaler
