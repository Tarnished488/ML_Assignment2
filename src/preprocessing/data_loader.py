import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Default: project_root/data/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DATA_DIR = str(_PROJECT_ROOT / "data")


def _resolve_data_dir(data_dir=None):
    """Use explicit path, then ASSIGNMENT2_DATA_DIR env var, then project data/."""
    return data_dir or os.environ.get("ASSIGNMENT2_DATA_DIR") or _DEFAULT_DATA_DIR


def load_labeled_data(data_dir=None):
    """Load labeled training data (features + labels)."""
    data_dir = _resolve_data_dir(data_dir)
    X = pd.read_csv(f"{data_dir}/train_labeled_features.csv")
    y = pd.read_csv(f"{data_dir}/train_labeled_labels.csv").values.ravel()
    return X.values, y


def load_unlabeled_data(data_dir=None):
    """Load unlabeled training features."""
    data_dir = _resolve_data_dir(data_dir)
    X = pd.read_csv(f"{data_dir}/train_unlabeled_features.csv")
    return X.values


def load_test_data(data_dir=None):
    """Load test features (with Id column)."""
    data_dir = _resolve_data_dir(data_dir)
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
