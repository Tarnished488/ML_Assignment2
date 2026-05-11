import sys
sys.path.insert(0, '.')

import numpy as np

from src.preprocessing.data_loader import (
    load_labeled_data,
    load_unlabeled_data,
    load_test_data,
    get_train_val_split,
    load_all,
)

print("=" * 50)
print("Testing data loader...")
print("=" * 50)

# 1. labeled data
X_l, y_l = load_labeled_data()
print(f"\n[Labeled]   X: {X_l.shape}, y: {y_l.shape}")
print(f"  Classes: {set(y_l)}, samples per class: {dict(zip(*np.unique(y_l, return_counts=True)))}")

# 2. unlabeled data
X_u = load_unlabeled_data()
print(f"\n[Unlabeled] X: {X_u.shape}")

# 3. test data
ids, X_t = load_test_data()
print(f"\n[Test]      ids: {ids.shape}, X: {X_t.shape}")

# 4. train/val split
X_tr, X_val, y_tr, y_val, scaler = get_train_val_split()
print(f"\n[Split]     train: {X_tr.shape}, val: {X_val.shape}")
print(f"  Scaler mean (first 5): {scaler.mean_[:5]}")

# 5. load_all
X_l2, y_l2, X_u2, ids2, X_t2, s2 = load_all()
print(f"\n[Load All]  labeled: {X_l2.shape}, unlabeled: {X_u2.shape}, test: {X_t2.shape}")
print(f"  All normalized: mean~={X_l2.mean():.6f}, std~={X_l2.std():.6f}")

print("\n" + "=" * 50)
print("All tests passed!")
