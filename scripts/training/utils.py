import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from typing import Tuple
from sklearn.metrics import (
    f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score
)
from sklearn.utils import resample


def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)


## Load Data

def load_data_splits(prefix, drop_cols=["eid", "Race"]):
    """
    Load X/y train/val/test CSV files using a prefix and
    optionally drop specified columns from X sets.
    """
    X_train = pd.read_csv(f'{prefix}_X_train.csv')
    y_train = pd.read_csv(f'{prefix}_y_train.csv')

    X_val = pd.read_csv(f'{prefix}_X_val.csv')
    y_val = pd.read_csv(f'{prefix}_y_val.csv')

    X_test = pd.read_csv(f'{prefix}_X_test.csv')
    y_test = pd.read_csv(f'{prefix}_y_test.csv')

    if drop_cols is not None:
        X_train = X_train.drop(columns=drop_cols, errors="ignore")
        X_val   = X_val.drop(columns=drop_cols, errors="ignore")
        X_test  = X_test.drop(columns=drop_cols, errors="ignore")
        y_train = y_train.drop(columns=drop_cols, errors="ignore")
        y_val   = y_val.drop(columns=drop_cols, errors="ignore")
        y_test  = y_test.drop(columns=drop_cols, errors="ignore")

    return X_train, y_train, X_val, y_val, X_test, y_test


def extract_blocks(X_df, feature_map):
    """
    Extract feature blocks from a DataFrame based on category mappings.

    Parameters:
        X_df: DataFrame with features.
        feature_map: DataFrame with 'Feature' and 'Category' columns.

    Returns:
        dict: Feature blocks keyed by category name.
    """
    blocks = {}

    for super_cat in feature_map['Category'].unique():
        expert_cols = feature_map[feature_map['Category'] == super_cat]['Feature'].tolist()
        expert_cols = [f for f in expert_cols if f in X_df.columns]
        blocks[super_cat] = X_df[expert_cols]

    return blocks


def save_predictions_csv(y_true, y_pred, y_probs, ids, output_path, fold_num=None):
    """
    Saves all test predictions to a CSV file, indicating whether they were correct or not.
    """
    y_true_arr = np.array(y_true).flatten()
    y_pred_arr = np.array(y_pred).flatten()
    y_probs_arr = np.array(y_probs).flatten()
    ids_arr = np.array(ids).flatten()

    is_correct = (y_true_arr == y_pred_arr)

    df = pd.DataFrame({
        'ID': ids_arr,
        'True_Label': y_true_arr,
        'Predicted_Label': y_pred_arr,
        'Predicted_Probability': y_probs_arr,
        'Is_Correct': is_correct
    })

    if fold_num is not None:
        df.insert(0, 'Fold', fold_num)

    file_exists = os.path.isfile(output_path)
    df.to_csv(output_path, mode='a', header=not file_exists, index=False)
    print(f"Saved {len(df)} predictions to {output_path}")


## Evaluate metrics

def evaluate_metrics(y_true, y_probs):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]

    y_pred = (y_probs >= best_thresh).astype(int)

    return {
        "auc": roc_auc_score(y_true, y_probs),
        "f1": f1_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "threshold": best_thresh
    }


## Bootstrap AUC CI

def bootstrap_auc_ci(y_true: np.ndarray, y_pred: np.ndarray,
                     n_bootstraps: int = 1000, confidence_level: float = 0.95,
                     random_state: int = 42) -> Tuple[float, float, np.ndarray]:
    """
    Calculate bootstrap confidence interval for AUC.

    Parameters:
    -----------
    y_true : array True binary labels
    y_pred : array Predicted probabilities or scores
    n_bootstraps : int Number of bootstrap iterations
    confidence_level : float Confidence level (default 0.95 for 95% CI)
    random_state : int Random seed for reproducibility

    Returns:
    --------
    lower_ci : float Lower confidence bound
    upper_ci : float Upper confidence bound
    bootstrap_aucs : array Array of bootstrap AUC values
    """
    np.random.seed(random_state)
    n_samples = len(y_true)
    bootstrap_aucs = []

    for i in range(n_bootstraps):
        indices = resample(np.arange(n_samples), replace=True,
                          n_samples=n_samples, random_state=random_state + i)

        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]

        if len(np.unique(y_true_boot)) < 2:
            continue

        try:
            auc_boot = roc_auc_score(y_true_boot, y_pred_boot)
            bootstrap_aucs.append(auc_boot)
        except Exception:
            continue

    bootstrap_aucs = np.array(bootstrap_aucs)

    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_ci = np.percentile(bootstrap_aucs, lower_percentile)
    upper_ci = np.percentile(bootstrap_aucs, upper_percentile)

    return lower_ci, upper_ci, bootstrap_aucs
