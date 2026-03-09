import os
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

from tensorflow.keras.losses import BinaryFocalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from utils import set_seeds, load_data_splits, extract_blocks, save_predictions_csv, evaluate_metrics, bootstrap_auc_ci
from model import pretrain_autoencoders, build_moe_model_adaptive_gates, train_moe_model, build_moe_wrapper_model
from explainability import apply_shap_ig_explainability

set_seeds(42)

#######################
## Load Data
#######################

feature_categories_df = pd.read_csv('../New_Data/final_variable_info.csv')

# Load feature mappings
feature_map = feature_categories_df.rename(columns={"colname_match_data": "Feature"})
feature_map['Category'] = feature_map['Category'].str.replace(' ', '_').str.lower()
feature_map['Feature'] = feature_map['Feature'].str.strip()

feature_categories_df = feature_map.copy()

# Load race-specific data
White_X_train, White_y_train, White_X_val, White_y_val, White_X_test, White_y_test = load_data_splits('data/standardized_data/white', drop_cols=['Race'])
Black_X_train, Black_y_train, Black_X_val, Black_y_val, Black_X_test, Black_y_test = load_data_splits('data/standardized_data/black', drop_cols=['Race'])
Asian_X_train, Asian_y_train, Asian_X_val, Asian_y_val, Asian_X_test, Asian_y_test = load_data_splits('data/standardized_data/asian', drop_cols=['Race'])

# Drop 'eid' from y dataframes
for y_df in [White_y_train, White_y_val, White_y_test,
             Black_y_train, Black_y_val, Black_y_test,
             Asian_y_train, Asian_y_val, Asian_y_test]:
    if 'eid' in y_df.columns:
        y_df.drop(columns=['eid'], inplace=True, errors='ignore')

# Extract feature blocks per race
X_train_blocks_w = extract_blocks(White_X_train, feature_map)
X_val_blocks_w   = extract_blocks(White_X_val, feature_map)
X_test_blocks_w  = extract_blocks(White_X_test, feature_map)

X_train_blocks_b = extract_blocks(Black_X_train, feature_map)
X_val_blocks_b   = extract_blocks(Black_X_val, feature_map)
X_test_blocks_b  = extract_blocks(Black_X_test, feature_map)

X_train_blocks_a = extract_blocks(Asian_X_train, feature_map)
X_val_blocks_a   = extract_blocks(Asian_X_val, feature_map)
X_test_blocks_a  = extract_blocks(Asian_X_test, feature_map)

expert_names = list(X_train_blocks_w.keys())

# Combine train + val for CV
White_X_cv = pd.concat([White_X_train, White_X_val])
White_y_cv = pd.concat([White_y_train, White_y_val])

Black_X_cv = pd.concat([Black_X_train, Black_X_val])
Black_y_cv = pd.concat([Black_y_train, Black_y_val])

Asian_X_cv = pd.concat([Asian_X_train, Asian_X_val])
Asian_y_cv = pd.concat([Asian_y_train, Asian_y_val])

X_train_blocks_w_full = extract_blocks(White_X_cv, feature_map)
X_train_blocks_b_full = extract_blocks(Black_X_cv, feature_map)
X_train_blocks_a_full = extract_blocks(Asian_X_cv, feature_map)

#######################
## Helper Functions
#######################

def get_feature_names(X_train_blocks, expert_names):
    feature_names_combined = []
    for name in expert_names:
        feature_names_combined.extend(X_train_blocks[name].columns.tolist())
    return feature_names_combined


def compute_class_weights(y_train):
    neg, pos = np.sum(y_train == 0), np.sum(y_train == 1)
    total = neg + pos
    return {
        0: total / (2.0 * neg),
        1: total / (2.0 * pos)
    }


focal = BinaryFocalCrossentropy(gamma=2.0, alpha=0.95)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

### 5 FOLD CV

########
# 5-Fold Stratified CV on Black_X_cv (Transfer Learning)
########

black_all_metrics = []

for fold, (train_idx, val_idx) in enumerate(skf.split(Black_X_cv, Black_y_cv)):

    print(f"\n========== Fold {fold+1} ==========")

    # Slice data and labels
    X_train_fold = Black_X_cv.iloc[train_idx]
    X_val_fold   = Black_X_cv.iloc[val_idx]
    y_train_fold = Black_y_cv.iloc[train_idx]
    y_val_fold = Black_y_cv.iloc[val_idx]

    # Extract blocks for this fold
    X_train_blocks_fold = extract_blocks(X_train_fold, feature_map)
    X_val_blocks_fold   = extract_blocks(X_val_fold, feature_map)

    pretrained_encoders_w_b = pretrain_autoencoders(X_train_blocks_w_full, X_train_blocks_fold, max_iter=500, batch_size=32, corruption_level=0.3)

    model = build_moe_model_adaptive_gates(X_train_blocks_fold, pretrained_encoders_w_b, seed=42, learning_rate=1e-6)
    model.load_weights('model_w_b.weights.h5', skip_mismatch=True)

    class_weights = compute_class_weights(y_train_fold)

    early_stopping = EarlyStopping(monitor='val_auc', patience=15, mode='max', restore_best_weights=True, verbose=1)
    lr_scheduler = ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=5, mode='max', min_lr=1e-7, verbose=1)

    history = train_moe_model(
        model, X_train_blocks_fold, y_train_fold, X_val_blocks_fold, y_val_fold, epochs=50, batch_size=16, verbose=1, 
        callbacks=[early_stopping, lr_scheduler], class_weight=class_weights)

    y_val_probs = model.predict(list(X_val_blocks_fold.values()), verbose=0).ravel()
    metrics = evaluate_metrics(y_val_fold, y_val_probs)
    ci_lower, ci_upper, _ = bootstrap_auc_ci(y_val_fold.values.flatten(), y_val_probs)
    metrics['auc_lower_ci'] = ci_lower
    metrics['auc_upper_ci'] = ci_upper

    print(f"Fold {fold+1} Metrics: {metrics}")
    black_all_metrics.append(metrics)

    # Explainability
    X_combined_train_b = np.hstack([X_train_blocks_fold[expert] for expert in expert_names])
    X_combined_val_b = np.hstack([X_val_blocks_fold[expert] for expert in expert_names])
    feature_names_combined = get_feature_names(X_train_blocks_fold, expert_names)

    wrapper_model_b = build_moe_wrapper_model(
        moe_model=model, 
        X_train_blocks=X_train_blocks_fold, 
        expert_names=expert_names, 
        wrapper_name=f"black_adaptive_moe_wrapper_fold_{fold+1}"
    )
    
    results_black = apply_shap_ig_explainability(
        model=wrapper_model_b,
        X_train=X_combined_train_b, 
        y_train=y_train_fold, 
        X_test=X_combined_val_b, 
        y_test=y_val_fold, 
        feature_names=feature_names_combined, 
        category_name=f'black_adaptive_moe_fold_{fold+1}', 
        feature_categories_df=feature_categories_df
    )

df_metrics = pd.DataFrame(black_all_metrics)
print("\n===== Fold-wise Metrics (Black) =====")
print(df_metrics[['auc', 'auc_lower_ci', 'auc_upper_ci']])
black_summary = df_metrics.agg(['mean', 'std'])
print("\n===== Cross-Validation Summary =====")
print(black_summary)


########
# 5-Fold Stratified CV on Asian_X_cv (Transfer Learning)
########

asian_all_metrics = []

for fold, (train_idx, val_idx) in enumerate(skf.split(Asian_X_cv, Asian_y_cv)):

    print(f"\n========== Fold {fold+1} ==========")

    # Slice data and labels
    X_train_fold = Asian_X_cv.iloc[train_idx]
    X_val_fold   = Asian_X_cv.iloc[val_idx]
    y_train_fold = Asian_y_cv.iloc[train_idx]
    y_val_fold = Asian_y_cv.iloc[val_idx]

    # Extract blocks for this fold
    X_train_blocks_fold = extract_blocks(X_train_fold, feature_map)
    X_val_blocks_fold   = extract_blocks(X_val_fold, feature_map)

    pretrained_encoders_w_a = pretrain_autoencoders(X_train_blocks_w_full, X_train_blocks_fold, max_iter=500, batch_size=32, corruption_level=0.3)

    # Rebuild model and load pretrained weights
    model = build_moe_model_adaptive_gates(X_train_blocks_fold, pretrained_encoders_w_a, seed=42, learning_rate=1e-6)
    model.load_weights('model_w_a.weights.h5', skip_mismatch=True)

    class_weights = compute_class_weights(y_train_fold)

    early_stopping = EarlyStopping(monitor='val_auc', patience=15, mode='max', restore_best_weights=True, verbose=1)
    lr_scheduler = ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=5, mode='max', min_lr=1e-7, verbose=1)

    history = train_moe_model(model, X_train_blocks_fold, y_train_fold, X_val_blocks_fold, y_val_fold, epochs=50, batch_size=16, verbose=1, callbacks=[early_stopping, lr_scheduler], class_weight=class_weights)

    y_val_probs = model.predict(list(X_val_blocks_fold.values()), verbose=0).ravel()
    metrics = evaluate_metrics(y_val_fold, y_val_probs)
    ci_lower, ci_upper, _ = bootstrap_auc_ci(y_val_fold.values.flatten(), y_val_probs)
    metrics['auc_lower_ci'] = ci_lower
    metrics['auc_upper_ci'] = ci_upper

    print(f"Fold {fold+1} Metrics: {metrics}")
    asian_all_metrics.append(metrics)

    # Explainability
    X_combined_train_a = np.hstack([X_train_blocks_fold[expert] for expert in expert_names])
    X_combined_val_a = np.hstack([X_val_blocks_fold[expert] for expert in expert_names])
    feature_names_combined = get_feature_names(X_train_blocks_fold, expert_names)

    wrapper_model_a = build_moe_wrapper_model(
        moe_model=model, 
        X_train_blocks=X_train_blocks_fold, 
        expert_names=expert_names, 
        wrapper_name=f"asian_adaptive_moe_wrapper_fold_{fold+1}"
    )

    results_asian = apply_shap_ig_explainability(
        model=wrapper_model_a,
        X_train=X_combined_train_a, 
        y_train=y_train_fold, 
        X_test=X_combined_val_a, 
        y_test=y_val_fold, 
        feature_names=feature_names_combined, 
        category_name=f'asian_adaptive_moe_fold_{fold+1}', 
        feature_categories_df=feature_categories_df
    )

df_metrics = pd.DataFrame(asian_all_metrics)
print("\n===== Fold-wise Metrics (Asian) =====")
print(df_metrics[['auc', 'auc_lower_ci', 'auc_upper_ci']])
asian_summary = df_metrics.agg(['mean', 'std'])
print("\n===== Cross-Validation Summary =====")
print(asian_summary)


########
# 5-Fold Stratified CV on White_X_cv
########

white_all_metrics = []

for fold, (train_idx, val_idx) in enumerate(skf.split(White_X_cv, White_y_cv)):

    print(f"\n========== Fold {fold+1} ==========")

    # Slice data and labels
    X_train_fold = White_X_cv.iloc[train_idx]
    X_val_fold   = White_X_cv.iloc[val_idx]
    y_train_fold = White_y_cv.iloc[train_idx]
    y_val_fold = White_y_cv.iloc[val_idx]

    # Extract blocks for this fold
    X_train_blocks_fold = extract_blocks(X_train_fold, feature_map)
    X_val_blocks_fold   = extract_blocks(X_val_fold, feature_map)

    pretrained_encoders_w_base = pretrain_autoencoders(X_train_blocks_fold, max_iter=500, batch_size=32, corruption_level=0.3)

    model = build_moe_model_adaptive_gates(X_train_blocks_fold, pretrained_encoders_w_base, loss_fn=focal, seed=42, learning_rate=1e-4)

    early_stopping = EarlyStopping(monitor='val_auc', patience=10, mode='max', restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=5, mode='max', min_lr=1e-7, verbose=1)

    history = train_moe_model(model, X_train_blocks_fold, y_train_fold, X_val_blocks_fold, y_val_fold, epochs=50, batch_size=256, verbose=1, callbacks=[early_stopping, lr_scheduler], class_weight=None)

    y_val_probs = model.predict(list(X_val_blocks_fold.values()), verbose=0).ravel()
    metrics = evaluate_metrics(y_val_fold, y_val_probs)
    ci_lower, ci_upper, _ = bootstrap_auc_ci(y_val_fold.values.flatten(), y_val_probs)
    metrics['auc_lower_ci'] = ci_lower
    metrics['auc_upper_ci'] = ci_upper

    print(f"Fold {fold+1} Metrics: {metrics}")
    white_all_metrics.append(metrics)

    # Explainability
    X_combined_train_w = np.hstack([X_train_blocks_fold[expert] for expert in expert_names])
    X_combined_val_w = np.hstack([X_val_blocks_fold[expert] for expert in expert_names])
    feature_names_combined = get_feature_names(X_train_blocks_fold, expert_names)

    wrapper_model_w = build_moe_wrapper_model(
        moe_model=model, 
        X_train_blocks=X_train_blocks_fold, 
        expert_names=expert_names, 
        wrapper_name=f"white_adaptive_moe_wrapper_fold_{fold+1}"
    )

    results_white = apply_shap_ig_explainability(
        model=wrapper_model_w, 
        X_train=X_combined_train_w, 
        y_train=y_train_fold, 
        X_test=X_combined_val_w, 
        y_test=y_val_fold, 
        feature_names=feature_names_combined, 
        category_name=f'white_adaptive_moe_fold_{fold+1}', 
        feature_categories_df=feature_categories_df
    )
    
df_metrics = pd.DataFrame(white_all_metrics)
print("\n===== Fold-wise Metrics (White) =====")
print(df_metrics[['auc', 'auc_lower_ci', 'auc_upper_ci']])
white_summary = df_metrics.agg(['mean', 'std'])
print("\n===== Cross-Validation Summary =====")
print(white_summary)


## Independent CV summary CSVs

########
# 5-Fold Stratified CV on Asian_X_cv (Independent)
########

asian_indep_metrics = []

for fold, (train_idx, val_idx) in enumerate(skf.split(Asian_X_cv, Asian_y_cv)):

    print(f"\n========== Fold {fold+1} ==========")

    # Slice data and labels
    X_train_fold = Asian_X_cv.iloc[train_idx]
    X_val_fold   = Asian_X_cv.iloc[val_idx]
    y_train_fold = Asian_y_cv.iloc[train_idx]
    y_val_fold   = Asian_y_cv.iloc[val_idx]

    # Extract blocks for this fold
    X_train_blocks_fold = extract_blocks(X_train_fold, feature_map)
    X_val_blocks_fold   = extract_blocks(X_val_fold, feature_map)

    pretrained_encoders_asian = pretrain_autoencoders(X_train_blocks_fold, max_iter=500, batch_size=32, corruption_level=0.3)

    model = build_moe_model_adaptive_gates(X_train_blocks_fold, pretrained_encoders_asian, seed=42, learning_rate=1e-4)

    class_weights = compute_class_weights(y_train_fold)

    early_stopping = EarlyStopping(monitor='val_auc', patience=15, mode='max', restore_best_weights=True, verbose=1)
    lr_scheduler = ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=5, mode='max', min_lr=1e-7, verbose=1)

    history = train_moe_model(model, X_train_blocks_fold, y_train_fold, X_val_blocks_fold, y_val_fold, epochs=50, batch_size=16, verbose=1, callbacks=[early_stopping, lr_scheduler], class_weight=class_weights)

    y_val_probs = model.predict(list(X_val_blocks_fold.values()), verbose=0).ravel()
    metrics = evaluate_metrics(y_val_fold, y_val_probs)
    ci_lower, ci_upper, _ = bootstrap_auc_ci(y_val_fold.values.flatten(), y_val_probs)
    metrics['auc_lower_ci'] = ci_lower
    metrics['auc_upper_ci'] = ci_upper

    print(f"Fold {fold+1} Metrics: {metrics}")
    asian_indep_metrics.append(metrics)

    # Explainability
    X_combined_train_a = np.hstack([X_train_blocks_fold[expert] for expert in expert_names])
    X_combined_val_a = np.hstack([X_val_blocks_fold[expert] for expert in expert_names])
    feature_names_combined = get_feature_names(X_train_blocks_fold, expert_names)

    wrapper_model_a = build_moe_wrapper_model(
        moe_model=model,
        X_train_blocks=X_train_blocks_fold,
        expert_names=expert_names,
        wrapper_name=f"asian_adaptive_moe_wrapper_fold_{fold+1}"
    )

    results_asian = apply_shap_ig_explainability(
        model=wrapper_model_a,
        X_train=X_combined_train_a,
        y_train=y_train_fold,
        X_test=X_combined_val_a,
        y_test=y_val_fold,
        feature_names=feature_names_combined,
        category_name=f'asian_adaptive_moe_fold_{fold+1}',
        feature_categories_df=feature_categories_df
    )

df_metrics = pd.DataFrame(asian_indep_metrics)
print("\n===== Fold-wise Metrics (Asian) =====")
print(df_metrics[['auc', 'auc_lower_ci', 'auc_upper_ci']])
asian_indep_summary = df_metrics.agg(['mean', 'std'])
print("\n===== Cross-Validation Summary =====")
print(asian_indep_summary)


########
# 5-Fold Stratified CV on Black_X_cv (Independent)
########

black_indep_metrics = []

for fold, (train_idx, val_idx) in enumerate(skf.split(Black_X_cv, Black_y_cv)):

    print(f"\n========== Fold {fold+1} ==========")

    # Slice data and labels
    X_train_fold = Black_X_cv.iloc[train_idx]
    X_val_fold   = Black_X_cv.iloc[val_idx]
    y_train_fold = Black_y_cv.iloc[train_idx]
    y_val_fold   = Black_y_cv.iloc[val_idx]

    # Extract blocks for this fold
    X_train_blocks_fold = extract_blocks(X_train_fold, feature_map)
    X_val_blocks_fold   = extract_blocks(X_val_fold, feature_map)

    pretrained_encoders_black = pretrain_autoencoders(X_train_blocks_fold, max_iter=500, batch_size=32, corruption_level=0.3)

    model = build_moe_model_adaptive_gates(X_train_blocks_fold, pretrained_encoders_black, seed=42, learning_rate=1e-4)

    class_weights = compute_class_weights(y_train_fold)

    early_stopping = EarlyStopping(monitor='val_auc', patience=15, mode='max', restore_best_weights=True, verbose=1)
    lr_scheduler = ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=5, mode='max', min_lr=1e-7, verbose=1)

    history = train_moe_model(model, X_train_blocks_fold, y_train_fold, X_val_blocks_fold, y_val_fold, epochs=50, batch_size=16, verbose=1, callbacks=[early_stopping, lr_scheduler], class_weight=class_weights)

    y_val_probs = model.predict(list(X_val_blocks_fold.values()), verbose=0).ravel()
    metrics = evaluate_metrics(y_val_fold, y_val_probs)
    ci_lower, ci_upper, _ = bootstrap_auc_ci(y_val_fold.values.flatten(), y_val_probs)
    metrics['auc_lower_ci'] = ci_lower
    metrics['auc_upper_ci'] = ci_upper

    print(f"Fold {fold+1} Metrics: {metrics}")
    black_indep_metrics.append(metrics)

    # Explainability
    X_combined_train_b = np.hstack([X_train_blocks_fold[expert] for expert in expert_names])
    X_combined_val_b = np.hstack([X_val_blocks_fold[expert] for expert in expert_names])
    feature_names_combined = get_feature_names(X_train_blocks_fold, expert_names)

    wrapper_model_b = build_moe_wrapper_model(
        moe_model=model,
        X_train_blocks=X_train_blocks_fold,
        expert_names=expert_names,
        wrapper_name=f"black_adaptive_moe_wrapper_fold_{fold+1}"
    )

    results_black = apply_shap_ig_explainability(
        model=wrapper_model_b,
        X_train=X_combined_train_b,
        y_train=y_train_fold,
        X_test=X_combined_val_b,
        y_test=y_val_fold,
        feature_names=feature_names_combined,
        category_name=f'black_adaptive_moe_fold_{fold+1}',
        feature_categories_df=feature_categories_df
    )

df_metrics = pd.DataFrame(black_indep_metrics)
print("\n===== Fold-wise Metrics (Black) =====")
print(df_metrics[['auc', 'auc_lower_ci', 'auc_upper_ci']])
black_indep_summary = df_metrics.agg(['mean', 'std'])
print("\n===== Cross-Validation Summary =====")
print(black_indep_summary)


#######################
## Combine Results
#######################

# Transfer learning results
df_black_tl_metrics = pd.DataFrame(black_all_metrics)
df_black_tl_metrics['Race'] = 'Black'
df_black_tl_metrics['Model'] = 'Transfer'
df_black_tl_metrics['Fold'] = range(1, len(df_black_tl_metrics) + 1)

df_asian_tl_metrics = pd.DataFrame(asian_all_metrics)
df_asian_tl_metrics['Race'] = 'Asian'
df_asian_tl_metrics['Model'] = 'Transfer'
df_asian_tl_metrics['Fold'] = range(1, len(df_asian_tl_metrics) + 1)

df_white_tl_metrics = pd.DataFrame(white_all_metrics)
df_white_tl_metrics['Race'] = 'White'
df_white_tl_metrics['Model'] = 'Baseline'
df_white_tl_metrics['Fold'] = range(1, len(df_white_tl_metrics) + 1)

# Independent results
df_black_indep_metrics = pd.DataFrame(black_indep_metrics)
df_black_indep_metrics['Race'] = 'Black'
df_black_indep_metrics['Model'] = 'Independent'
df_black_indep_metrics['Fold'] = range(1, len(df_black_indep_metrics) + 1)

df_asian_indep_metrics = pd.DataFrame(asian_indep_metrics)
df_asian_indep_metrics['Race'] = 'Asian'
df_asian_indep_metrics['Model'] = 'Independent'
df_asian_indep_metrics['Fold'] = range(1, len(df_asian_indep_metrics) + 1)

all_metrics = pd.concat([
    df_black_tl_metrics,
    df_asian_tl_metrics,
    df_white_tl_metrics,
    df_black_indep_metrics,
    df_asian_indep_metrics,
], ignore_index=True)

cols = ['Race', 'Model', 'Fold'] + [col for col in all_metrics.columns if col not in ['Race', 'Model', 'Fold']]
all_metrics = all_metrics[cols]

all_metrics.to_csv('fold_wise_metrics_all.csv', index=False)
print("\nSaved fold-wise metrics for all groups to 'fold_wise_metrics_all.csv'")