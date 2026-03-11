import os
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'

import numpy as np
import pandas as pd

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

#######################
## Autoencoder Pretraining
#######################

pretrained_encoders_w_base = pretrain_autoencoders(X_train_blocks_w, max_iter=500, batch_size=32, corruption_level=0.3)
pretrained_encoders_w_b = pretrain_autoencoders(X_train_blocks_w, X_train_blocks_b, max_iter=500, batch_size=32, corruption_level=0.3)
pretrained_encoders_w_a = pretrain_autoencoders(X_train_blocks_w, X_train_blocks_a, max_iter=500, batch_size=32, corruption_level=0.3)

#######################
## Helper funcitons
#######################

def get_feature_names(X_train_blocks, expert_names):
    feature_names_combined = []
    for name in expert_names:
        block = X_train_blocks[name]
        feature_names_combined.extend(block.columns.tolist())
    return feature_names_combined


def compute_class_weights(y_train):
    neg, pos = np.sum(y_train == 0), np.sum(y_train == 1)
    total = neg + pos
    return {
        0: total / (2.0 * neg),
        1: total / (2.0 * pos)
    }

focal = BinaryFocalCrossentropy(gamma=2.0, alpha=0.95)

#######################
## Stage 1: Source Training (White → Black)
## Train White architecture with White+Black encoders, save weights for Black target
#######################

source_w_b_model = build_moe_model_adaptive_gates(X_train_blocks_w, pretrained_encoders_w_b, loss_fn=focal, seed=42, learning_rate=1e-4)

early_stopping = EarlyStopping(monitor='val_auc', patience=10, mode='max', restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=5, mode='max', min_lr=1e-7, verbose=1)

train_moe_model(
    source_w_b_model, X_train_blocks_w, White_y_train, X_val_blocks_w, White_y_val,
    epochs=50, batch_size=256, verbose=1, callbacks=[early_stopping, lr_scheduler], class_weight=None
)

source_w_b_model.save_weights('model_w_b.weights.h5')
print("Saved source model weights to model_w_b.weights.h5")

#######################
## Stage 1: Source Training (White → Asian)
## Train White architecture with White+Asian encoders, save weights for Asian target
#######################

source_w_a_model = build_moe_model_adaptive_gates(X_train_blocks_w, pretrained_encoders_w_a, loss_fn=focal, seed=42, learning_rate=1e-4)

early_stopping = EarlyStopping(monitor='val_auc', patience=10, mode='max', restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=5, mode='max', min_lr=1e-7, verbose=1)

train_moe_model(
    source_w_a_model, X_train_blocks_w, White_y_train, X_val_blocks_w, White_y_val,
    epochs=50, batch_size=256, verbose=1, callbacks=[early_stopping, lr_scheduler], class_weight=None
)

source_w_a_model.save_weights('model_w_a.weights.h5')
print("Saved source model weights to model_w_a.weights.h5")

#######################
## Stage 2: Target Fine-tuning (Black)
## Build Black MoE with White+Black encoders, load source weights, fine-tune on Black data
#######################

black_adaptive_moe_ae_model = build_moe_model_adaptive_gates(X_train_blocks_b, pretrained_encoders_w_b, seed=42, learning_rate=1e-6)
black_adaptive_moe_ae_model.load_weights('model_w_b.weights.h5', skip_mismatch=True)

early_stopping = EarlyStopping(monitor='val_auc', patience=15, mode='max', restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=5, mode='max', min_lr=1e-7, verbose=1)

class_weights = compute_class_weights(Black_y_train)

black_adaptive_moe_ae_model_history = train_moe_model(
    black_adaptive_moe_ae_model, X_train_blocks_b, Black_y_train, X_val_blocks_b, Black_y_val,
    epochs=50, batch_size=16, verbose=1, callbacks=[early_stopping, lr_scheduler], class_weight=class_weights
)

black_adaptive_moe_ae_model.save_weights('black_adaptive_moe_ae_model.weights.h5')

# Predict and evaluate
y_test_probs = black_adaptive_moe_ae_model.predict(list(X_test_blocks_b.values()), verbose=0).ravel()
black_metrics = evaluate_metrics(Black_y_test, y_test_probs)

ci_lower, ci_upper, bootstrap_aucs = bootstrap_auc_ci(Black_y_test.values.flatten(), y_test_probs)
black_metrics['auc_lower_ci'] = ci_lower
black_metrics['auc_upper_ci'] = ci_upper
black_metrics['bootstrap_aucs'] = float(f"{bootstrap_aucs.mean():.4f}")

print(f"Test Metrics: {black_metrics}")

# Explainability
X_combined_train_b = np.hstack([X_train_blocks_b[expert] for expert in expert_names])
X_combined_test_b = np.hstack([X_test_blocks_b[expert] for expert in expert_names])
feature_names_combined = get_feature_names(X_train_blocks_b, expert_names)

wrapper_model_b = build_moe_wrapper_model(
    moe_model=black_adaptive_moe_ae_model,
    X_train_blocks=X_train_blocks_b,
    expert_names=expert_names,
    wrapper_name="black_adaptive_moe_wrapper_test"
)

results_black = apply_shap_ig_explainability(
    model=wrapper_model_b,
    X_train=X_combined_train_b,
    y_train=Black_y_train,
    X_test=X_combined_test_b,
    y_test=Black_y_test,
    feature_names=feature_names_combined,
    category_name='black_adaptive_moe_test',
    feature_categories_df=feature_categories_df
)

# Save predictions
best_threshold = black_metrics['threshold']
y_pred_test = (y_test_probs >= best_threshold).astype(int)
test_ids = Black_X_test['eid'] if 'eid' in Black_X_test.columns else Black_X_test.index

save_predictions_csv(
    y_true=Black_y_test, y_pred=y_pred_test, y_probs=y_test_probs,
    ids=test_ids, output_path='black_moe_predictions_test.csv'
)

#######################
## Stage 2: Target Fine-tuning (Asian)
## Build Asian MoE with White+Asian encoders, load source weights, fine-tune on Asian data
#######################

asian_adaptive_moe_ae_model = build_moe_model_adaptive_gates(X_train_blocks_a, pretrained_encoders_w_a, seed=42, learning_rate=1e-6)
asian_adaptive_moe_ae_model.load_weights('model_w_a.weights.h5', skip_mismatch=True)

early_stopping = EarlyStopping(monitor='val_auc', patience=15, mode='max', restore_best_weights=True, verbose=1)
lr_scheduler = ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=5, mode='max', min_lr=1e-7, verbose=1)

class_weights = compute_class_weights(Asian_y_train)

asian_adaptive_moe_ae_model_history = train_moe_model(
    asian_adaptive_moe_ae_model, X_train_blocks_a, Asian_y_train, X_val_blocks_a, Asian_y_val,
    epochs=50, batch_size=16, verbose=1, callbacks=[early_stopping, lr_scheduler], class_weight=class_weights
)

asian_adaptive_moe_ae_model.save_weights('asian_adaptive_moe_ae_model.weights.h5')

# Predict and evaluate
y_test_probs = asian_adaptive_moe_ae_model.predict(list(X_test_blocks_a.values()), verbose=0).ravel()
asian_metrics = evaluate_metrics(Asian_y_test, y_test_probs)

ci_lower, ci_upper, bootstrap_aucs = bootstrap_auc_ci(Asian_y_test.values.flatten(), y_test_probs)
asian_metrics['auc_lower_ci'] = ci_lower
asian_metrics['auc_upper_ci'] = ci_upper
asian_metrics['bootstrap_aucs'] = float(f"{bootstrap_aucs.mean():.4f}")

print(f"Test Metrics: {asian_metrics}")

# Explainability
X_combined_train_a = np.hstack([X_train_blocks_a[expert] for expert in expert_names])
X_combined_test_a = np.hstack([X_test_blocks_a[expert] for expert in expert_names])
feature_names_combined = get_feature_names(X_train_blocks_a, expert_names)

wrapper_model_a = build_moe_wrapper_model(
    moe_model=asian_adaptive_moe_ae_model,
    X_train_blocks=X_train_blocks_a,
    expert_names=expert_names,
    wrapper_name="asian_adaptive_moe_wrapper_test"
)

results_asian = apply_shap_ig_explainability(
    model=wrapper_model_a,
    X_train=X_combined_train_a,
    y_train=Asian_y_train,
    X_test=X_combined_test_a,
    y_test=Asian_y_test,
    feature_names=feature_names_combined,
    category_name='asian_adaptive_moe_test',
    feature_categories_df=feature_categories_df
)

# Save predictions
best_threshold = asian_metrics['threshold']
y_pred_test = (y_test_probs >= best_threshold).astype(int)
test_ids = Asian_X_test['eid'] if 'eid' in Asian_X_test.columns else Asian_X_test.index

save_predictions_csv(
    y_true=Asian_y_test, y_pred=y_pred_test, y_probs=y_test_probs,
    ids=test_ids, output_path='asian_moe_predictions_test.csv'
)

#######################
##White Base Model
## Train White MoE with White-only encoders (no transfer), evaluate on White test set
#######################
    
white_adaptive_moe_ae_model = build_moe_model_adaptive_gates(X_train_blocks_w, pretrained_encoders_w_base, loss_fn=focal, seed=42, learning_rate=1e-4)

early_stopping = EarlyStopping(monitor='val_auc', patience=10, mode='max', restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=5, mode='max', min_lr=1e-7, verbose=1)

white_adaptive_moe_ae_model_history = train_moe_model(
    white_adaptive_moe_ae_model, X_train_blocks_w, White_y_train, X_val_blocks_w, White_y_val,
    epochs=50, batch_size=256, verbose=1, callbacks=[early_stopping, lr_scheduler], class_weight=None
)

white_adaptive_moe_ae_model.save_weights('white_adaptive_moe_ae_model.weights.h5')

# Predict and evaluate
y_test_probs = white_adaptive_moe_ae_model.predict(list(X_test_blocks_w.values()), verbose=0).ravel()
white_metrics = evaluate_metrics(White_y_test, y_test_probs)

ci_lower, ci_upper, bootstrap_aucs = bootstrap_auc_ci(White_y_test.values.flatten(), y_test_probs)
white_metrics['auc_lower_ci'] = ci_lower
white_metrics['auc_upper_ci'] = ci_upper
white_metrics['bootstrap_aucs'] = float(f"{bootstrap_aucs.mean():.4f}")

print(f"Test Metrics: {white_metrics}")

# Explainability
X_combined_train_w = np.hstack([X_train_blocks_w[expert] for expert in expert_names])
X_combined_test_w = np.hstack([X_test_blocks_w[expert] for expert in expert_names])
feature_names_combined = get_feature_names(X_train_blocks_w, expert_names)

wrapper_model_w = build_moe_wrapper_model(
    moe_model=white_adaptive_moe_ae_model,
    X_train_blocks=X_train_blocks_w,
    expert_names=expert_names,
    wrapper_name="white_adaptive_moe_wrapper_test"
)

results_white = apply_shap_ig_explainability(
    model=wrapper_model_w,
    X_train=X_combined_train_w,
    y_train=White_y_train,
    X_test=X_combined_test_w,
    y_test=White_y_test,
    feature_names=feature_names_combined,
    category_name='white_adaptive_moe_test',
    feature_categories_df=feature_categories_df
)

# Save predictions
best_threshold = white_metrics['threshold']
y_pred_test = (y_test_probs >= best_threshold).astype(int)
test_ids = White_X_test['eid'] if 'eid' in White_X_test.columns else White_X_test.index

save_predictions_csv(
    y_true=White_y_test, y_pred=y_pred_test, y_probs=y_test_probs,
    ids=test_ids, output_path='white_moe_predictions_test.csv'
)

#######################
## Combine All Results
#######################

black_metrics['Race'] = 'Black'
asian_metrics['Race'] = 'Asian'
white_metrics['Race'] = 'White'

all_races_metrics = pd.DataFrame([black_metrics, asian_metrics, white_metrics])

cols = ['Race'] + [c for c in all_races_metrics.columns if c != 'Race']
all_races_metrics = all_races_metrics[cols]

print("\n===== All Races Test Metrics =====")
print(all_races_metrics)

all_races_metrics.to_csv('test_metrics_all_races.csv', index=False)
print("\nSaved combined test metrics to 'test_metrics_all_races.csv'")
