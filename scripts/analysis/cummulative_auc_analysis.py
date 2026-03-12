import numpy as np
import pandas as pd
from tensorflow.keras.losses import BinaryFocalCrossentropy

from utils import set_seeds, load_data_splits, extract_blocks, bootstrap_cummulative_auc
from model import pretrain_autoencoders, build_moe_model_adaptive_gates

set_seeds(42)

#######################
## Load Data
#######################

feature_categories_df = pd.read_csv('../New_Data/final_variable_info.csv')
feature_map = feature_categories_df.rename(columns={"colname_match_data": "Feature"})
feature_map['Category'] = feature_map['Category'].str.replace(' ', '_').str.lower()
feature_map['Feature'] = feature_map['Feature'].str.strip()

White_X_train, White_y_train, White_X_val, White_y_val, White_X_test, White_y_test = load_data_splits('../standardized_data/white', drop_cols=['eid', 'Race'])
Black_X_train, Black_y_train, Black_X_val, Black_y_val, Black_X_test, Black_y_test = load_data_splits('../standardized_data/black', drop_cols=['eid', 'Race'])
Asian_X_train, Asian_y_train, Asian_X_val, Asian_y_val, Asian_X_test, Asian_y_test = load_data_splits('../standardized_data/asian', drop_cols=['eid', 'Race'])

X_train_blocks_w = extract_blocks(White_X_train, feature_map)
X_train_blocks_b = extract_blocks(Black_X_train, feature_map)
X_train_blocks_a = extract_blocks(Asian_X_train, feature_map)

expert_names = list(X_train_blocks_w.keys())

#######################
## Pretrain Autoencoders
#######################

pretrained_encoders_w_base = pretrain_autoencoders(X_train_blocks_w, max_iter=500, batch_size=32, corruption_level=0.3)
pretrained_encoders_w_b    = pretrain_autoencoders(X_train_blocks_w, X_train_blocks_b, max_iter=500, batch_size=32, corruption_level=0.3)
pretrained_encoders_w_a    = pretrain_autoencoders(X_train_blocks_w, X_train_blocks_a, max_iter=500, batch_size=32, corruption_level=0.3)

#######################
## Load Models
#######################

focal = BinaryFocalCrossentropy(gamma=2.0, alpha=0.95)

moe_model_w_base = build_moe_model_adaptive_gates(X_train_blocks_w, pretrained_encoders_w_base, loss_fn=focal, seed=42, learning_rate=1e-4)
moe_model_w_base.load_weights('model_weights/white_adaptive_moe_ae_model.weights.h5', skip_mismatch=True)

black_adaptive_moe_ae_model = build_moe_model_adaptive_gates(X_train_blocks_b, pretrained_encoders_w_b, seed=42, learning_rate=1e-6)
black_adaptive_moe_ae_model.load_weights('model_weights/black_adaptive_moe_ae_model.weights.h5', skip_mismatch=True)

asian_adaptive_moe_ae_model = build_moe_model_adaptive_gates(X_train_blocks_a, pretrained_encoders_w_a, seed=42, learning_rate=1e-6)
asian_adaptive_moe_ae_model.load_weights('model_weights/asian_adaptive_moe_ae_model.weights.h5', skip_mismatch=True)

#######################
## Cumulative AUC
#######################

def compute_cumulative_auc(model, X_test_blocks_ordered, ordered_feature_ids, y_test, expert_names, n_boot=400, top_k=50):
    masked_blocks = {name: np.zeros_like(X_test_blocks_ordered[name]) for name in expert_names}
    cumulative_aucs = []
    cumulative_aucs_ci = []

    for k in range(1, top_k + 1):
        top_k_ids = set(ordered_feature_ids[:k])

        for name in expert_names:
            block_df = X_test_blocks_ordered[name]
            use_feats = [fid for fid in block_df.columns if fid in top_k_ids]
            masked_df = pd.DataFrame(0, index=block_df.index, columns=block_df.columns)
            masked_df[use_feats] = block_df[use_feats]
            masked_blocks[name] = masked_df.values

        x_input = [masked_blocks[name] for name in expert_names]
        mean_auc, (ci_low, ci_high) = bootstrap_cummulative_auc(model, x_input, y_test, n_boot=n_boot)

        cumulative_aucs.append(mean_auc)
        cumulative_aucs_ci.append((ci_low, ci_high))

    return cumulative_aucs, cumulative_aucs_ci


def save_cumulative_auc_results(feature_df, cumulative_aucs, cumulative_aucs_ci, output_file, top_k=50):
    lower_bounds, upper_bounds = zip(*cumulative_aucs_ci)

    feature_df['Cumulative_AUC'] = np.nan
    feature_df['Lower_95CI']     = np.nan
    feature_df['Upper_95CI']     = np.nan

    feature_df.loc[:top_k - 1, 'Cumulative_AUC'] = cumulative_aucs[:top_k]
    feature_df.loc[:top_k - 1, 'Lower_95CI']     = list(lower_bounds[:top_k])
    feature_df.loc[:top_k - 1, 'Upper_95CI']     = list(upper_bounds[:top_k])

    feature_df.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")


datasets = [
    ('White', moe_model_w_base,           White_X_test, White_y_test, 'final_white_all_features.csv', 'white_features_with_cumulative_auc_50.csv', 100),
    ('Black', black_adaptive_moe_ae_model, Black_X_test, Black_y_test, 'final_black_all_features.csv', 'black_features_with_cumulative_auc_50.csv', 400),
    ('Asian', asian_adaptive_moe_ae_model, Asian_X_test, Asian_y_test, 'final_asian_all_features.csv', 'asian_features_with_cumulative_auc_50.csv', 400),
]

for race, model, X_test, y_test, feature_file, output_file, n_boot in datasets:
    print(f"\n========== {race} ==========")

    feature_df = pd.read_csv(feature_file)
    ordered_feature_ids = feature_df['Feature'].tolist()

    X_test_ordered = X_test[ordered_feature_ids].copy()
    X_test_blocks_ordered = extract_blocks(X_test_ordered, feature_map)

    cumulative_aucs, cumulative_aucs_ci = compute_cumulative_auc(
        model, X_test_blocks_ordered, ordered_feature_ids, y_test.values, expert_names, n_boot=n_boot
    )
    save_cumulative_auc_results(feature_df, cumulative_aucs, cumulative_aucs_ci, output_file)
