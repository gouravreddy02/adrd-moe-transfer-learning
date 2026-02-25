import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import shap
import time


def integrated_gradients(model, inputs, baseline=None, steps=100, target_class_idx=0):
    """
    Calculate integrated gradients for a TensorFlow model.

    Parameters:
        model (tf.keras.Model): Trained TensorFlow model
        inputs (tf.Tensor): Input data for which to calculate integrated gradients
        baseline (tf.Tensor, optional): Baseline input (typically zeros)
        steps (int): Number of steps in the Riemann approximation
        target_class_idx (int): Index of the target class for multiclass models

    Returns:
        numpy.ndarray: Integrated gradients for each feature
        numpy.ndarray: Prediction for the input
    """
    if not isinstance(inputs, tf.Tensor):
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)

    if baseline is None:
        baseline = tf.zeros_like(inputs)
    elif not isinstance(baseline, tf.Tensor):
        baseline = tf.convert_to_tensor(baseline, dtype=tf.float32)

    alphas = tf.linspace(0.0, 1.0, steps+1)

    path_gradients = []

    for alpha in alphas:
        interpolated_input = baseline + alpha * (inputs - baseline)

        with tf.GradientTape() as tape:
            tape.watch(interpolated_input)
            prediction = model(interpolated_input)
            if prediction.shape[-1] > 1:  # Multi-class case
                prediction = prediction[:, target_class_idx]
            else:  # Binary classification case
                prediction = tf.squeeze(prediction)

        gradients = tape.gradient(prediction, interpolated_input)
        path_gradients.append(gradients)

    path_gradients = tf.stack(path_gradients)

    avg_gradients = tf.reduce_mean(path_gradients, axis=0)

    integrated_gradients_result = (inputs - baseline) * avg_gradients

    prediction = model(inputs)

    return integrated_gradients_result.numpy(), prediction.numpy()


def calculate_shap_importance(model, X_train, X_test, feature_names, max_train_samples=10000, max_test_samples=5000, seed=42):
    """
    Calculate SHAP feature importance with optimized sampling.

    Parameters:
        model: Trained TensorFlow model
        X_train: Training data for SHAP explainer background
        X_test: Test data to explain
        feature_names: List of feature names
        max_train_samples: Maximum training samples for explainer background
        max_test_samples: Maximum test samples to explain

    Returns:
        pandas.DataFrame: SHAP importance results
    """
    print("Calculating SHAP values...")
    shap_start = time.time()

    train_sample_size = min(max_train_samples, len(X_train))
    test_sample_size = min(max_test_samples, len(X_test))

    rng = np.random.default_rng(seed)

    train_idx = rng.choice(len(X_train), size=train_sample_size, replace=False)
    test_idx  = rng.choice(len(X_test),  size=test_sample_size,  replace=False)

    shap_train_sample = X_train[train_idx]
    shap_test_sample  = X_test[test_idx]

    print(f"Using {train_sample_size} training samples and {test_sample_size} test samples for SHAP")

    explainer = shap.DeepExplainer(model, shap_train_sample)

    shap_values = explainer.shap_values(shap_test_sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    if len(shap_values.shape) > 2:
        shap_values = shap_values.squeeze(axis=-1)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'SHAP': mean_abs_shap
    }).sort_values('SHAP', ascending=False)

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, shap_test_sample, feature_names=feature_names, show=False)
    plt.title("SHAP Feature Importance Summary")
    plt.tight_layout()
    plt.show()

    print(f"SHAP calculation completed in {time.time() - shap_start:.2f} seconds")

    return shap_df, shap_values


def calculate_ig_importance(model, X_test, feature_names, max_samples=10000, steps=100, seed=42):
    """
    Calculate Integrated Gradients feature importance.

    Parameters:
        model: Trained TensorFlow model
        X_test: Test data to explain
        feature_names: List of feature names
        max_samples: Maximum samples to process
        steps: Number of integration steps

    Returns:
        pandas.DataFrame: IG importance results
    """
    print("Calculating Integrated Gradients...")
    ig_start = time.time()

    sample_size = min(max_samples, len(X_test))
    rng = np.random.default_rng(seed)
    ig_idx = rng.choice(len(X_test), size=sample_size, replace=False)
    ig_sample = X_test[ig_idx]

    print(f"Using {sample_size} test samples for Integrated Gradients")

    ig_values, predictions = integrated_gradients(
        model,
        tf.convert_to_tensor(ig_sample, dtype=tf.float32),
        steps=steps
    )

    mean_abs_ig = np.abs(ig_values).mean(axis=0)

    ig_df = pd.DataFrame({
        'Feature': feature_names,
        'IG': mean_abs_ig
    }).sort_values('IG', ascending=False)

    plt.figure(figsize=(10, 8))
    top_20_ig = ig_df.head(20)
    plt.barh(range(len(top_20_ig)), top_20_ig['IG'])
    plt.yticks(range(len(top_20_ig)), top_20_ig['Feature'])
    plt.xlabel('Integrated Gradients Importance')
    plt.title('Top 20 Features by Integrated Gradients')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    print(f"Integrated Gradients calculation completed in {time.time() - ig_start:.2f} seconds")

    return ig_df, ig_values


def calculate_sum_of_ranks(shap_df, ig_df, weight_shap=0.5, weight_ig=0.5):
    """
    Calculate sum of ranks using SHAP and IG importance scores.

    Parameters:
        shap_df: DataFrame with SHAP importance scores
        ig_df: DataFrame with IG importance scores
        weight_shap: Weight for SHAP ranking (default 0.5)
        weight_ig: Weight for IG ranking (default 0.5)

    Returns:
        pandas.DataFrame: Combined results with sum of ranks
    """
    print("Calculating sum of ranks from SHAP and IG...")

    combined_df = shap_df.merge(ig_df, on='Feature', how='outer')
    combined_df = combined_df.fillna(0)

    combined_df['SHAP_Rank'] = combined_df['SHAP'].rank(ascending=False, method='min')
    combined_df['IG_Rank'] = combined_df['IG'].rank(ascending=False, method='min')

    combined_df['Weighted_Sum_Ranks'] = (
        weight_shap * combined_df['SHAP_Rank'] +
        weight_ig * combined_df['IG_Rank']
    )

    combined_df['Sum_of_Ranks'] = combined_df['SHAP_Rank'] + combined_df['IG_Rank']

    for col in ['SHAP', 'IG']:
        min_val = combined_df[col].min()
        max_val = combined_df[col].max()
        if max_val > min_val:
            combined_df[f'{col}_Normalized'] = (combined_df[col] - min_val) / (max_val - min_val)
        else:
            combined_df[f'{col}_Normalized'] = 0

    combined_df['Average_Importance'] = combined_df[['SHAP_Normalized', 'IG_Normalized']].mean(axis=1)

    combined_df = combined_df.sort_values('Weighted_Sum_Ranks')
    combined_df = combined_df.reset_index(drop=True)

    print("Sum of ranks calculation completed")

    return combined_df


def plot_shap_ig_comparison(combined_df, category_name=None, top_k=15):
    """
    Plot comparison between SHAP and IG importance scores.

    Parameters:
        combined_df: DataFrame with combined results
        category_name: Optional category name for title and file saving
        top_k: Number of top features to display
    """
    top_features = combined_df.head(top_k)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    x = np.arange(len(top_features))
    width = 0.35

    ax1.barh(x - width/2, top_features['SHAP'], width, label='SHAP', alpha=0.8, color='blue')
    ax1.barh(x + width/2, top_features['IG'], width, label='IG', alpha=0.8, color='red')
    ax1.set_yticks(x)
    ax1.set_yticklabels(top_features['Feature'], fontsize=8)
    ax1.set_xlabel('Importance Score')
    ax1.set_title(f'SHAP vs IG Importance{" - " + category_name if category_name else ""}')
    ax1.legend()
    ax1.invert_yaxis()

    ax2.barh(x - width/2, top_features['SHAP_Normalized'], width, label='SHAP (Normalized)', alpha=0.8, color='lightblue')
    ax2.barh(x + width/2, top_features['IG_Normalized'], width, label='IG (Normalized)', alpha=0.8, color='lightcoral')
    ax2.set_yticks(x)
    ax2.set_yticklabels(top_features['Feature'], fontsize=8)
    ax2.set_xlabel('Normalized Importance')
    ax2.set_title('Normalized SHAP vs IG Importance')
    ax2.legend()
    ax2.invert_yaxis()

    ax3.scatter(top_features['SHAP_Rank'], top_features['IG_Rank'], alpha=0.7, s=60)
    ax3.set_xlabel('SHAP Rank')
    ax3.set_ylabel('IG Rank')
    ax3.set_title('SHAP vs IG Rankings')
    ax3.plot([1, max(top_features['SHAP_Rank'])], [1, max(top_features['IG_Rank'])], 'r--', alpha=0.5)

    for i, row in top_features.iterrows():
        ax3.annotate(row['Feature'][:10], (row['SHAP_Rank'], row['IG_Rank']),
                    fontsize=6, alpha=0.7)

    ax4.barh(range(len(top_features)), top_features['Sum_of_Ranks'], alpha=0.8, color='green')
    ax4.set_yticks(range(len(top_features)))
    ax4.set_yticklabels(top_features['Feature'], fontsize=8)
    ax4.set_xlabel('Sum of Ranks (Lower = Better)')
    ax4.set_title('Combined Feature Ranking')
    ax4.invert_yaxis()

    plt.tight_layout()

    if category_name:
        plt.savefig(f'shap_ig_comparison_{category_name}.png', dpi=300, bbox_inches='tight')

    plt.show()


def analyze_method_agreement(combined_df, top_k=20):
    """
    Analyze agreement between SHAP and IG methods.

    Parameters:
        combined_df: DataFrame with combined results
        top_k: Number of top features to analyze

    Returns:
        dict: Agreement analysis results
    """
    print(f"Analyzing agreement between SHAP and IG for top {top_k} features...")

    top_shap = set(combined_df.nsmallest(top_k, 'SHAP_Rank')['Feature'])
    top_ig = set(combined_df.nsmallest(top_k, 'IG_Rank')['Feature'])
    top_combined = set(combined_df.head(top_k)['Feature'])

    shap_ig_overlap = len(top_shap.intersection(top_ig))
    shap_combined_overlap = len(top_shap.intersection(top_combined))
    ig_combined_overlap = len(top_ig.intersection(top_combined))

    correlation_raw = combined_df['SHAP'].corr(combined_df['IG'])
    correlation_normalized = combined_df['SHAP_Normalized'].corr(combined_df['IG_Normalized'])
    correlation_ranks = combined_df['SHAP_Rank'].corr(combined_df['IG_Rank'])

    only_shap = top_shap - top_ig
    only_ig = top_ig - top_shap

    agreement_results = {
        'top_k': top_k,
        'shap_ig_overlap': shap_ig_overlap,
        'overlap_percentage': (shap_ig_overlap / top_k) * 100,
        'correlation_raw_scores': correlation_raw,
        'correlation_normalized_scores': correlation_normalized,
        'correlation_ranks': correlation_ranks,
        'features_only_in_shap_top': list(only_shap),
        'features_only_in_ig_top': list(only_ig),
        'consensus_features': list(top_shap.intersection(top_ig))
    }

    print(f"Agreement Analysis Results:")
    print(f"  - Features in both top-{top_k}: {shap_ig_overlap}/{top_k} ({agreement_results['overlap_percentage']:.1f}%)")
    print(f"  - Correlation (raw scores): {correlation_raw:.3f}")
    print(f"  - Correlation (normalized): {correlation_normalized:.3f}")
    print(f"  - Correlation (ranks): {correlation_ranks:.3f}")
    print(f"  - Features only in SHAP top-{top_k}: {len(only_shap)}")
    print(f"  - Features only in IG top-{top_k}: {len(only_ig)}")

    return agreement_results


def apply_shap_ig_explainability(
    model, X_train, y_train, X_test, y_test,
    feature_names=None, category_name=None,
    feature_categories_df=None,
    shap_train_samples=5000, shap_test_samples=5000,
    ig_samples=10000, weight_shap=0.5, weight_ig=0.5
):
    """
    Apply SHAP and IG explainability methods and calculate sum of ranks.

    Parameters:
        model: Trained model
        X_train: Training features (for SHAP background)
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        feature_names: List of feature names
        category_name: Category name for saving files
        feature_categories_df: DataFrame with feature category mappings
        shap_train_samples: Number of training samples for SHAP background
        shap_test_samples: Number of test samples for SHAP explanation
        ig_samples: Number of samples for IG calculation
        weight_shap: Weight for SHAP in final ranking
        weight_ig: Weight for IG in final ranking

    Returns:
        dict: Comprehensive explainability results
    """
    print(f"Starting SHAP + IG explainability analysis for {category_name if category_name else 'model'}")
    start_time = time.time()

    if feature_names is None:
        if hasattr(X_test, 'columns'):
            feature_names = X_test.columns.tolist()
        else:
            feature_names = [f'Feature_{i}' for i in range(X_test.shape[1])]

    X_train_values = X_train.values if hasattr(X_train, 'values') else X_train
    X_test_values = X_test.values if hasattr(X_test, 'values') else X_test

    # 1. Calculate SHAP importance
    shap_df, shap_values = calculate_shap_importance(
        model, X_train_values, X_test_values, feature_names,
        max_train_samples=shap_train_samples,
        max_test_samples=shap_test_samples
    )

    # 2. Calculate IG importance
    ig_df, ig_values = calculate_ig_importance(
        model, X_test_values, feature_names,
        max_samples=ig_samples
    )

    # 3. Calculate sum of ranks
    combined_df = calculate_sum_of_ranks(shap_df, ig_df, weight_shap, weight_ig)

    # 4. Add category information if available
    if feature_categories_df is not None:
        combined_df = combined_df.merge(
            feature_categories_df,
            left_on='Feature', right_on='Feature', how='left'
        )

    # 5. Analyze method agreement
    agreement_analysis = analyze_method_agreement(combined_df)

    # 6. Save results
    if category_name:
        combined_df.to_csv(f'shap_ig_combined_{category_name}.csv', index=False)
        shap_df.to_csv(f'shap_detailed_{category_name}.csv', index=False)
        ig_df.to_csv(f'ig_detailed_{category_name}.csv', index=False)

        agreement_df = pd.DataFrame([agreement_analysis])
        agreement_df.to_csv(f'method_agreement_{category_name}.csv', index=False)

    # 7. Create visualizations
    plot_shap_ig_comparison(combined_df, category_name)

    total_time = time.time() - start_time
    print(f"SHAP + IG analysis completed in {total_time:.2f} seconds")

    return {
        'combined_df': combined_df,
        'shap_df': shap_df,
        'ig_df': ig_df,
        'shap_values': shap_values,
        'ig_values': ig_values,
        'agreement_analysis': agreement_analysis,
        'total_time': total_time,
        'feature_names': feature_names
    }
