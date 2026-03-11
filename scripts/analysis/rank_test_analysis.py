import os
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'

import time
import pandas as pd
import numpy as np

from scipy import stats

def calculate_ranks(values):
    """Calculate ranks where 1 = highest value (most important)."""
    return stats.rankdata(-np.array(values), method='min')

def calculate_pvalue_uniform_ranks(observed_sum, n_features, n_permutations=5000, random_state=42):
    """Calculate p-value using uniform rank permutation method (low tail)."""
    np.random.seed(random_state)
    count_favorable = 0
    for k in range(n_permutations):
        R1_k = np.random.randint(1, n_features + 1)
        R2_k = np.random.randint(1, n_features + 1)
        sum_k = R1_k + R2_k
        if sum_k <= observed_sum:
            count_favorable += 1
    return count_favorable / n_permutations

def calculate_pvalue_high_tail(observed_sum, n_features, n_permutations=5000, random_state=42):
    """Calculate p-value for high tail: P(Sum >= observed_sum)."""
    np.random.seed(random_state + 1000)
    count_high = 0
    for k in range(n_permutations):
        R1_k = np.random.randint(1, n_features + 1)
        R2_k = np.random.randint(1, n_features + 1)
        sum_k = R1_k + R2_k
        if sum_k >= observed_sum:
            count_high += 1
    return count_high / n_permutations

def benjamini_hochberg_qvalues(p_values, pi_0=1.0):
    """
    Calculate q-values using standard Benjamini-Hochberg procedure.

    Formula: q_i = min(1, min_{j>=i}(m * p_j / j)) for sorted p-values

    Parameters:
        p_values (array): P-values for all features
        pi_0 (float): Estimate of proportion of null hypotheses (conservative: 1.0)

    Returns:
        numpy.ndarray: Q-values in original order
    """
    p_values = np.array(p_values)
    m = len(p_values)

    # Get indices that would sort p-values
    sorted_indices = np.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]

    # Calculate q-values for sorted p-values
    sorted_q_values = np.zeros(m)

    # Apply Benjamini-Hochberg formula
    for i in range(m):
        # Formula: min over j>=i of (m * p_j / j)
        # Note: using 1-based indexing for ranks (j+1)
        min_val = float('inf')
        for j in range(i, m):
            val = (pi_0 * m * sorted_p_values[j]) / (j + 1)
            min_val = min(min_val, val)
        sorted_q_values[i] = min(1.0, min_val)

    # Ensure monotonicity (non-decreasing)
    for i in range(m - 2, -1, -1):
        sorted_q_values[i] = min(sorted_q_values[i], sorted_q_values[i + 1])

    # Restore original order
    q_values = np.zeros(m)
    q_values[sorted_indices] = sorted_q_values

    return q_values

def uniform_rank_permutation_test_corrected_qvalues(shap_values, ig_values, feature_names,
                                                   n_permutations=5000, alpha=0.05, random_state=42,
                                                   pi_0=1.0):
    """
    Perform uniform rank permutation test with CORRECTED Benjamini-Hochberg q-values.
    """

    shap_vals = np.array(shap_values)
    ig_vals = np.array(ig_values)
    n_features = len(shap_vals)

    # Calculate ranks (1 = highest importance)
    shap_ranks = calculate_ranks(shap_vals)
    ig_ranks = calculate_ranks(ig_vals)
    observed_sums = shap_ranks + ig_ranks
    n_terms = 2

    # Calculate theoretical expectations
    theoretical_mean = n_terms * (n_features + 1) / 2
    theoretical_std = np.sqrt(n_terms * (n_features ** 2 - 1) / 12)

    print(f"Analyzing {n_features} features with {n_permutations} permutations each...")
    print(f"Expected sum of ranks: {theoretical_mean:.2f}")
    print(f"Using CORRECTED Benjamini-Hochberg q-values")

    # Calculate p-values for all features
    print("Calculating p-values...")
    start_time = time.time()

    p_values_low = []
    p_values_high = []
    results = []

    for j in range(n_features):
        observed_sum = observed_sums[j]

        # Calculate both p-values
        p_value_low = calculate_pvalue_uniform_ranks(
            observed_sum, n_features, n_permutations, random_state + j
        )
        p_value_high = calculate_pvalue_high_tail(
            observed_sum, n_features, n_permutations, random_state + j
        )

        p_values_low.append(p_value_low)
        p_values_high.append(p_value_high)

        # Calculate two-sided p-value
        p_value_two_sided = min(2 * min(p_value_low, p_value_high), 1.0)

        # Calculate effect size
        z_score = (observed_sum - theoretical_mean) / theoretical_std
        effect_size = abs(z_score)

        results.append({
            'Feature': feature_names[j],
            'SHAP_Rank': shap_ranks[j],
            'IG_Rank': ig_ranks[j],
            'Sum_of_Ranks': observed_sum,
            'P_Value_Low': p_value_low,
            'P_Value_High': p_value_high,
            'P_Value_Two_Sided': p_value_two_sided,
            'Z_Score': z_score,
            'Effect_Size': effect_size
        })

        if (j + 1) % 50 == 0 or j == n_features - 1:
            elapsed = time.time() - start_time
            print(f"Processed {j + 1}/{n_features} features in {elapsed:.2f} seconds")

    # Calculate q-values using Benjamini-Hochberg procedure
    print("Calculating q-values using Benjamini-Hochberg procedure...")
    q_values_low = benjamini_hochberg_qvalues(p_values_low, pi_0)
    q_values_two_sided = benjamini_hochberg_qvalues([r['P_Value_Two_Sided'] for r in results], pi_0)

    # Add q-values and significance flags to results
    for j, result in enumerate(results):
        result['Q_Value_Low'] = q_values_low[j]
        result['Q_Value_Two_Sided'] = q_values_two_sided[j]

        # Significance based on q-values
        result['Significant_Qvalue_Low'] = q_values_low[j] < alpha
        result['Significant_Qvalue_Two_Sided'] = q_values_two_sided[j] < alpha

        # Significance based on p-values (for comparison)
        result['Significant_Pvalue_Low'] = result['P_Value_Low'] < alpha
        result['Significant_Pvalue_High'] = result['P_Value_High'] < alpha
        result['Significant_Pvalue_Two_Sided'] = result['P_Value_Two_Sided'] < alpha

        # Interpretation based on q-value (using low tail for high importance)
        observed_sum = result['Sum_of_Ranks']
        q_value = result['Q_Value_Low']

        interpretation = "No Significant Consensus"
        if result['Significant_Qvalue_Low'] and observed_sum < theoretical_mean:
            if q_value < 0.001:
                interpretation = "Very Strong HIGH Importance Consensus"
            elif q_value < 0.01:
                interpretation = "Strong HIGH Importance Consensus"
            else:
                interpretation = "Moderate HIGH Importance Consensus"
        elif result['Significant_Qvalue_Two_Sided']:
            if q_values_two_sided[j] < 0.001:
                if observed_sum < theoretical_mean:
                    interpretation = "Very Strong HIGH Importance Consensus"
                else:
                    interpretation = "Very Strong LOW Importance Consensus"
            elif q_values_two_sided[j] < 0.01:
                if observed_sum < theoretical_mean:
                    interpretation = "Strong HIGH Importance Consensus"
                else:
                    interpretation = "Strong LOW Importance Consensus"
            else:
                if observed_sum < theoretical_mean:
                    interpretation = "Moderate HIGH Importance Consensus"
                else:
                    interpretation = "Moderate LOW Importance Consensus"

        result['Consensus_Interpretation'] = interpretation

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Sum_of_Ranks').reset_index(drop=True)
    results_df['Consensus_Rank'] = range(1, len(results_df) + 1)

    total_time = time.time() - start_time
    print(f"Analysis completed in {total_time:.2f} seconds")

    return results_df

def analyze_dataset_corrected_qvalues(file_path, dataset_name, shap_col='SHAP', ig_col='IG',
                                     feature_col='Feature', category_col='Category',
                                     n_permutations=5000, pi_0=1.0, alpha=0.05):
    
    """Analyze dataset with corrected Benjamini-Hochberg q-values."""

    print(f"\n{'='*70}")
    print(f"ANALYZING {dataset_name.upper()} DATASET WITH CORRECTED BH Q-VALUES")
    print(f"{'='*70}")

    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} features from {file_path}")

    shap_values = df[shap_col].values
    ig_values = df[ig_col].values
    feature_names = df[feature_col].values
    categories = df[category_col].values if category_col in df.columns else None

    results = uniform_rank_permutation_test_corrected_qvalues(
        shap_values, ig_values, feature_names,
        n_permutations=n_permutations, pi_0=pi_0, alpha=alpha
    )

    if categories is not None:
        feature_to_category = dict(zip(feature_names, categories))
        results['Category'] = results['Feature'].map(feature_to_category)

        cols = ['Consensus_Rank', 'Feature', 'Category', 'Sum_of_Ranks', 'SHAP_Rank', 'IG_Rank',
                'P_Value_Low', 'P_Value_High', 'P_Value_Two_Sided',
                'Q_Value_Low', 'Q_Value_Two_Sided', 'Effect_Size', 'Z_Score',
                'Significant_Pvalue_Low', 'Significant_Pvalue_High', 'Significant_Pvalue_Two_Sided',
                'Significant_Qvalue_Low', 'Significant_Qvalue_Two_Sided', 'Consensus_Interpretation']
        results = results[cols]

    high_importance_q = results[results['Consensus_Interpretation'].str.contains('HIGH Importance', na=False)]
    significant_qvalue = results[results['Significant_Qvalue_Low'] | results['Significant_Qvalue_Two_Sided']]
    significant_pvalue = results[results['Significant_Pvalue_Two_Sided']]

    print(f"\n{dataset_name} Dataset Summary:")
    print(f"Total features: {len(results)}")
    print(f"High importance consensus (BH q-value): {len(high_importance_q)}")
    print(f"Significant by q-value (α={alpha}): {len(significant_qvalue)}")
    print(f"Significant by p-value (α={alpha}): {len(significant_pvalue)}")
    print(f"π₀ (null proportion estimate): {pi_0}")

    return results

def main_corrected_qvalues():

    print("UNIFORM RANK PERMUTATION TEST WITH CORRECTED BENJAMINI-HOCHBERG Q-VALUES")
    print("=" * 80)
    print("CORRECTION: Using standard BH procedure instead of MAPE permutation-based formula")
    print("=" * 80)

    files = {
        'Black': 'results/shap_ig_combined_black_adaptive_moe.csv',
        'Asian': 'results/shap_ig_combined_asian_adaptive_moe.csv',
        'White': 'results/shap_ig_combined_white_adaptive_moe.csv',
    }

    configs = {
        'Black': {'shap_col': 'SHAP', 'ig_col': 'IG'},
        'Asian': {'shap_col': 'SHAP', 'ig_col': 'IG'},
        'White': {'shap_col': 'SHAP', 'ig_col': 'IG'}
    }

    results = {}
    n_permutations = 5000
    alpha = 0.05
    pi_0 = 1.0

    for dataset_name in ['Black', 'Asian', 'White']:
        try:
            results[dataset_name] = analyze_dataset_corrected_qvalues(
                files[dataset_name], dataset_name, n_permutations=n_permutations,
                pi_0=pi_0, alpha=alpha, **configs[dataset_name]
            )
        except Exception as e:
            print(f"Error analyzing {dataset_name}: {e}")
            continue

    if len(results) > 0:

        for dataset_name, result_df in results.items():
            filename = f"corrected_BH_qvalues_{dataset_name.lower()}.csv"
            result_df.to_csv(filename, index=False)
            print(f"Saved: {filename}")

    print("\nCORRECTED analysis completed successfully!")
    return results

if __name__ == "__main__": 
    results = main_corrected_qvalues()