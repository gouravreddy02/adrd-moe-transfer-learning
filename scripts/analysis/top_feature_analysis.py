import os
import pandas as pd

feature_names_df = pd.read_csv('../New_Data/final_variable_info.csv')
feature_names_df = feature_names_df.rename(columns={"colname_match_data": "Feature"})


def sort_by_p_and_rank(df, p_col="P_Value_Low", rank_col="Sum_of_Ranks", p_threshold=0.05):
    """Sorts DataFrame with p<threshold first (by rank), then non-significant (by rank)."""
    sig = df[df[p_col] < p_threshold].sort_values(by=rank_col, ascending=True)
    nonsig = df[df[p_col] >= p_threshold].sort_values(by=rank_col, ascending=True)
    return pd.concat([sig, nonsig], ignore_index=True)


def process_datasets():
    datasets = ['Black', 'Asian', 'White']

    cols_to_add = [c for c in feature_names_df.columns if c != "Category"]

    for dataset_name in datasets:
        print(f"\nProcessing {dataset_name} dataset...")

        input_file = f"results/corrected_BH_qvalues_{dataset_name.lower()}.csv"

        if not os.path.exists(input_file):
            print(f"  Warning: File not found: {input_file}")
            continue

        df = pd.read_csv(input_file)

        # All features sorted: significant first, then non-significant, both by rank
        full_df = sort_by_p_and_rank(df)
        full_df = full_df.merge(feature_names_df[cols_to_add], on="Feature", how="left")
        full_output = f"results/final_{dataset_name.lower()}_all_features.csv"
        full_df.to_csv(full_output, index=False)
        print(f"  Saved all features ({len(full_df)}) to: {full_output}")

        # Significant features only (P_Value_Low < 0.05), sorted by rank
        if "P_Value_Low" not in df.columns:
            print(f"  Error: P_Value_Low column missing in {input_file}")
            continue

        filtered_df = (
            df[df["P_Value_Low"] < 0.05]
            .sort_values(by="Sum_of_Ranks", ascending=True)
            .reset_index(drop=True)
        )
        filtered_df = filtered_df.merge(feature_names_df[cols_to_add], on="Feature", how="left")
        filtered_output = f"results/final_{dataset_name.lower()}_top_features.csv"
        filtered_df.to_csv(filtered_output, index=False)
        print(f"  Saved top features ({len(filtered_df)}) to: {filtered_output}")


if __name__ == "__main__":
    process_datasets()