

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Dict, Any
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier


# ===============================
# Utility Functions
# ===============================
def _save_plot(fig, save_dir: str, filename: str):
    """Save plot to a directory if provided."""
    if save_dir:
        save_path = os.path.join(save_dir, "eda_results")
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)
        fig.savefig(path, bbox_inches="tight")
        print(f"[+] Plot saved to: {path}")


def _encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Label encode all categorical columns."""
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(exclude=[np.number]).columns:
        df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))
    return df_encoded


# ===============================
# Main EDA Function
# ===============================
def run_eda(df: pd.DataFrame, save_dir: str = "/content/drive/MyDrive/flood-detect", top_k: int = 10) -> Dict[str, Any]:
    """
    Perform advanced EDA and feature selection for flood prediction.

    Args:
        df (pd.DataFrame): Dataset containing 'flood_event' as the target column.
        save_dir (str, optional): Directory to save visual outputs.
        top_k (int): Number of top features to display.

    Returns:
        dict: Summary including top features and data quality insights.
    """

    print("=" * 80)
    print(f"🌊 Advanced EDA Started for Flood Prediction Dataset")
    print("=" * 80)

    # Basic info
    print("\n[1] Dataset Info")
    print(df.info())
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Missing values
    print("\n[2] Missing Values Summary")
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_summary = pd.DataFrame({
        "Missing_Count": missing,
        "Missing_%": missing_percent
    }).sort_values(by="Missing_%", ascending=False)
    print(missing_summary[missing_summary["Missing_Count"] > 0].head(10))

    # Duplicates
    dup_count = df.duplicated().sum()
    print(f"\n[3] Duplicate Rows: {dup_count}")
    if dup_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"Duplicates removed. New shape: {df.shape}")

    # Separate types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    print(f"\n[4] Numeric Features ({len(numeric_cols)}): {numeric_cols}")
    print(f"[5] Categorical Features ({len(cat_cols)}): {cat_cols}")

    # Correlation
    if len(numeric_cols) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df[numeric_cols].corr(), cmap="coolwarm", annot=False, ax=ax)
        ax.set_title("Correlation Heatmap")
        _save_plot(fig, save_dir, "correlation_heatmap.png")
        plt.close(fig)

    # Numeric distributions
    if numeric_cols:
        df[numeric_cols].hist(figsize=(14, 10), bins=25, edgecolor='black')
        plt.suptitle("Numeric Feature Distributions", fontsize=16)
        _save_plot(plt.gcf(), save_dir, "numeric_distributions.png")
        plt.show()

    # Categorical distributions
    for col in cat_cols:
        fig, ax = plt.subplots(figsize=(8, 4))
        df[col].value_counts().head(10).plot(kind="bar", color="teal", ax=ax)
        ax.set_title(f"Top Categories in '{col}'")
        _save_plot(fig, save_dir, f"{col}_categories.png")
        plt.close(fig)

    # ====================================
    # Feature Importance (target: flood_event)
    # ====================================
    print("\n[6] Feature Selection & Importance (Target: 'flood_event')")

    target_col = "flood_event"
    if target_col not in df.columns:
        raise ValueError("Target column 'flood_event' not found in DataFrame")

    # Encode categorical columns
    df_encoded = _encode_categorical(df)

    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]

    # Normalize numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1️⃣ RandomForest Importance
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_scaled, y)
    rf_importance = pd.Series(rf.feature_importances_, index=X.columns, name="RandomForest").sort_values(ascending=False)

    # 2️⃣ Mutual Information
    mi = mutual_info_classif(X_scaled, y)
    mi_scores = pd.Series(mi, index=X.columns, name="Mutual_Info").sort_values(ascending=False)

    # 3️⃣ ANOVA F-Test
    f_selector = SelectKBest(score_func=f_classif, k=min(top_k, X.shape[1]))
    f_selector.fit(X_scaled, y)
    f_scores = pd.Series(f_selector.scores_, index=X.columns, name="ANOVA_F").sort_values(ascending=False)

    # Combine results
    feature_summary = pd.concat([rf_importance, mi_scores, f_scores], axis=1).fillna(0)
    feature_summary["Mean_Score"] = feature_summary.mean(axis=1)
    top_features = feature_summary.sort_values("Mean_Score", ascending=False).head(top_k)

    # Plot feature importance
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x=top_features["Mean_Score"],
        y=top_features.index,
        palette="viridis",
        ax=ax
    )
    ax.set_title(f"Top {top_k} Important Features for Flood Prediction")
    ax.set_xlabel("Average Importance Score")
    _save_plot(fig, save_dir, "top_features_combined.png")
    plt.close(fig)

    print("\n===== TOP FEATURES =====")
    print(top_features.round(3))

    # Return summary
    summary = {
        "shape": df.shape,
        "missing_summary": missing_summary.to_dict(),
        "duplicates": int(dup_count),
        "numeric_columns": numeric_cols,
        "categorical_columns": cat_cols,
        "top_features": top_features
    }

    print("\n✅ EDA Completed Successfully.")
    print("=" * 80)
    return summary


# Entry point for direct script testing
if __name__ == "__main__":
    print("This module provides run_eda(df) for flood_event feature analysis.")
