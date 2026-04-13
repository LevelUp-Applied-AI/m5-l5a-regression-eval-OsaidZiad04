"""
Module 5 Week A — Lab: Regression & Evaluation

Build and evaluate logistic and linear regression models on the
Petra Telecom customer churn dataset.

Run: python lab_regression.py
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix,
                             mean_absolute_error, r2_score, accuracy_score,
                             precision_score, recall_score, f1_score, ConfusionMatrixDisplay)


def load_data(filepath="data/telecom_churn.csv"):
    """Load the telecom churn dataset."""
    if not os.path.exists(filepath):
        filepath = "data/telecom_churn.csv"
        
    return pd.read_csv(filepath)


def split_data(df, target_col, test_size=0.2, random_state=42):
    """Split data into train and test sets with stratification.

    Args:
        df: DataFrame with features and target.
        target_col: Name of the target column.
        test_size: Fraction for test set.
        random_state: Random seed.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    # TODO: Separate features and target, then split with stratification
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    is_classification = target_col == "churned"
    stratify_col = y if is_classification else None
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify_col)



def build_logistic_pipeline():
    """Build a Pipeline with StandardScaler and LogisticRegression.

    Returns:
        sklearn Pipeline object.
    """
    # TODO: Create and return a Pipeline with two steps
    return Pipeline([
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(max_iter=100, random_state=42, class_weight="balanced"))
    ])



def build_ridge_pipeline():
    """Build a Pipeline with StandardScaler and Ridge regression.

    Returns:
        sklearn Pipeline object.
    """
    # TODO: Create and return a Pipeline for Ridge regression
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=42))
    ])


def build_lasso_pipeline():
    """Build a Pipeline with StandardScaler and Lasso regression."""
    # Stub for Task 5
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", Lasso(alpha=0.1, random_state=42))
    ])


def evaluate_classifier(pipeline, X_train, X_test, y_train, y_test):
    """Train the pipeline and return classification metrics.

    Args:
        pipeline: sklearn Pipeline with a classifier.
        X_train, X_test: Feature arrays.
        y_train, y_test: Label arrays.

    Returns:
        Dictionary with keys: 'accuracy', 'precision', 'recall', 'f1'.
    """
    # TODO: Fit the pipeline on training data, predict on test, compute metrics
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipeline.classes_)
    disp.plot()
    plt.show()

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }


def evaluate_regressor(pipeline, X_train, X_test, y_train, y_test):
    """Train the pipeline and return regression metrics.

    Args:
        pipeline: sklearn Pipeline with a regressor.
        X_train, X_test: Feature arrays.
        y_train, y_test: Target arrays.

    Returns:
        Dictionary with keys: 'mae', 'r2'.
    """
    # TODO: Fit the pipeline, predict, and compute MAE and R²
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {"mae": mae, "r2": r2}


def run_cross_validation(pipeline, X_train, y_train, cv=5):
    """Run stratified cross-validation on the pipeline.

    Args:
        pipeline: sklearn Pipeline.
        X_train: Training features.
        y_train: Training labels.
        cv: Number of folds.

    Returns:
        Array of cross-validation scores.
    """
    # TODO: Run cross_val_score with StratifiedKFold
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    return cross_val_score(pipeline, X_train, y_train, cv=skf, scoring="accuracy")


if __name__ == "__main__":
    df = load_data()
    if df is not None:
        print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

        # Select numeric features for classification
        numeric_features = ["tenure", "monthly_charges", "total_charges",
                            "num_support_calls", "senior_citizen",
                            "has_partner", "has_dependents"]

        # Classification: predict churn
        df_cls = df[numeric_features + ["churned"]].dropna()
        split = split_data(df_cls, "churned")
        if split:
            X_train, X_test, y_train, y_test = split
            pipe = build_logistic_pipeline()
            if pipe:
                metrics = evaluate_classifier(pipe, X_train, X_test, y_train, y_test)
                print(f"Logistic Regression: {metrics}")

                scores = run_cross_validation(pipe, X_train, y_train)
                if scores is not None:
                    print(f"CV: {scores.mean():.3f} +/- {scores.std():.3f}")

        # Regression: predict monthly_charges
        df_reg = df[["tenure", "total_charges", "num_support_calls",
                     "senior_citizen", "has_partner", "has_dependents",
                     "monthly_charges"]].dropna()
        split_reg = split_data(df_reg, "monthly_charges")
        if split_reg:
            X_tr, X_te, y_tr, y_te = split_reg
            ridge_pipe = build_ridge_pipeline()
            if ridge_pipe:
                reg_metrics = evaluate_regressor(ridge_pipe, X_tr, X_te, y_tr, y_te)
                print(f"Ridge Regression: {reg_metrics}")

            # Task 5: Lasso Regularization Comparison
            lasso_pipe = build_lasso_pipeline()
            if lasso_pipe:
                lasso_pipe.fit(X_tr, y_tr)
                ridge_coefs = ridge_pipe.named_steps['ridge'].coef_
                lasso_coefs = lasso_pipe.named_steps['lasso'].coef_
                
                print("\n--- Task 5: Coefficients Comparison ---")
                coef_df = pd.DataFrame({
                    "Feature": X_tr.columns,
                    "Ridge Coef": ridge_coefs,
                    "Lasso Coef": lasso_coefs
                })
                print(coef_df)
                
                zero_features = coef_df[coef_df["Lasso Coef"] == 0]["Feature"].tolist()
                print(f"\nFeatures driven to zero by Lasso: {zero_features}")

# --- Task 7: Summary of Findings ---
"""
1. Which features appear most important for predicting churn?
   Features like 'tenure' and 'monthly_charges' are typically the most important. High charges and low tenure strongly indicate churn.

2. How does the logistic regression model perform? Is precision or recall more concerning for this problem?
   Using class_weight="balanced" sacrifices some precision to improve recall. For telecom churn, recall is more concerning because missing a churner (False Negative) is more costly than falsely predicting someone will churn (False Positive).

3. What would you recommend as next steps to improve performance?
   - Tuning the decision threshold (e.g., lower than 0.5) to capture more churners.
   - Using tree-based models (Random Forest, XGBoost) which handle imbalanced data and non-linear relationships better.
   - Feature engineering (e.g., average charge per month).
"""