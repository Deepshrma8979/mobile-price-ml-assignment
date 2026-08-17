"""
train_models.py

Trains 5 classification models on a chosen dataset, evaluates each with
6 metrics (Accuracy, AUC, Precision, Recall, F1, MCC), and saves every
trained model (preprocessing + classifier, bundled as one Pipeline) with
joblib so the Streamlit app can load and use them directly on raw input.

Usage:
    python train_models.py --data mobile_train.csv
    python train_models.py --data mobile_train.csv --test-size 0.25
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

RANDOM_STATE = 42

# CHANGE THIS if your dataset's label column has a different name
TARGET_COL = "price_range"

MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found in {path}. "
            f"Available columns: {list(df.columns)}. "
            f"Update TARGET_COL at the top of this script if needed."
        )
    return df


def build_models() -> dict:
    """
    One sklearn Pipeline per model: median-imputer -> StandardScaler -> classifier.
    Bundling preprocessing INSIDE the pipeline (not as separate fit_transform calls)
    means the saved .joblib file is self-contained: at inference time (in the
    Streamlit app) you can feed it raw, unscaled rows and it will handle
    imputation + scaling internally before predicting.
    """
    return {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]),
        "kNN": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier()),
        ]),
        "Naive Bayes": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", GaussianNB()),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=RANDOM_STATE)),
        ]),
    }


def evaluate(model, X_test, y_test, classes) -> dict:
    """Compute all 6 required metrics. Handles binary vs multi-class automatically."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    is_multiclass = len(classes) > 2

    if is_multiclass:
        y_test_bin = label_binarize(y_test, classes=classes)
        auc = roc_auc_score(y_test_bin, y_proba, average="weighted", multi_class="ovr")
    else:
        auc = roc_auc_score(y_test, y_proba[:, 1])

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }


def main(data_path: str, test_size: float = 0.2):
    df = load_data(data_path)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    classes = sorted(y.unique())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    # Save the held-out test split — this becomes test_data.csv, the file
    # required in the GitHub repo AND the file you upload into the Streamlit app.
    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test.values
    test_df.to_csv("test_data.csv", index=False)
    print(f"Saved held-out test split -> test_data.csv ({len(test_df)} rows)")

    models = build_models()
    results = {}

    for name, pipeline in models.items():
        print(f"Training {name} ...")
        pipeline.fit(X_train, y_train)
        results[name] = evaluate(pipeline, X_test, y_test, classes)

        filename = MODEL_DIR / (name.lower().replace(" ", "_") + ".joblib")
        joblib.dump(pipeline, filename)
        print(f"  Saved -> {filename}")

    results_df = pd.DataFrame(results).T[
        ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    ].round(4)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON (paste this into your README table)")
    print("=" * 70)
    print(results_df.to_string())

    results_df.to_csv("model_comparison.csv")
    print("\nSaved comparison table -> model_comparison.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate 5 classifiers.")
    parser.add_argument("--data", type=str, required=True, help="Path to the raw dataset CSV")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction (default 0.2)")
    args = parser.parse_args()
    main(args.data, args.test_size)
