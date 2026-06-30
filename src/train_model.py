"""
Train Machine Learning Models
"""

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def train_models(data_path):
    """
    Train multiple machine learning models and save the best one.
    """

    # Load dataset
    df = pd.read_csv(data_path)

    # Features and Target
    X = df.drop(columns=["Customer ID", "Churn"])
    y = df["Churn"]

    # Save feature names
    joblib.dump(list(X.columns), "../models/feature_columns.pkl")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Scale data for Logistic Regression
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    joblib.dump(scaler, "../models/scaler.pkl")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
    }

    best_model = None
    best_auc = 0

    print("=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)

    for name, model in models.items():

        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test_scaled)
            prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            prob = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, prob)

        print(f"\n{name}")
        print("-" * 30)
        print("Accuracy :", accuracy_score(y_test, pred))
        print("Precision:", precision_score(y_test, pred))
        print("Recall   :", recall_score(y_test, pred))
        print("F1 Score :", f1_score(y_test, pred))
        print("ROC AUC  :", auc)

        if auc > best_auc:
            best_auc = auc
            best_model = model

    # Save best model
    joblib.dump(best_model, "../models/churn_model.pkl")

    print("\nBest Model Saved Successfully!")

    return best_model