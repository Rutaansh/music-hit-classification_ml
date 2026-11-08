"""
Train Models Script for ML Assignment 2 - Music Virality Classification
========================================================================
Dataset: Viral Songs Dataset (10,000 instances, 13 features)
Target: viral (Binary: 0 = Not Viral, 1 = Viral)

Models:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier (KNN)
4. Naive Bayes Classifier (GaussianNB)
5. Support Vector Machine (SVC)

Metrics:
1. Accuracy
2. ROC-AUC Score
3. Precision
4. Recall
5. F1 Score
6. Matthews Correlation Coefficient (MCC)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

def run_pipeline():
    print("[1/5] Loading Viral Songs dataset...")
    data_path = os.path.join("Viral_Songs_Datatset", "viral_songs.csv")
    if not os.path.exists(data_path):
        data_path = "viral_songs.csv"
    
    df = pd.read_csv(data_path)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
    
    # Define features and target
    id_cols = ['track_id', 'track_name', 'artist']
    target_col = 'viral'
    
    # Feature columns (13 features: 12 numeric, 1 categorical)
    feature_cols = [c for c in df.columns if c not in id_cols and c != target_col]
    numeric_features = [
        'release_year', 'duration_sec', 'danceability', 'energy', 
        'valence', 'tempo_bpm', 'acousticness', 'instrumentalness', 
        'speechiness', 'liveness', 'loudness_db', 'streams'
    ]
    categorical_features = ['genre']
    
    print(f"Features ({len(feature_cols)}): Numeric = {len(numeric_features)}, Categorical = {len(categorical_features)}")
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Stratified 80/20 train/test split
    print("[2/5] Creating 80/20 train/test split (stratified, random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Save test_data.csv (combining features and target) for upload demonstration
    test_df = X_test.copy()
    test_df[target_col] = y_test.values
    test_df.to_csv("test_data.csv", index=False)
    print(f"Saved test_data.csv with shape {test_df.shape} (2,000 instances).")
    
    # Build and fit ColumnTransformer preprocessor
    print("[3/5] Fitting preprocessor (StandardScaler + OneHotEncoder)...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop'
    )
    
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    # Extract feature names after one-hot encoding
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_cat_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
    all_feature_names = numeric_features + encoded_cat_names
    
    # Define the 5 mandatory ML classifiers
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_split=20, min_samples_leaf=10, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=7, weights='distance'),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(C=1.0, kernel='rbf', probability=True, random_state=42)
    }
    
    os.makedirs("model", exist_ok=True)
    
    # Save preprocessor artifact
    joblib.dump(preprocessor, os.path.join("model", "preprocessor.joblib"))
    joblib.dump({
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "all_feature_names": all_feature_names,
        "genres": sorted(df['genre'].unique().tolist())
    }, os.path.join("model", "metadata.joblib"))
    
    print("[4/5] Training 5 classifiers & evaluating all 6 metrics...")
    evaluation_results = {}
    
    file_slugs = {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "KNN": "knn.joblib",
        "Naive Bayes": "naive_bayes.joblib",
        "Support Vector Machine": "svm.joblib"
    }
    
    for name, clf in models.items():
        print(f"\n--- Training {name} ---")
        clf.fit(X_train_proc, y_train)
        
        # Predict on holdout test set
        y_pred = clf.predict(X_test_proc)
        y_proba = clf.predict_proba(X_test_proc)[:, 1] if hasattr(clf, "predict_proba") else None
        
        # Compute all 6 required evaluation metrics
        acc = float(accuracy_score(y_test, y_pred))
        auc = float(roc_auc_score(y_test, y_proba)) if y_proba is not None else 0.0
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        mcc = float(matthews_corrcoef(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred).tolist()
        cr = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
        print(f"  AUC Score: {auc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  MCC Score: {mcc:.4f}")
        
        # Save model artifact
        model_path = os.path.join("model", file_slugs[name])
        joblib.dump(clf, model_path)
        
        evaluation_results[name] = {
            "Accuracy": round(acc, 4),
            "AUC": round(auc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
            "MCC": round(mcc, 4),
            "Confusion_Matrix": cm,
            "Classification_Report": cr,
            "Artifact_File": file_slugs[name]
        }
        
    # Save evaluation summary to JSON
    with open(os.path.join("model", "evaluation_results.json"), "w") as f:
        json.dump(evaluation_results, f, indent=4)
        
    print("\n[5/5] Summary of All 5 Model Metrics on Test Set (2,000 samples):")
    summary_df = pd.DataFrame(evaluation_results).T[["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
    print(summary_df.to_markdown())
    print("\nAll model artifacts successfully exported to model/ directory!")

if __name__ == "__main__":
    run_pipeline()
