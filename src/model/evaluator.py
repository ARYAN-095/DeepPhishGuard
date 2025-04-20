# src/model/evaluator.py

import os
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.xgboost

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from matplotlib import pyplot as plt

MODEL_DIR      = "models"
RESULTS_DIR    = "outputs"
CONF_MATRIX_PNG = os.path.join(RESULTS_DIR, "confusion_matrix.png")
FI_PLOT_PNG     = os.path.join(RESULTS_DIR, "feature_importance.png")
PRED_CSV        = os.path.join(RESULTS_DIR, "predictions.csv")

def plot_confusion_matrix(y_true, y_pred, labels=(0,1)):
    """Plot & save the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(np.arange(len(labels)), labels)
    plt.yticks(np.arange(len(labels)), labels)
    # annotate counts
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, cm[i, j], 
                     ha="center", va="center")
    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(CONF_MATRIX_PNG)
    plt.close()
    return CONF_MATRIX_PNG

def plot_feature_importance(model, max_num=20):
    """Plot & save the top `max_num` feature importances."""
    # XGBoost stores feature importances in .feature_importances_
    importances = model.feature_importances_
    indices = np.argsort(importances)[-max_num:][::-1]
    plt.figure()
    plt.bar(range(len(indices)), importances[indices], align="center")
    plt.xticks(range(len(indices)), indices, rotation=90)
    plt.title("Top %d Feature Importances" % max_num)
    plt.tight_layout()
    plt.savefig(FI_PLOT_PNG)
    plt.close()
    return FI_PLOT_PNG

def export_predictions(X_test, y_test, y_pred, y_proba):
    """Save a CSV of URL index, true label, predicted label, probability."""
    df = pd.DataFrame({
        "index": np.arange(len(y_test)),
        "true": y_test,
        "pred": y_pred,
        "proba": y_proba
    })
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(PRED_CSV, index=False)
    return PRED_CSV

def evaluate_and_log(model, X_test, y_test):
    """
    Run evaluation, produce plots, export CSV, and log everything to MLflow.
    """
    # 1) Predictions & metrics
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    auc     = roc_auc_score(y_test, y_proba)
    report  = classification_report(y_test, y_pred, output_dict=True)

    print("=== Final Classification Report ===")
    print(classification_report(y_test, y_pred))
    print("Final AUC:", auc)

    # 2) Plots
    cm_path = plot_confusion_matrix(y_test, y_pred)
    fi_path = plot_feature_importance(model)

    # 3) Export predictions
    preds_path = export_predictions(X_test, y_test, y_pred, y_proba)

    # 4) Log to MLflow
    mlflow.log_metric("final_auc", auc)
    mlflow.log_metric("final_precision", report["1"]["precision"])
    mlflow.log_metric("final_recall",    report["1"]["recall"])
    mlflow.log_metric("final_f1_score",  report["1"]["f1-score"])

    mlflow.log_artifact(cm_path,     artifact_path="evaluation")
    mlflow.log_artifact(fi_path,     artifact_path="evaluation")
    mlflow.log_artifact(preds_path,  artifact_path="evaluation")

    print(f"Evaluation artifacts logged to MLflow under run {mlflow.active_run().info.run_id}")
