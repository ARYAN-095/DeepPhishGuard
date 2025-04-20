# src/model/trainer.py

import os
import joblib
import numpy as np
import mlflow
import mlflow.xgboost

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

from .evaluator import evaluate_and_log
import numpy as np
from sklearn.model_selection import train_test_split

DATA_DIR       = "data/processed"
MODEL_DIR      = "models"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_best.pkl")

def train_with_tuning():
    # 1) Load data
    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y = np.load(os.path.join(DATA_DIR, "y.npy"))

    # 2) Train/test split (20% hold‑out)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 3) Set up MLflow experiment
    mlflow.set_experiment("Phishing-XGBoost-GridSearch")
    with mlflow.start_run():
        # 4) Base estimator
        base_clf = XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            n_jobs=-1,
            verbosity=0
        )

        # 5) Parameter grid for tuning
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth":    [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
        }
        mlflow.log_params({"param_grid": str(param_grid)})

        # 6) 5‑fold stratified CV
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(
            estimator=base_clf,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv,
            verbose=2,
            n_jobs=-1
        )

        # 7) Run grid search
        print("Starting GridSearchCV …")
        grid.fit(X_train, y_train)
        best_params   = grid.best_params_
        best_cv_score = grid.best_score_

        print(f"Best params: {best_params}")
        print(f"Best CV AUC: {best_cv_score:.4f}")

        # Log best params & CV score
        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_auc", best_cv_score)

        # 8) Evaluate on hold‑out set
        best_clf = grid.best_estimator_
        y_pred   = best_clf.predict(X_test)
        y_proba  = best_clf.predict_proba(X_test)[:,1]

        report = classification_report(y_test, y_pred, output_dict=True)
        auc     = roc_auc_score(y_test, y_proba)

        print("\n=== Hold‑Out Test Metrics ===")
        print(classification_report(y_test, y_pred))
        print("Hold‑Out AUC:", auc)

        # Log detailed metrics
        mlflow.log_metric("test_auc", auc)
        mlflow.log_metric("test_precision", report["1"]["precision"])
        mlflow.log_metric("test_recall",    report["1"]["recall"])
        mlflow.log_metric("test_f1_score",  report["1"]["f1-score"])

        # 9) Save best model locally
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(best_clf, BEST_MODEL_PATH)
        print(f"Best model saved to {BEST_MODEL_PATH}")

        # 10) Log model artifact to MLflow
        mlflow.xgboost.log_model(best_clf, artifact_path="model")

        run_id = mlflow.active_run().info.run_id
        print(f"MLflow run completed with run_id={run_id}")

        evaluate_and_log(best_clf, X_test, y_test)

    return best_clf



if __name__ == "__main__":
    train_with_tuning()
