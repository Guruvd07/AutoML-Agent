from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    r2_score,
    roc_auc_score
)
from xgboost import XGBRegressor, XGBClassifier


def train_and_evaluate(df, target, models, problem_type, threshold=0.6):

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}

    for name, model in models.items():
        try:

            # -----------------------------
            # FIX XGBOOST (NO STRING ISSUE)
            # -----------------------------
            if name == "XGBoost":
                if problem_type == "regression":
                    model = XGBRegressor(
                        n_estimators=300,
                        learning_rate=0.05,
                        max_depth=4,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="reg:squarederror",
                        random_state=42
                    )
                else:
                    model = XGBClassifier(
                        n_estimators=300,
                        learning_rate=0.05,
                        max_depth=4,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        eval_metric="logloss",
                        random_state=42,
                        use_label_encoder=False
                    )

            # TRAIN
            model.fit(X_train, y_train)

            # -----------------------------
            # REGRESSION
            # -----------------------------
            if problem_type == "regression":
                preds = model.predict(X_test)

                mse = mean_squared_error(y_test, preds)
                r2 = r2_score(y_test, preds)

                results[name] = {
                    "model": model,
                    "mse": mse,
                    "r2": r2,
                    "preds": preds
                }

            # -----------------------------
            # CLASSIFICATION
            # -----------------------------
            else:
                # Safe probability handling
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_test)[:, 1]
                else:
                    probs = model.predict(X_test)

                preds = (probs > threshold).astype(int)

                acc = accuracy_score(y_test, preds)
                precision = precision_score(y_test, preds, zero_division=0)
                recall = recall_score(y_test, preds, zero_division=0)
                f1 = f1_score(y_test, preds, zero_division=0)
                cm = confusion_matrix(y_test, preds)
                auc = roc_auc_score(y_test, probs)

                results[name] = {
                    "model": model,
                    "accuracy": acc,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "roc_auc": auc,
                    "confusion_matrix": cm,
                    "preds": preds
                }

        except Exception as e:
            print(f"⚠️ Model {name} failed: {e}")
            continue

    return results