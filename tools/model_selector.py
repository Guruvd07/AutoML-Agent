from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression


def get_models(problem_type):
    if "regression" in problem_type:
        return {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                random_state=42
            ),
            "XGBoost": None  # handled in training
        }
    else:
        return {
            "LogisticRegression": LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                class_weight="balanced",
                random_state=42
            ),
            "XGBoost": None  # handled in training
        }