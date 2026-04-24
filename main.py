# ==========================================
# AI DATA SCIENTIST AGENT (FINAL PIPELINE)
# ==========================================

import pandas as pd

# === IMPORT YOUR MODULES ===
from tools.data_loader import load_data
from tools.preprocessing import preprocess_with_reflection
from tools.model_selector import get_models
from tools.train_and_evaluate import train_and_evaluate
from agents.problem_detector import detect_problem
from agents.model_selector_agent import select_best


# ==============================
# QUICK TRAIN FUNCTION (for reflection loop)
# ==============================
def quick_train_eval(df, target, problem_type):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.metrics import mean_squared_error, accuracy_score

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    if "regression" in problem_type:
        model = RandomForestRegressor(n_estimators=50)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return mean_squared_error(y_test, preds)
    else:
        model = RandomForestClassifier(n_estimators=50)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return accuracy_score(y_test, preds)


# ==============================
# FULL AUTOML LOOP
# ==============================
def improve_loop(df, target, models, problem_type, max_iter=3):
    best_model = None
    best_score = float("inf") if "regression" in problem_type else 0

    for i in range(max_iter):
        print(f"\n🔁 Model Iteration {i+1}")

        results = train_and_evaluate(df, target, models, problem_type)

        name, best = select_best(results, problem_type)

        print(f"Best Model: {name}, Score: {best['score']}")

        if "regression" in problem_type:
            if best["score"] < best_score:
                best_score = best["score"]
                best_model = best
        else:
            if best["score"] > best_score:
                best_score = best["score"]
                best_model = best

    return best_model


# ==============================
# REPORT GENERATOR
# ==============================
def generate_report(problem_type, best_model):
    report = {
        "Problem Type": problem_type,
        "Best Score": best_model["score"],
        "Model Info": str(best_model["model"])
    }
    return report


# ==============================
# MAIN EXECUTION
# ==============================
if __name__ == "__main__":

    # ===== CONFIG =====
    FILE_PATH = "data.csv"     # change this
    TARGET = "price"           # change this

    print("\n📂 Loading dataset...")
    df = load_data(FILE_PATH)

    print("\n🧠 Detecting problem type...")
    problem_type = detect_problem(df, TARGET)
    print("Detected:", problem_type)

    print("\n🧹 Intelligent preprocessing (LLM + Reflection)...")
    df = preprocess_with_reflection(
        df,
        TARGET,
        train_fn=lambda d: quick_train_eval(d, TARGET, problem_type),
        problem_type=problem_type,
        iterations=2
    )

    print("\n⚙️ Selecting models...")
    models = get_models(problem_type)

    print("\n🤖 Running AutoML loop...")
    best_model = improve_loop(df, TARGET, models, problem_type)

    print("\n🏆 FINAL BEST SCORE:", best_model["score"])

    print("\n🧾 Generating report...")
    report = generate_report(problem_type, best_model)

    print("\n📄 FINAL REPORT:")
    for k, v in report.items():
        print(f"{k}: {v}")