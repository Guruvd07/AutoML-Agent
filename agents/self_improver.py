# agents/self_improver.py

def improve_loop(df, target, models, problem_type, max_iter=3):
    best_model = None
    best_score = float("inf") if problem_type == "regression" else 0

    for i in range(max_iter):
        print(f"\nIteration {i+1}")

        results = train_and_evaluate(df, target, models, problem_type)

        name, best = select_best(results, problem_type)

        print(f"Best model: {name}, Score: {best['score']}")

        if problem_type == "regression":
            if best["score"] < best_score:
                best_score = best["score"]
                best_model = best
        else:
            if best["score"] > best_score:
                best_score = best["score"]
                best_model = best

    return best_model