def select_best(results, problem_type):
    if problem_type == "regression":
        return min(results.items(), key=lambda x: x[1]["mse"])
    else:
        return max(results.items(), key=lambda x: x[1]["accuracy"])