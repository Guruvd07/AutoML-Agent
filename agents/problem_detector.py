from llm.groq_llm import GroqLLM

llm = GroqLLM()

def detect_problem(df, target_column):
    # ✅ deterministic + zero API dependency

    if df[target_column].dtype == "object":
        return "classification"

    unique_values = df[target_column].nunique()

    if unique_values < 20:
        return "classification"
    else:
        return "regression"