import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tools.preprocessing import preprocess
from tools.model_selector import get_models
from tools.train_and_evaluate import train_and_evaluate
from agents.problem_detector import detect_problem


# ===================================
# PAGE
# ===================================

st.set_page_config(
    page_title="AI Data Scientist Agent",
    layout="wide"
)

st.title("🤖 AI Data Scientist Agent")


# ===================================
# SESSION STATE INIT
# ===================================

defaults = {
    "results":None,
    "problem_type":None,
    "processed_df":None,
    "best_model_name":None,
    "selected_model":"None",
    "scaler":None,
    "feature_columns":None,
    "raw_feature_columns":None,
    "target":None
}

for k,v in defaults.items():
    if k not in st.session_state:
        st.session_state[k]=v


# ===================================
# FILE UPLOAD
# ===================================

uploaded_file = st.file_uploader(
    "📂 Upload CSV",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())


    target = st.selectbox(
        "🎯 Select Target Column",
        df.columns
    )


    feature_cols = st.multiselect(
        "🧩 Select Feature Columns",
        [c for c in df.columns if c != target],
        default=[c for c in df.columns if c != target]
    )


    problem_type = detect_problem(df,target)


    # threshold for classification
    threshold = 0.50

    if problem_type=="classification":
        threshold = st.slider(
            "Decision Threshold",
            0.1,
            0.9,
            0.5,
            0.05
        )


    # ===============================
    # RUN AGENT
    # ===============================

    if st.button("🚀 Run AI Agent"):

        df_filtered = df[feature_cols+[target]]

        # NEW PREPROCESS
        processed_df, scaler, feature_columns = preprocess(
            df_filtered,
            target
        )

        models = get_models(problem_type)

        results = train_and_evaluate(
            processed_df,
            target,
            models,
            problem_type,
            threshold
        )


        # best model logic
        if problem_type=="regression":

            best_name = min(
                results,
                key=lambda x: results[x]["mse"]
            )

        else:

            best_name = max(
                results,
                key=lambda x: results[x]["accuracy"]
            )


        # SAVE EVERYTHING
        st.session_state.results = results
        st.session_state.problem_type = problem_type
        st.session_state.processed_df = processed_df
        st.session_state.best_model_name = best_name
        st.session_state.selected_model = best_name

        st.session_state.scaler = scaler
        st.session_state.feature_columns = feature_columns
        st.session_state.raw_feature_columns = feature_cols
        st.session_state.target = target



# ===================================
# SHOW RESULTS
# ===================================

if st.session_state.results:

    st.success("✅ Process Completed!")


    results = st.session_state.results
    problem_type = st.session_state.problem_type


    # ===================================
    # MODEL COMPARISON
    # ===================================

    st.subheader("📊 Model Comparison")


    rows=[]

    for name,res in results.items():

        if problem_type=="regression":

            rows.append({
                "Model":name,
                "MSE":round(res["mse"],2),
                "RMSE":round(np.sqrt(res["mse"]),2),
                "R2":round(res["r2"],3)
            })

        else:

            rows.append({
                "Model":name,
                "Accuracy":round(res["accuracy"],3),
                "Precision":round(res["precision"],3),
                "Recall":round(res["recall"],3),
                "F1":round(res["f1"],3),
                "ROC-AUC":round(res["roc_auc"],3)
            })


    compare_df = pd.DataFrame(rows)

    st.dataframe(
        compare_df,
        width="stretch"
    )


    # ===================================
    # MODEL SELECT (FIXED)
    # ===================================

    names=list(results.keys())

    default_index=names.index(
        st.session_state.selected_model
    )


    selected_model = st.selectbox(
        "🏆 Select Model",
        names,
        index=default_index,
        key="model_picker"
    )


    st.session_state.selected_model=selected_model


    model_data = results[selected_model]


    # ===================================
    # METRICS
    # ===================================

    st.subheader("📈 Selected Model Metrics")


    if problem_type=="regression":

        st.write(
            f"MSE: {model_data['mse']:.2f}"
        )

        st.write(
            f"RMSE: {np.sqrt(model_data['mse']):.2f}"
        )

        st.write(
            f"R² Score: {model_data['r2']:.3f}"
        )


    else:

        st.write(
            f"Accuracy: {model_data['accuracy']:.2f}"
        )

        st.write(
            f"Precision: {model_data['precision']:.2f}"
        )

        st.write(
            f"Recall: {model_data['recall']:.2f}"
        )

        st.write(
            f"F1 Score: {model_data['f1']:.2f}"
        )

        st.write(
            f"ROC-AUC: {model_data['roc_auc']:.2f}"
        )


        st.subheader("📊 Confusion Matrix")

        cm = pd.DataFrame(
            model_data["confusion_matrix"],
            index=["Actual0","Actual1"],
            columns=["Pred0","Pred1"]
        )

        st.dataframe(
            cm,
            width="stretch"
        )


# ===================================
# DOWNLOAD TRAINED MODEL
# ===================================

    st.subheader("💾 Download Trained Model")

    artifact = {
        "model": model_data["model"],
        "scaler": st.session_state["scaler"],
        "features": st.session_state["feature_columns"]
    }

    model_bytes = pickle.dumps(artifact)

    st.download_button(
        label="⬇ Download Model Pipeline",
        data=model_bytes,
        file_name="model_pipeline.pkl",
        mime="application/octet-stream"
    )

    # ===================================
    # DATA PREVIEW
    # ===================================

    st.subheader("📊 Processed Dataset Preview")

    st.dataframe(
        st.session_state.processed_df.head(),
        width="stretch"
    )


    # ===================================
    # PREDICTION
    # ===================================

    if "scaler" in st.session_state:

        st.subheader("🔮 Make Prediction")

        raw_features = st.session_state.raw_feature_columns

        user_inputs={}

        for col in raw_features:

            user_inputs[col]=st.number_input(
                col,
                value=0.0
            )


        if st.button("Predict"):

            input_df = pd.DataFrame(
                [user_inputs]
            )


            # USE TRAINING SCALER
            input_processed = preprocess(
                input_df,
                scaler=st.session_state.scaler,
                fit_scaler=False,
                training_columns=
                st.session_state.feature_columns
            )


            model = model_data["model"]

            pred = model.predict(
                input_processed
            )[0]


            if problem_type=="classification":

                if pred==1:
                    st.success(
                        "Prediction: Positive Class"
                    )
                else:
                    st.error(
                        "Prediction: Negative Class"
                    )

            else:

                st.success(
                    f"Predicted Value: {pred:.2f}"
                )