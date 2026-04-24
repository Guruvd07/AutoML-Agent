# 🤖 AutoML Agent (Agentic AI Data Scientist)

An end-to-end **AutoML Agent** that automatically:

- Detects problem type (Regression / Classification)
- Preprocesses uploaded datasets
- Trains multiple ML models
- Compares models automatically
- Selects best model
- Evaluates with metrics + visualizations
- Generates predictions interactively
- Saves trained models as `.pkl`
- Deployed live on Render

---

## 🚀 Live Demo
https://automl-agent-r7yp.onrender.com

> Free Render instance may take ~30–50 seconds to wake if idle.

---

# ✨ Features

## Automated Data Science Pipeline
✔ Missing value handling  
✔ Categorical encoding  
✔ Feature scaling  
✔ Automatic problem detection  
✔ Multi-model training  
✔ Model selection

---

## Supported Models

### Regression
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

Metrics:
- MSE
- RMSE
- R² Score

---

### Classification
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

Metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- ROC Curve Comparison

---

## 📊 Model Comparison Dashboard
Compare all models in UI:

- Performance table
- Select model dropdown
- Compare metrics
- Confusion matrix
- ROC curves

---

## 🔮 Prediction Interface
Users can:

- Enter custom feature values
- Get live predictions
- Use trained selected model
- Download / save models as `.pkl`

---

# 🧠 Why “Agentic AI”?
This project uses autonomous agent-style workflow:

1. Problem Detection Agent  
2. Preprocessing Agent  
3. Model Selection Agent  
4. Training & Evaluation Agent  
5. Prediction Agent  

The system makes decisions automatically with minimal user intervention.

---

# 🏗 Tech Stack

## Frontend
- Streamlit

## ML / Data
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib

## Deployment
- Render

## Monitoring
- UptimeRobot (keep-alive pings)

---

# 📂 Project Structure

```bash
AutoML-Agent/
│
├── agents/
│   ├── problem_detector.py
│   └── model_selector_agent.py
│
├── tools/
│   ├── preprocessing.py
│   ├── train_and_evaluate.py
│   └── model_selector.py
│
├── saved_models/
├── app.py
├── requirements.txt
└── runtime.txt
```

---

# ⚙ Installation

## Clone repo

```bash
git clone https://github.com/Guruvd07/AutoML-Agent.git
cd AutoML-Agent
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run locally

```bash
streamlit run app.py
```

---

# 🖥 Example Datasets Tested
- Fuel Consumption (Regression)
- Salary Prediction
- Loan Approval (Classification)

---

# 📸 Screenshots
(Add screenshots here)

- Upload dataset
- Model comparison
- Confusion matrix
- Prediction interface

---

# 🔥 Future Improvements
Planned upgrades:

- Hyperparameter Optimization Agent
- Feature Engineering Agent
- Explainable AI (SHAP)
- LLM-based Insight Agent
- Time Series AutoML
- Model Deployment API
- Multi-agent orchestration (LangGraph / CrewAI)

---

# 📈 Example Workflow

Upload CSV  
↓  
Select target  
↓  
Choose features  
↓  
AutoML Agent trains models  
↓  
Best model selected  
↓  
Compare results  
↓  
Make predictions  
↓  
Save model

---

# 🌐 Deployment

Deployed on Render:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

Python:
```txt
python-3.11.9
```

---

## ❤️ If you like this project
Star the repo ⭐

---

## Author
**Guru Dahiphale**

Computer Engineering | Data Science | Agentic AI

GitHub:
https://github.com/Guruvd07
