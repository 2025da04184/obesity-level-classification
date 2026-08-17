import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Obesity Level Classifier", layout="wide")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
NEEDS_SCALING = {"Logistic Regression", "kNN"}
MODEL_FILES = {
    "Logistic Regression": "Logistic_Regression.pkl",
    "Decision Tree":       "Decision_Tree.pkl",
    "kNN":                 "kNN.pkl",
    "Naive Bayes":         "Naive_Bayes.pkl",
    "Random Forest":       "Random_Forest.pkl",
}

@st.cache_resource
def load_model(name):
    return joblib.load(os.path.join(MODEL_DIR, MODEL_FILES[name]))

@st.cache_resource
def load_scaler():
    return joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

@st.cache_resource
def load_label_encoder():
    return joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

def compute_metrics(name, X_test, y_test, scaler):
    m = load_model(name)
    X = scaler.transform(X_test) if name in NEEDS_SCALING else X_test.values
    y_pred = m.predict(X)
    y_prob = m.predict_proba(X)
    return {
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "AUC":       round(roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro"), 4),
        "Precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred, average="weighted"), 4),
        "F1":        round(f1_score(y_test, y_pred, average="weighted"), 4),
        "MCC":       round(matthews_corrcoef(y_test, y_pred), 4),
    }

# Sidebar
with st.sidebar:
    st.title("Obesity Level Classifier")
    st.markdown("**Dataset:** Obesity Levels — Kaggle")
    st.markdown("**Task:** Multi-Class Classification (7 levels)")
    st.markdown("**Instances:** 2,111 | **Features:** 16")
    st.markdown("---")
    selected_model = st.selectbox("Select Model", list(MODEL_FILES.keys()))
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload test_data.csv", type=["csv"])

# Main
st.title("Obesity Level Prediction")
st.write("This app evaluates machine learning models that predict a person's obesity level based on dietary habits and physical activity.")
st.markdown("---")

tab1, tab2 = st.tabs(["Model Evaluation", "All Models Comparison"])

if uploaded_file is None:
    with tab1:
        st.info("Upload test_data.csv from the sidebar to evaluate the selected model.")
    with tab2:
        st.info("Upload test_data.csv from the sidebar to see the comparison table.")
else:
    df = pd.read_csv(uploaded_file)
    if "target" not in df.columns:
        st.error("The uploaded file must contain a 'target' column.")
        st.stop()

    X_test = df.drop("target", axis=1)
    y_test = df["target"].astype(int)
    le     = load_label_encoder()
    scaler = load_scaler()
    class_names = le.classes_

    with tab2:
        st.subheader("All Models — Evaluation on Uploaded Test Set")
        all_metrics = {name: compute_metrics(name, X_test, y_test, scaler) for name in MODEL_FILES}
        comparison_df = pd.DataFrame(all_metrics).T
        st.dataframe(comparison_df.style.background_gradient(cmap="Blues", axis=0),
                     use_container_width=True)
        best = comparison_df["Accuracy"].idxmax()
        st.success(f"Best performing model: **{best}**")

    with tab1:
        model   = load_model(selected_model)
        X_input = scaler.transform(X_test) if selected_model in NEEDS_SCALING else X_test.values
        y_pred  = model.predict(X_input)
        y_prob  = model.predict_proba(X_input)

        st.subheader(f"Results for: {selected_model}")
        st.caption(f"{len(df)} test samples | {X_test.shape[1]} features")

        metrics = {
            "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "AUC":       round(roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro"), 4),
            "Precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "Recall":    round(recall_score(y_test, y_pred, average="weighted"), 4),
            "F1 Score":  round(f1_score(y_test, y_pred, average="weighted"), 4),
            "MCC":       round(matthews_corrcoef(y_test, y_pred), 4),
        }

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        for col, (k, v) in zip([col1, col2, col3, col4, col5, col6], metrics.items()):
            col.metric(k, v)

        st.markdown("")
        st.dataframe(pd.DataFrame([metrics]), use_container_width=True, hide_index=True)

        st.markdown("---")
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=class_names, yticklabels=class_names, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            plt.xticks(rotation=35, ha="right", fontsize=8)
            plt.yticks(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_right:
            st.subheader("Classification Report")
            report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
            report_df = pd.DataFrame(report_dict).T.round(2)
            st.dataframe(report_df, use_container_width=True)
