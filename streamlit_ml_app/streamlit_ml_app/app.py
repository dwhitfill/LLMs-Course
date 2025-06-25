import streamlit as st
import pickle
import os

# Define the base path to the models folder
MODEL_PATH = os.path.join("models")

@st.cache_resource
def load_models():
    with open(os.path.join(MODEL_PATH, "tfidf_vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    with open(os.path.join(MODEL_PATH, "decision_tree_model.pkl"), "rb") as f:
        dt_model = pickle.load(f)
    with open(os.path.join(MODEL_PATH, "adaboost_model.pkl"), "rb") as f:
        ada_model = pickle.load(f)
    with open(os.path.join(MODEL_PATH, "svm_grid_model.pkl"), "rb") as f:
        svm_model = pickle.load(f)
    return vectorizer, dt_model, ada_model, svm_model

vectorizer, dt_model, ada_model, svm_model = load_models()

# App UI
st.title("AI vs Human Text Classifier ")
st.write("Paste any paragraph, and this app will classify whether it's AI- or Human-written using 3 different models.")

user_input = st.text_area(" Enter your text below:", height=150)

if user_input:
    # Make predictions directly (the pipelines include vectorizers)
    dt_pred = dt_model.predict([user_input])[0]
    ada_pred = ada_model.predict([user_input])[0]
    svm_pred = svm_model.predict([user_input])[0]

    # Label mapping
    label_map = {0: "Human", 1: "AI"}

    st.subheader(" Classification Results")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("** Decision Tree**")
        st.success(f"Prediction: **{label_map[dt_pred]}**")
        if hasattr(dt_model, "predict_proba"):
            prob = dt_model.predict_proba([user_input])[0]
            st.write(f"Confidence — Human: {prob[0]:.2f}, AI: {prob[1]:.2f}")

    with col2:
        st.markdown("** AdaBoost**")
        st.success(f"Prediction: **{label_map[ada_pred]}**")
        if hasattr(ada_model, "predict_proba"):
            prob = ada_model.predict_proba([user_input])[0]
            st.write(f"Confidence — Human: {prob[0]:.2f}, AI: {prob[1]:.2f}")

    with col3:
        st.markdown("** SVM**")
        st.success(f"Prediction: **{label_map[svm_pred]}**")
        if hasattr(svm_model, "decision_function"):
            decision_score = svm_model.decision_function([user_input])[0]
            st.write(f"Decision Score: {decision_score:.2f}")
