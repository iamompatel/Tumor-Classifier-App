import streamlit as st
import joblib
import numpy as np

model = joblib.load("tumor_model.pkl")
features = joblib.load("feature_names.pkl")
stats = joblib.load("feature_stats.pkl")

st.set_page_config(
    page_title="Tumor Classifier",   # browser tab text
    page_icon="🔬",                # browser tab icon
    layout="centered"              # or "wide" for full width
)

st.title("Breast Cancer Tumor Classifier")
st.write("Adjust the measurements, then click Predict.")

st.sidebar.header("Tumor Measurements")
inputs = []
for i, name in enumerate(features):
    value = st.sidebar.slider(
        name,
        float(stats["min"][i]),
        float(stats["max"][i]),
        float(stats["mean"][i]),
    )
    inputs.append(value)

if st.button("🔍 Predict", use_container_width=True):  
    X_new = [inputs]
    prediction = model.predict(X_new)[0]
    proba = model.predict_proba(X_new)[0]

    st.divider()                         
    col1, col2 = st.columns(2)          

    if prediction == 1:
        conf = proba[1]
        with col1:
            st.success("### Benign ✓")     
        with col2:
            st.metric("Confidence", f"{conf*100:.1f}%")  
    else:
        conf = proba[0]
        with col1:
            st.error("### Malignant ⚠")
        with col2:
            st.metric("Confidence", f"{conf*100:.1f}%")

    st.progress(float(conf))         
    st.caption("⚕️ Educational demo only — not medical advice.")