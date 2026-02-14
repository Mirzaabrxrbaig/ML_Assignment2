
import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics import confusion_matrix, classification_report

st.title("ML Classification App")

uploaded = st.file_uploader("Upload test CSV", type=["csv"])

model_names = [f.replace(".pkl","") for f in os.listdir("saved_models")]
selected = st.selectbox("Select model", model_names)

if uploaded:
    data = pd.read_csv(uploaded)
    model = joblib.load(f"saved_models/{selected}.pkl")

    if "Exited" in data.columns:
        y_true=data["Exited"]
        X=data.drop("Exited",axis=1)
    else:
        y_true=None
        X=data

    predictions=model.predict(X)

    st.write("Predictions:")
    st.write(predictions)

    if y_true is not None:
        st.write("Confusion Matrix")
        st.write(confusion_matrix(y_true,predictions))

        st.write("Classification Report")
        st.text(classification_report(y_true,predictions))
