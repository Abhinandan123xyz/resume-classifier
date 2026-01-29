import streamlit as st
import pickle

model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

st.title("Resume Classifier")

text = st.text_area("Paste resume text here")

if st.button("Predict"):
    vect = vectorizer.transform([text])
    pred = model.predict(vect)
    st.success(f"Predicted Category: {pred[0]}")
