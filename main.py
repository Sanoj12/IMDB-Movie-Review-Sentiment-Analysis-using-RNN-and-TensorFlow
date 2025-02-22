import numpy as np
import tensorflow as tf
import streamlit as st

from keras.models import load_model

import joblib

from tensorflow.keras.preprocessing.sequence import pad_sequences


model = load_model("model.h5")
tokenizer = joblib.load("tokenizer.pkl")




def predictive_system(review):
    sequences = tokenizer.texts_to_sequences([review])
    padded_sequences = pad_sequences(sequences ,maxlen =200)
    prediction = model.predict(padded_sequences)
   
    #print(prediction[0][0])
    return prediction[0][0]


####streamlit app

st.title('IMDB Movie Review Sentiment Analysis')
st.write("Enter a Movie Review")


#user inout
user_input = st.text_area("Movie Review.....")

if st.button("Predict"):
    result = predictive_system(user_input)
    sentiment = "POSTIVE" if result > 0.5 else "NEGATIVE"
    st.write(f"The sentiment of the movie review is: {sentiment}")
    st.write(f"Prediction Score: {result}")

else:
    st.write("Waiting for user input...")