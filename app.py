import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="Spam Mesaage Detection", layout="centered")

st.title("Spam Message Detector")

@st.cache_resource
def load_assets():
    model = load_model("model_spam_detector.keras")
    with open("model_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer    

model, tokenizer = load_assets()     

message = st.text_area("Enter the message to test")

if st.button("check"):
    if message:
        seq = tokenizer.texts_to_sequences([message])
        padded = pad_sequences(seq, maxlen=50, padding="post")
        prediction = model.predict(padded)[0][0]


        if prediction > 0.5:
            st.error(f"Spam ({prediction:.2%} confidence)")
        else:
            st.success(f"Ham / Not Spam ({(1-prediction):.2%} confidence)")
    else:
        st.warning("Please enter a message!")
