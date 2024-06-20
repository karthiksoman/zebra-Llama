import streamlit as st
import requests

def get_response(user_input):
    url = "https://jfn1so7p7a.execute-api.us-west-1.amazonaws.com/Prod/inference"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "text": user_input
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()

st.title("Chat EDS with Zebra Llama")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for chat in st.session_state.chat_history:
    st.write(chat)

user_input = st.text_input("User:")

if st.button("Send"):
    if user_input:
        st.session_state.chat_history.append(f"User: {user_input}")
        with st.spinner("Zebra Llama is processing ..."):
            response = get_response(user_input)
            st.session_state.chat_history.append(f"zebra-Llama: {response}")
        st.rerun()
