import streamlit as st
import requests
import time

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

def stream_response(response_text):
    for i in range(0, len(response_text), 5):  # Adjust chunk size as needed
        yield response_text[:i + 5]

st.title("Chat with Zebra Llama")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for chat in st.session_state.chat_history:
    st.write(chat, unsafe_allow_html=True)

user_input = st.text_input("User:")
st.session_state.messages.append({"role": "user", "content": user_input})

def response_generator(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


if st.button("Send"):
    if user_input:
        st.session_state.chat_history.append(f"<b>User:</b> {user_input}")
        with st.spinner("Zebra Llama is processing..."):
            response_text = get_response(user_input)
            with st.chat_message("Zebra-Llama"):
                response_text_stream = st.write_stream(response_generator(response_text))
            st.session_state.messages.append({"role": "Zebra-Llama", "content": response_text})

            
        st.rerun()
