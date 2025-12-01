import streamlit as st
from langchain_helper import execute_user_query

st.title("RAG Chatbot Demo")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.chat_history = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

query = st.chat_input("Ask something...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    answer = execute_user_query(query, st.session_state.chat_history)

    st.session_state.chat_history.append(("user", query))
    st.session_state.chat_history.append(("assistant", answer))

    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.write(answer)
