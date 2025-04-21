import streamlit as st
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from agent import agent

st.set_page_config(page_title="StockBot")
st.title("Stockbot - Your Financial Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

#Displaychat history
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)

user_query = st.chat_input("Ask me anything about stocks, finance...")

if user_query:
    st.chat_message("user").markdown(user_query)
    st.session_state.chat_history.append(("user", user_query))

    with st.chat_message("assistant"):
        callback_handler = StreamlitCallbackHandler(st.container())
        with st.spinner("Thinking..."):
            response = agent.run(user_query, callbacks=[callback_handler])
        st.markdown(response)
        st.session_state.chat_history.append(("assistant", response))