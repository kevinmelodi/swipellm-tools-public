from openai import OpenAI
import streamlit as st

st.title("ChatGPT-like clone")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
assistantId = 'asst_dzvy5mIYrwNScAeFUqnN9Llt'

if 'initialized' not in st.session_state:
    # Set the 'initialized' key to True to indicate that initialization is done
    st.session_state['initialized'] = True


if "thread" not in st.session_state:
    thread = client.beta.threads.create()
    threadId = str(thread.id)
    st.session_state['thread'] = threadId
    st.session_state['thread']
st.write(st.session_state)


if prompt := st.chat_input("What is up?"):
    message = client.beta.threads.messages.create(
        thread_id=st.session_state['thread'],
        role="user",
        content=prompt
    )

    with st.chat_message("assistant"):
        run = client.beta.threads.runs.create(
            thread_id=st.session_state['thread'],
            assistant_id=assistantId,
            instructions="Please address the user as Jane Doe. The user has a premium account."
            )

    thread_messages = client.beta.threads.messages.list(st.session_state['thread'])

    for message in reversed(thread_messages.data):
        st_object = {}
        st_object["role"] = message.role
        for text in message.content:
            st_object["content"] = text.text.value
            with st.chat_message(st_object["role"]):
                st.markdown(st_object["content"])
