from openai import OpenAI
import streamlit as st
import time

st.title("ChatGPT-like clone")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
assistantId = 'asst_dzvy5mIYrwNScAeFUqnN9Llt'


if "thread" not in st.session_state:
    thread = client.beta.threads.create()
    threadId = str(thread.id)
    st.session_state['thread'] = threadId
    st.session_state['thread']
st.write(st.session_state)


thread_messages = client.beta.threads.messages.list(st.session_state['thread'])

initial_message_ids = [message.id for message in thread_messages.data]

for message in reversed(thread_messages.data):
    with st.chat_message(message.role):
        st.markdown(message.content[0].text.value)


if prompt := st.chat_input("What is up?"):
    user_message = client.beta.threads.messages.create(
        thread_id=st.session_state['thread'],
        role="user",
        content=prompt
    )

    initial_message_ids.append(user_message.id)

    with st.chat_message("assistant"):
        run = client.beta.threads.runs.create(
            thread_id=st.session_state['thread'],
            assistant_id=assistantId,
            instructions="Please address the user as Jane Doe. The user has a premium account."
            )

    with st.status("Starting work...", expanded=False):
        while run.status != "completed":
            time.sleep(5)
            run = client.beta.threads.runs.retrieve(
                thread_id=st.session_state['thread'], run_id=run.id
            )

        # Fetch the new messages after the assistant's response
        new_thread_messages = client.beta.threads.messages.list(st.session_state['thread'])

        # Determine the new messages by comparing IDs
        new_message_ids = set(message.id for message in new_thread_messages.data) - set(initial_message_ids)

        # Display only new messages
        for message in reversed(new_thread_messages.data):
            if message.id in new_message_ids:
                st_object = {"role": message.role, "content": message.content[0].text.value}
                with st.chat_message(st_object["role"]):
                    st.markdown(st_object["content"])