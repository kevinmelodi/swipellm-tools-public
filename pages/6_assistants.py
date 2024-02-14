from openai import OpenAI
import streamlit as st
import time

st.title("ChatGPT-like clone")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
assistantId = 'asst_dzvy5mIYrwNScAeFUqnN9Llt'

# Initialize session state variables
if "thread" not in st.session_state:
    thread = client.beta.threads.create()
    st.session_state['thread'] = str(thread.id)
    st.session_state['displayed_messages'] = set()  # Store message IDs that are displayed

# Display thread messages
thread_messages = client.beta.threads.messages.list(st.session_state['thread'])

# Loop through messages in reverse order and display if not displayed already
for message in reversed(thread_messages.data):
    if message.id not in st.session_state['displayed_messages']:
        st_object = {"role": message.role, "content": message.content[0].text.value}
        with st.chat_message(st_object["role"]):
            st.markdown(st_object["content"])
        st.session_state['displayed_messages'].add(message.id)  # Mark message as displayed

# User input
if prompt := st.chat_input("What is up?"):
    message = client.beta.threads.messages.create(
        thread_id=st.session_state['thread'],
        role="user",
        content=prompt
    )
    
    # Assume a message ID is returned and add to displayed messages
    st.session_state['displayed_messages'].add(message.id)

    with st.chat_message("assistant"):
        run = client.beta.threads.runs.create(
            thread_id=st.session_state['thread'],
            assistant_id=assistantId,
            instructions="Please address the user as Jane Doe. The user has a premium account."
        )

    # Wait for the assistant's response
    with st.status("Starting work...", expanded=False) as status_box:
        while run.status != "completed":
            time.sleep(5)
            status_box.update(label=f"{run.status}...", state="running")
            run = client.beta.threads.runs.retrieve(
                thread_id=st.session_state['thread'], run_id=run.id
            )
        # Assume run completion means new message is available
        # Refresh thread messages to get the latest
        thread_messages = client.beta.threads.messages.list(st.session_state['thread'])
        
    # Display new messages after user input
    for message in reversed(thread_messages.data):
        if message.id not in st.session_state['displayed_messages']:
            st_object = {"role": message.role, "content": message.content[0].text.value}
            with st.chat_message(st_object["role"]):
                st.markdown(st_object["content"])
            st.session_state['displayed_messages'].add(message.id)  # Mark message as displayed
