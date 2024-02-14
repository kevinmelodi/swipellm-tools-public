import json
from openai import OpenAI
import streamlit as st
import time
import requests

st.title("OpenAI Assistants API toy demo")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
p_key = st.secrets["PERPLEXITY_API_KEY"]
assistantId = 'asst_l5xzlCbeq77iRJ1sfmhVsejE'

def research_company(company_url):
    template = f"Hey GPT! I’d like you to go to the following URL address ({company_url}) and research the company, its products, services, and About Us page."

    # Define the API URL
    url = "https://api.perplexity.ai/chat/completions"

    # Define the payload
    payload = {
        "model": "pplx-70b-online",
        "messages": [
            {
                "role": "user",
                "content": template
            }
        ]
    }

    # Define the headers
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {p_key}"
    }

    # Make the POST request to the API
    response = requests.post(url, json=payload, headers=headers)

    # Assuming you want to return the response from the function
    return response

# Streamed response emulator
def fake_stream(message):
    for word in message.split():
        yield word + " "
        time.sleep(0.05)




if "thread" not in st.session_state:
    thread = client.beta.threads.create()
    threadId = str(thread.id)
    st.session_state['thread'] = threadId


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
    with st.chat_message("user"):
        st.markdown(prompt)

    initial_message_ids.append(user_message.id)


    run = client.beta.threads.runs.create(
        thread_id=st.session_state['thread'],
        assistant_id=assistantId,
        )

    with st.spinner('Please wait while the assistant processes your input...'):
        while run.status != "completed":
            time.sleep(.3)  # Adjust sleep time if needed
            run = client.beta.threads.runs.retrieve(
                thread_id=st.session_state['thread'], run_id=run.id
            )
            if run.status == 'requires_action':
                st.markdown('Message requires action')
                run_steps = client.beta.threads.runs.steps.list(thread_id=st.session_state['thread'],run_id=run.id)
                for step in run_steps.data:
                    for tool_call in step.step_details.tool_calls:
                        if tool_call.type == 'function':
                            if tool_call.function.name == 'research_url':
                                function_arguments = json.loads(tool_call.function.arguments)
                                research_target_url = function_arguments.get('URL')
                                reasearch_response = research_company(research_target_url)
                                st.markdown(reasearch_response)
                break


    # Fetch the new messages after the assistant's response
    new_thread_messages = client.beta.threads.messages.list(st.session_state['thread'])

    # Determine the new messages by comparing IDs
    new_message_ids = set(message.id for message in new_thread_messages.data) - set(initial_message_ids)

    # Display only new messages
    for message in reversed(new_thread_messages.data):
        if message.id in new_message_ids:
            with st.chat_message(message.role):
                st.write_stream(fake_stream(message.content[0].text.value))

