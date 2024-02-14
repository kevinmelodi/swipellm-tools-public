import json
from openai import OpenAI
import streamlit as st
import time
import requests





st.title("Marketing research assistant w openai")
st.caption("An OpenAI custom GPT with web browsing by Perplexity. Ask me to research a website 🔎")

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
    i = 0
    while i < len(message):
        # If the current character is a special Markdown character, we might need to handle it differently
        if message[i] in ['*', '_', '`', '#', '-', '+', '[', '!']:
            # Look ahead to find the end of the Markdown formatting
            if message[i] == '*' or message[i] == '_':  # Bold or italic
                next_char = '*' if message[i] == '*' else '_'
                end_format = message.find(next_char, i + 1)
                if end_format != -1:
                    # Output the formatted text at once
                    yield message[i:end_format + 1]
                    i = end_format
            elif message[i] == '`':  # Inline code
                end_format = message.find('`', i + 1)
                if end_format != -1:
                    yield message[i:end_format + 1]
                    i = end_format
            elif message[i] in ['#', '-', '+']:  # Headers or lists
                # Find the end of the line
                end_of_line = message.find('\n', i + 1)
                if end_of_line != -1:
                    yield message[i:end_of_line + 1]
                    i = end_of_line
            elif message[i] == '[' or message[i] == '!':  # Links or images
                end_bracket = message.find(']', i + 1)
                start_parenthesis = message.find('(', end_bracket + 1)
                end_parenthesis = message.find(')', start_parenthesis + 1)
                if end_bracket != -1 and start_parenthesis != -1 and end_parenthesis != -1:
                    yield message[i:end_parenthesis + 1]
                    i = end_parenthesis
        else:
            # Output character by character
            yield message[i]
            time.sleep(0.005)
        i += 1


if "thread" not in st.session_state:
    thread = client.beta.threads.create()
    threadId = str(thread.id)
    st.session_state['thread'] = threadId


thread_messages = client.beta.threads.messages.list(st.session_state['thread'])
initial_message_ids = [message.id for message in thread_messages.data]

for message in reversed(thread_messages.data):
    with st.chat_message(message.role):
        st.markdown(message.content[0].text.value)


if prompt := st.chat_input("What should we research today?"):
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
                st.markdown('Conducting external research, this may take up to a minute')
                run_steps = client.beta.threads.runs.steps.list(thread_id=st.session_state['thread'],run_id=run.id)
                for step in run_steps.data:
                    for tool_call in step.step_details.tool_calls:
                        if tool_call.type == 'function':
                            toolCallId = tool_call.id
                            if tool_call.function.name == 'research_url':
                                function_arguments = json.loads(tool_call.function.arguments)
                                research_target_url = function_arguments.get('URL')
                                research_response = research_company(research_target_url)
                                research_content = json.loads(research_response.content)
                                research_content_text = str(research_content["choices"][0]["message"]["content"])
                                
                                run = client.beta.threads.runs.submit_tool_outputs(
                                        thread_id=st.session_state['thread'],
                                        run_id=run.id,
                                        tool_outputs=[
                                            {
                                            "tool_call_id": toolCallId,
                                            "output": research_content_text
                                            }
                                        ]
                                        )

    # Fetch the new messages after the assistant's response
    new_thread_messages = client.beta.threads.messages.list(st.session_state['thread'])

    # Determine the new messages by comparing IDs
    new_message_ids = set(message.id for message in new_thread_messages.data) - set(initial_message_ids)

    # Display only new messages
    for message in reversed(new_thread_messages.data):
        if message.id in new_message_ids:
            with st.chat_message(message.role):
                st.write_stream(fake_stream(message.content[0].text.value))
    

