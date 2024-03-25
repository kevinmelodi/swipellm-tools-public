import json
from openai import OpenAI
import pandas as pd
import requests
import streamlit as st
import time

st.set_page_config(page_title="Melodi | Create Evaluation", page_icon='melodi_transparent.png', initial_sidebar_state='expanded',)

with st.sidebar:
    melodi_api_key = st.text_input("Melodi API Key", type="password")

def create_experiment(experiment_name, experiment_instructions, items, experiment_type='Bake-off', project=None, binary_version=None, template_type='default'):
    api_key = melodi_api_key
    base_url = "https://app.melodi.fyi/api/external/experiments?apiKey="
    
    if template_type == 'conversational':
        base_url = "https://app.melodi.fyi/api/external/experiments/templates/conversational?apiKey="
    elif template_type == 'json':
        base_url = "https://app.melodi.fyi/api/external/experiments/templates/json?apiKey="
    
    url = base_url + api_key
    headers = {"Content-Type": "application/json"}

    # Base data structure
    data = {
        "experiment": {
            "name": experiment_name,
            "instructions": experiment_instructions
        }
    }

    # Add 'project' to the experiment data if provided
    if project is not None:
        data["experiment"]["project"] = project

    # Adjust data structure based on experiment type
    if experiment_type == 'Bake-off':
        data["comparisons"] = items
    elif experiment_type == 'Binary':
        data["samples"] = items
        if project is not None and binary_version is not None:
            data["experiment"]["version"] = binary_version
    else:
        raise ValueError("Invalid experiment type. Choose 'Bake-off' or 'Binary'.")

    # Make the POST request
    response = requests.post(url, headers=headers, data=json.dumps(data))

    return response

def json_to_markdown_recursive(json_dict, level=1):
    """
    This function recursively converts a JSON dictionary to a markdown string.
    It handles nested dictionaries by increasing the heading level and 
    shows keys with no values as "_no result_".

    :param json_dict: Dictionary containing the message logs
    :param level: Current heading level
    :return: A string formatted in markdown
    """
    markdown_str = ""
    for key, value in json_dict.items():
        # Use "No result" if the value is None or an empty list/string
        display_value = "_no value_" if (value is None or value == [] or value == "") else value

        # Handle non-boolean and non-empty values or non-dict, non-list
        if not isinstance(display_value, (dict, list, bool)):
            markdown_str += f"{'###' * level} {key}\n\n{display_value}\n\n"
        # Handle list and dict types
        elif isinstance(display_value, list):
            markdown_str += f"{'###' * level} {key}\n\n"
            if display_value:  # If the list is not empty
                for item in display_value:
                    # If the list item is a dict, call the function recursively increasing the level
                    if isinstance(item, dict):
                        markdown_str += json_to_markdown_recursive(item, level + 1)
                    else:
                        markdown_str += f"{item}\n"
            else:
                markdown_str += "_no value_\n"
            markdown_str += "\n"
        elif isinstance(display_value, dict):
            # If it's a dict, add a heading and call the function recursively increasing the level
            markdown_str += f"{'###' * level} {key}\n\n{json_to_markdown_recursive(display_value, level + 1)}"
        # Handle boolean values, showing them as Yes/No
        elif isinstance(display_value, bool):
            yes_no = 'Yes' if display_value else 'No'
            markdown_str += f"{'###' * level} {key}\n\n{yes_no}\n\n"

    return markdown_str.strip()
st.image('melodi_transparent.png', width=70)
st.title("Create Melodi Evaluation")
st.caption("Enter your Melodi API key in the sidebar to begin. Copy and paste results directly from ChatGPT or use a CSV for batch import. To learn about creating experiments via API, [read the API docs.](https://melodi.notion.site/Melodi-Experiments-API-08b6d362277d49e9aa167c75bce153a0)")


col_name, col_project = st.columns(2)

with col_name:
    experiment_name = st.text_input(
        "**Experiment name** *(Reviewers won't see this)*",
    )

with col_project:
    project = st.text_input(
            "**Project** *Optional*", help = "Projects are like folders for your experiments. They should group experiments related to the same model, task, or product feature."
        )



experiment_instructions = st.text_input(
        "**Instructions for reviewers**",
    )



if 'disabled' not in st.session_state:
    st.session_state['disabled'] = False

if 'GPT data' not in st.session_state:
    st.session_state['GPT data'] = False

if 'JSON data' not in st.session_state:
    st.session_state['JSON data'] = False

if 'CSV data' not in st.session_state:
    st.session_state['CSV data'] = False

col_radio, col_images = st.columns(2)


with col_radio: 
    st.session_state['eval_type'] = st.radio("**Evaluation Type**", ['Bake-off','Binary'], captions=['A/B test responses from two models', 'Pass/fail assess responses from one model'])

with col_images:

    if st.session_state['eval_type'] == 'Bake-off':
        st.image('images/bakeoff.png')
    else:
        st.image('images/binary.png')



st.header('Import data')
tab_csv, tab_GPT, tab_json, tab_manual = st.tabs([ "CSV Upload","GPT Conversation (JSONL)","JSONL", "Manual Entry"])

if st.session_state['eval_type'] == 'Bake-off':
    prompt_a_label = 'Model or Prompt Version Name (A)'
else:
    prompt_a_label = 'Model or Prompt Version Name'

with tab_manual:

    col1, col2 = st.columns(2)

    if 'help' not in st.session_state:
        st.session_state.help = "Reviewers won't see this"

    with col1:
        prompt_1 = st.text_input(
            prompt_a_label,
            value="Original Prompt",
            #disabled=st.session_state.disabled,
            help="Names can't be updated if data is present in the table."
        )

    with col2:
        if st.session_state['eval_type'] == 'Bake-off':
            prompt_2 = st.text_input(
                'Model or Prompt Version Name (B)',
                value="New Prompt",
                disabled=st.session_state.disabled,
                help="Names can't be updated if data is present in the table."
            )
        else:
            pass


    if st.session_state['eval_type'] == 'Bake-off':
        df = pd.DataFrame(columns=[prompt_1, prompt_2])
        
        config = {
            prompt_1 : st.column_config.TextColumn(f"Generated by {prompt_1}", width='large', required=True),
            prompt_2 : st.column_config.TextColumn(f" Generated by {prompt_2}", width='large', required=True),
        }
    else:
        df = pd.DataFrame(columns=["Responses"])
        
        config = {
            "Responses" : st.column_config.TextColumn(f"Generated by {prompt_1}", width='large', required=True),
        }

    samples = st.data_editor(df, column_config = config, num_rows='dynamic',hide_index=True)


    if len(samples.index) > 0:
        st.session_state.disabled = True
        st.session_state.help = "Prompt names can't be updated when data is present in the table."


with tab_csv:
    if st.session_state['eval_type'] == 'Bake-off':
        uploaded_file = st.file_uploader(label='For **bake-off evaluations** a header row must be provided to label the models (such as "New prompt" and "Old prompt"). Column 1 should contain responses generated from Prompt 1. Column 2 should contain resposnes generated from Prompt 2. ', type=['csv'])
        if uploaded_file is not None:
            st.session_state['file'] = True
            file = pd.read_csv(uploaded_file,header=0)
            num_columns = len(file.columns)
            if num_columns == 2:
                samples = file.iloc[:,0:2]
                prompt_1, prompt_2 = samples.columns.tolist()
            elif num_columns == 3:
                samples = file.iloc[:, 0:3]
                message, prompt_1, prompt_2 = samples.columns.tolist()
            st.dataframe(samples, hide_index=True)
    else:
        uploaded_file = st.file_uploader(label="For **binary evaluations**, a header row is required, but the header value is not used to create the evaluation. The file should be one column: a list of example LLM responses from one prompt/model.", type=['csv'])
        if uploaded_file is not None:
            st.session_state['file'] = True
            file = pd.read_csv(uploaded_file,header=0)
            samples = file.iloc[:,0:1]
            prompt_1 = file.columns[0]
            st.dataframe(samples, hide_index=True)
            st.session_state['CSV data'] = True

with tab_GPT:
    st.write("Data should follow the OpenAI fine tuning dataset structure. For details, [read the OpenAI fine tune docs](https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset)")
    if st.session_state['eval_type'] == 'Bake-off':
            col_jsonl_a, col_jsonl_b = st.columns(2)
            with col_jsonl_a:
                prompt_1 = st.text_input(
                prompt_a_label,
                value="Original Prompt",
                #disabled=st.session_state.disabled,
                help="Names can't be updated if data is present in the table.",
                key = "GPT prompt 1")
                
                uploaded_file_A = st.file_uploader(label='Conversation Samples (A)', type=['jsonl'], label_visibility="hidden")
                if uploaded_file_A is not None:
                    st.session_state['file'] = True
                    samples_A = []
                    for line in uploaded_file_A.read().decode('utf-8').splitlines():
                        samples_A.append(json.loads(line))
                    st.session_state['GPT data'] = True

            with col_jsonl_b:
                prompt_2 = st.text_input(
                'Model or Prompt Version Name (B)',
                value="New Prompt",
                disabled=st.session_state.disabled,
                help="Names can't be updated if data is present in the table.",
                key="GPT prompt 2")

                uploaded_file_B = st.file_uploader(label='Conversation Samples (B)', type=['jsonl'], label_visibility="hidden")
                if uploaded_file_B is not None:
                    st.session_state['file'] = True
                    samples_B = []
                    for line in uploaded_file_B.read().decode('utf-8').splitlines():
                        samples_B.append(json.loads(line))
    else:
        uploaded_file_A = st.file_uploader(label='Conversation Samples', type=['jsonl'])
        if uploaded_file_A is not None:
            st.session_state['file'] = True
            samples_A = []
            for line in uploaded_file_A.read().decode('utf-8').splitlines():
                samples_A.append(json.loads(line))
            st.session_state['GPT data'] = True

    if st.session_state['GPT data']:
        try:
            st.subheader("Conversation Data Preview",divider='rainbow')
            if st.session_state['eval_type'] == 'Bake-off':
                col_message_A_preview, col_message_B_preview = st.columns(2)
                with col_message_A_preview:
                    for message in samples_A[0]['messages']:
                        with st.chat_message(message['role']):
                            st.markdown(message['content'])
                with col_message_B_preview:
                    for message in samples_B[0]['messages']:
                        with st.chat_message(message['role']):
                            st.markdown(message['content'])
            else:
                for message in samples_A[0]['messages']:
                        with st.chat_message(message['role']):
                            st.markdown(message['content'])
        except:
            pass

with tab_json:
    st.write("JSONL files should have a valid JSON dictionary on each newline of the file.")
    if st.session_state['eval_type'] == 'Bake-off':
            col_jsonl_a, col_jsonl_b = st.columns(2)
            with col_jsonl_a:
                prompt_1 = st.text_input(
                prompt_a_label,
                value="Original Prompt",
                #disabled=st.session_state.disabled,
                help="Names can't be updated if data is present in the table.",
                key = "JSON prompt 1")
                
                uploaded_file_A = st.file_uploader(label='JSON Samples (A)', type=['jsonl'], label_visibility="hidden")
                if uploaded_file_A is not None:
                    st.session_state['file'] = True
                    samples_A = []
                    for line in uploaded_file_A.read().decode('utf-8').splitlines():
                        samples_A.append(json.loads(line))
                    st.session_state['JSON data'] = True

            with col_jsonl_b:
                prompt_2 = st.text_input(
                'Model or Prompt Version Name (B)',
                value="New Prompt",
                disabled=st.session_state.disabled,
                help="Names can't be updated if data is present in the table.",
                key="JSON prompt 2")

                uploaded_file_B = st.file_uploader(label='JSON Samples (B)', type=['jsonl'], label_visibility="hidden")
                if uploaded_file_B is not None:
                    st.session_state['file'] = True
                    samples_B = []
                    for line in uploaded_file_B.read().decode('utf-8').splitlines():
                        samples_B.append(json.loads(line))
    else:
        uploaded_file_A = st.file_uploader(label='JSON Samples', type=['jsonl'])
        if uploaded_file_A is not None:
            st.session_state['file'] = True
            samples_A = []
            for line in uploaded_file_A.read().decode('utf-8').splitlines():
                samples_A.append(json.loads(line))
            st.session_state['JSON data'] = True

    if st.session_state['JSON data']:
        try:
            st.subheader("JSON Data Preview",divider='rainbow')
            if st.session_state['eval_type'] == 'Bake-off':
                col_json_A_preview, col_json_B_preview = st.columns(2)
                with col_json_A_preview:
                    expander_A = st.expander("Preview data")
                    expander_A.markdown(json_to_markdown_recursive(samples_A[0]))
                with col_json_B_preview:
                    expander_B = st.expander("Preview data")
                    expander_B.markdown(json_to_markdown_recursive(samples_B[0]))
            else:
                expander = st.expander("Preview data")
                expander.markdown(json_to_markdown_recursive(samples_A[0]))
        except:
            pass

if st.button('Create Experiment'):

    if st.session_state['GPT data'] == True: ### HANDLE GPT DATA ####
        if st.session_state['eval_type'] == 'Bake-off':
            comparisons = []
            promptLabel = 'promptLabel'
            if project:
                promptLabel = 'version'
            else:
                project = None

            for thread_A, thread_B in zip(samples_A, samples_B): # Loop through rows with iterrows() 
                comparisons.append({"samples":[
                    {"response": thread_A, promptLabel: prompt_1}, 
                    {"response": thread_B, promptLabel: prompt_2} 
                ]})

            melodi_response = create_experiment(
                experiment_name,
                experiment_instructions,
                items=comparisons,
                experiment_type='Bake-off',
                project=project,
                template_type='conversational'
            )
        
        else: ### create binary (gpt)
            binary_samples = []
            for thread in samples_A:
                binary_samples.append({"response": thread})
            if project:
                binary_version = prompt_1
            else:
                project = None
                binary_version = None
            melodi_response = create_experiment(experiment_name, experiment_instructions, experiment_type='Binary', items=binary_samples, project=project, binary_version=binary_version, template_type='conversational')

    elif st.session_state['JSON data'] == True:
        if st.session_state['eval_type'] == 'Bake-off':
            comparisons = []
            promptLabel = 'promptLabel'
            if project:
                promptLabel = 'version'
            else:
                project = None

            for thread_A, thread_B in zip(samples_A, samples_B): # Loop through rows with iterrows() 
                comparisons.append({"samples":[
                    {"response": thread_A, promptLabel: prompt_1}, 
                    {"response": thread_B, promptLabel: prompt_2} 
                ]})

            melodi_response = create_experiment(
                experiment_name,
                experiment_instructions,
                items=comparisons,
                experiment_type='Bake-off',
                project=project,
                template_type='json'
            )
        
        else: ### create binary (gpt)
            binary_samples = []
            for thread in samples_A:
                binary_samples.append({"response": thread})
            if project:
                binary_version = prompt_1
            else:
                project = None
                binary_version = None
            melodi_response = create_experiment(experiment_name, experiment_instructions, experiment_type='Binary', items=binary_samples, project=project, binary_version=binary_version, template_type='json')

    else:
        if st.session_state['eval_type'] == 'Bake-off':
            comparisons = []
            promptLabel = 'promptLabel'
            if project:
                promptLabel = 'version'
            else:
                project = None
            if num_columns == 2:
                for index, sample in samples.iterrows(): # Loop through rows with iterrows() 
                    comparisons.append({"samples":[
                        {"response": sample[prompt_1], promptLabel: prompt_1}, 
                        {"response": sample[prompt_2], promptLabel: prompt_2} 
                    ]})
            elif num_columns == 3:
                for index, sample in samples.iterrows():
                    # Create a dictionary for each sample, handling missing values by using None, which converts to null in JSON
                    sample_1_response = sample.get(prompt_1) if pd.notnull(sample.get(prompt_1)) else None
                    sample_2_response = sample.get(prompt_2) if pd.notnull(sample.get(prompt_2)) else None
                    sample_message = sample[message] if pd.notnull(sample[message]) else None

                    # Add the samples to the comparisons list
                    comparisons.append({
                        "samples": [
                            {"response": sample_1_response, promptLabel: prompt_1, "message": sample_message},
                            {"response": sample_2_response, promptLabel: prompt_2, "message": sample_message}
                        ]
                    })
            melodi_response = create_experiment(
                experiment_name,
                experiment_instructions,
                items=comparisons,
                experiment_type='Bake-off',
                project=project
            )
        
        else:
            binary_samples = []
            for index, row in samples.iterrows():
                binary_samples.append({"response": row.iloc[0]})
            if project:
                binary_version = prompt_1
            else:
                project = None
                binary_version = None
            melodi_response = create_experiment(experiment_name, experiment_instructions, experiment_type='Binary', items=binary_samples, project=project, binary_version=binary_version)
    
    #### Display results links
    if melodi_response.status_code == 200:
        res = melodi_response.json()
        feedback_url = res.get('feedbackUrl')
        results_url = res.get('resultsUrl')
        
        col_app, col_results = st.columns(2)
        with col_app:
            st.link_button("Preview and Share Evaluation",f"{feedback_url}", type="primary", use_container_width=True)

        with col_results:
            st.link_button("View Results Dashboard", f"{results_url}", type="primary", use_container_width=True)


    else:
        response_content = json.loads(melodi_response.content)
        try:
            st.write(f"Failed to create experiment: {response_content['error']}")
        except:
            st.write(f"Failed to create experiment: {response_content}")

