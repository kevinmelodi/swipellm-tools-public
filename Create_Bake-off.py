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

if 'Braintrust data' not in st.session_state:
    st.session_state['Braintrust data'] = False

col_radio, col_images = st.columns(2)


with col_radio: 
    st.session_state['eval_type'] = st.radio("**Evaluation Type**", ['Bake-off','Binary'], captions=['A/B test responses from two models', 'Pass/fail assess responses from one model'])

with col_images:

    if st.session_state['eval_type'] == 'Bake-off':
        st.image('images/bakeoff.png')
    else:
        st.image('images/binary.png')



st.header('Import data')
tab_csv, tab_GPT, tab_json, tab_braintrust, tab_manual = st.tabs([ "CSV","GPT Thread (Fine Tune)","JSONL", "Braintrust CSV", "Manual Entry"])

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


def create_comparisons_corrected(df):
    def transform_to_json_dicts_assuming_structure(df):
        json_dicts_standard = []
        json_dicts_handwritten = []
        for index, row in df.iterrows():
            user_message = {'role': 'user', 'content': row[df.columns[0]]}
            standard_response = {'role': 'assistant', 'content': row[df.columns[1]]}
            handwritten_response = {'role': 'assistant', 'content': row[df.columns[2]]}
            json_dicts_standard.append([user_message, standard_response])
            json_dicts_handwritten.append([user_message, handwritten_response])
        return json_dicts_standard, json_dicts_handwritten

    json_dicts_standard, json_dicts_handwritten = transform_to_json_dicts_assuming_structure(df)
    comparisons = []
    for standard, handwritten in zip(json_dicts_standard, json_dicts_handwritten):
        comparisons.append({'samples': [standard, handwritten]})
    return comparisons


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
                st.dataframe(samples, hide_index=True)
            elif num_columns == 3:
                samples = file.iloc[:, 0:3]
                message, prompt_1, prompt_2 = samples.columns.tolist()
                try:
                    st.subheader("Conversation Data Preview",divider='rainbow')
                    if st.session_state['eval_type'] == 'Bake-off':
                        comparisons_csv_3col = create_comparisons_corrected(file)
                        col_message_A_preview, col_message_B_preview = st.columns(2)
                        with col_message_A_preview:
                            for message in comparisons_csv_3col[0]['samples'][0]:
                                with st.chat_message(message['role']):
                                    st.markdown(message['content'])
                        with col_message_B_preview:
                            for message in comparisons_csv_3col[0]['samples'][1]:
                                with st.chat_message(message['role']):
                                    st.markdown(message['content'])
                    else:
                        st.write('promple')

                except:
                    pass

            st.session_state['CSV data'] = True
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
                prompt_1_GPT = st.text_input(
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
                prompt_2_GPT = st.text_input(
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
                prompt_1_JSON = st.text_input(
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
                prompt_2_JSON = st.text_input(
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


def is_valid_json(json_string):
    try:
        return True, json.loads(json_string)  # Return True and the parsed JSON if it's valid
    except json.JSONDecodeError:
        # Attempt to clean the string and check again
        if json_string.startswith('"') and json_string.endswith('"'):
            json_string = json_string[1:-1]
        json_string = json_string.replace(r'\"', '"').replace('\\n', ' ').replace('\\r', ' ')
        try:
            return True, json.loads(json_string)  # Return True and the parsed JSON if it's now valid
        except json.JSONDecodeError:
            return False, json_string  # Return False and the original string if it's still not valid

def upload_and_process_file(label, columns_to_display):
    samples = None
    has_valid_json = False  # Initialize flag to track JSON validity
    uploaded_file = st.file_uploader(label=label, type=['csv'], label_visibility='hidden')
    if uploaded_file is not None:
        st.session_state['file'] = True
        file = pd.read_csv(uploaded_file, header=0)
        if all(col in file.columns for col in columns_to_display):
            samples = file[columns_to_display]
            # Process each cell, check if it's valid JSON and clean if necessary
            for col in columns_to_display:
                samples[col] = samples[col].apply(lambda x: is_valid_json(x)[1] if isinstance(x, str) else x)
                # Check if any cell in the column contains valid JSON
                has_valid_json = has_valid_json or samples[col].apply(lambda x: is_valid_json(x)[0] if isinstance(x, str) else False).any()
            # Update the 'JSON data' session state based on the presence of valid JSON
            st.session_state['JSON data'] = has_valid_json
            st.session_state['Braintrust data'] = True
        else:
            st.error(f"Missing expected columns: {columns_to_display}")
    return samples

with tab_braintrust:
    st.write('This option expects a CSV export of a [Braintrust Data Evaluation](https://www.braintrustdata.com/docs/guides/evals). Only the LLM responses are will be stored in Melodi for review - **inputs are never stored**. No data is stored or retained by Streamlit, and only LLM responses will be sent to Melodi. For more information on Streamlit data handling, [see here](https://docs.streamlit.io/knowledge-base/using-streamlit/where-file-uploader-store-when-deleted).' )
    expand_bt_image = st.expander('Braintrust CSV Export Detail')
    expand_bt_image.image('images/braintrust CSV.png')
    st.write('Melodi validates and attempts to repair JSON in the LLM responses. If the responses can not be validated or fixed, Melodi will default to rendering the samples as markdown text.')
    if st.session_state['CSV data'] == False:
        if st.session_state['eval_type'] == 'Bake-off':
            samples = upload_and_process_file('Braintrust', ['output', 'expected'])
            # Display custom JSON preview component if JSON data is present
            if samples is not None and st.session_state.get('JSON data'):
                st.subheader("JSON Data Preview", divider='rainbow')
                col_json_A_preview, col_json_B_preview = st.columns(2)
                with col_json_A_preview:
                    expander_A = st.expander("Preview data")
                    expander_A.markdown(json_to_markdown_recursive(json.loads(samples['output'].iloc[0])))
                with col_json_B_preview:
                    expander_B = st.expander("Preview data")
                    expander_B.markdown(json_to_markdown_recursive(json.loads(samples['expected'].iloc[0])))
            elif samples is not None and st.session_state.get('JSON data')==False:
                st.dataframe(samples, hide_index=True)
        else:
            samples = upload_and_process_file('braintrust binary', ['output'])
            # Display custom JSON preview component for binary evaluation type
            if samples is not None and st.session_state.get('JSON data'):
                expander = st.expander("Preview data")
                expander.markdown(json_to_markdown_recursive(json.loads(samples['output'].iloc[0])))
            elif samples is not None and st.session_state.get('JSON data')==False:
                st.dataframe(samples, hide_index=True)
    else:
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
                    {"response": thread_A, promptLabel: prompt_1_GPT}, 
                    {"response": thread_B, promptLabel: prompt_2_GPT} 
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
                binary_version = prompt_1_GPT
            else:
                project = None
                binary_version = None
            melodi_response = create_experiment(experiment_name, experiment_instructions, experiment_type='Binary', items=binary_samples, project=project, binary_version=binary_version, template_type='conversational')

    elif st.session_state['JSON data'] == True:
        if st.session_state['Braintrust data']:
            samples_A = [json.loads(line) for line in samples.get('output')]
            if 'expected' in samples.columns:
                samples_B = [json.loads(line) for line in samples.get('expected')]
        if st.session_state['eval_type'] == 'Bake-off':
            comparisons = []
            promptLabel = 'promptLabel'
            if project:
                promptLabel = 'version'
            else:
                project = None


            for thread_A, thread_B in zip(samples_A, samples_B): # Loop through rows with iterrows() 
                comparisons.append({"samples":[
                    {"response": thread_A, promptLabel: prompt_1_JSON}, 
                    {"response": thread_B, promptLabel: prompt_2_JSON} 
                ]})

            melodi_response = create_experiment(
                experiment_name,
                experiment_instructions,
                items=comparisons,
                experiment_type='Bake-off',
                project=project,
                template_type='json'
            )
        
        else:
            binary_samples = []
            for thread in samples_A:
                binary_samples.append({"response": thread})
            if project:
                binary_version = prompt_1_JSON
            else:
                project = None
                binary_version = None
            melodi_response = create_experiment(experiment_name, experiment_instructions, experiment_type='Binary', items=binary_samples, project=project, binary_version=binary_version, template_type='json')

   
    else: ## CSV data
        if st.session_state['eval_type'] == 'Bake-off':
            template_type='default'
            comparisons = []
            promptLabel = 'promptLabel'
            if project:
                promptLabel = 'version'
            else:
                project = None
            if st.session_state['Braintrust data']:
                num_columns = 2
                prompt_1 = 'output'
                prompt_2 = 'expected'
            if num_columns == 2:
                for index, sample in samples.iterrows(): # Loop through rows with iterrows() 
                    comparisons.append({"samples":[
                        {"response": sample[prompt_1], promptLabel: prompt_1}, 
                        {"response": sample[prompt_2], promptLabel: prompt_2} 
                    ]})
                melodi_response = create_experiment(
                experiment_name,
                experiment_instructions,
                items=comparisons,
                experiment_type='Bake-off',
                project=project,
                template_type=template_type
            )
            elif num_columns == 3:
                template_type = 'conversational'
                comparisons = []
                promptLabel = 'promptLabel'
                if project:
                    promptLabel = 'version'
                else:
                    project = None
                for item in comparisons_csv_3col: # Loop through rows with iterrows() 
                    comparisons.append({"samples":[
                        {"response": {"messages": item['samples'][0]}, promptLabel: prompt_1_GPT}, 
                        {"response": {"messages": item['samples'][1]}, promptLabel: prompt_2_GPT} 
                    ]})
                
                melodi_response = create_experiment(
                    experiment_name,
                    experiment_instructions,
                    items=comparisons,
                    experiment_type='Bake-off',
                    project=project,
                    template_type='conversational'
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
