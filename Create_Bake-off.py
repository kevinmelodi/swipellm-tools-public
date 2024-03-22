import json
from openai import OpenAI
import pandas as pd
import requests
import streamlit as st
import time

st.set_page_config(page_title="Melodi | Create Evaluation", page_icon='melodi_transparent.png', initial_sidebar_state='expanded',)

with st.sidebar:
    melodi_api_key = st.text_input("Melodi API Key", type="password")

def create_experiment(experiment_name, experiment_instructions, items, experiment_type='Bake-off', project=None, binary_version=None):
    api_key = melodi_api_key  # Make sure to replace this with your actual API key
    base_url = "https://app.melodi.fyi/api/external/experiments?apiKey="
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

col_radio, col_images = st.columns(2)


with col_radio: 
    st.session_state['eval_type'] = st.radio("**Evaluation Type**", ['Bake-off','Binary'], captions=['A/B test responses from two models', 'Pass/fail assess responses from one model'])

with col_images:

    if st.session_state['eval_type'] == 'Bake-off':
        st.image('images/bakeoff.png')
    else:
        st.image('images/binary.png')



st.subheader('Import data')
tab1, tab2 = st.tabs(["Manual Entry", "CSV Upload"])

if st.session_state['eval_type'] == 'Bake-off':
    prompt_a_label = 'Model or Prompt Version Name (A)'
else:
    prompt_a_label = 'Model or Prompt Version Name'

with tab1:

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


with tab2:
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

if st.button('Create Experiment'):
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
                comparisons.append({"samples": [
                    {"response": sample[prompt_1], promptLabel: prompt_1, "message": sample[message]},
                    {"response": sample[prompt_2], promptLabel: prompt_2, "message": sample[message]}
                ]})
        melodi_response = create_experiment(experiment_name, experiment_instructions, items=comparisons, experiment_type='Bake-off', project=project)
    
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

