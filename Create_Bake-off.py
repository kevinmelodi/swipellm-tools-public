import json
from openai import OpenAI
import pandas as pd
import requests
import streamlit as st
import time

st.set_page_config(page_title="Melodi | Create Evaluation", page_icon='melodi_transparent.png', initial_sidebar_state='expanded',)

with st.sidebar:
    melodi_api_key = st.text_input("Melodi API Key", type="password")

def create_experiment(experiment_name, experiment_instructions, comparisons):
    api_key = melodi_api_key
    base_url = "https://app.melodi.fyi/api/external/experiments?apiKey="
    url = base_url + api_key
    headers = {"Content-Type": "application/json"}

    # Data payload for the POST request
    data = {
        "experiment": {
            "name": experiment_name,
            "instructions": experiment_instructions
        },
        "comparisons": comparisons
    }

    # Make the POST request
    response = requests.post(url, headers=headers, data=json.dumps(data))

    return response

def create_experiment_binary(experiment_name, experiment_instructions, samples):
    api_key = melodi_api_key
    base_url = "https://app.melodi.fyi/api/external/experiments?apiKey="
    url = base_url + api_key
    headers = {"Content-Type": "application/json"}

    # Data payload for the POST request
    data = {
        "experiment": {
            "name": experiment_name,
            "instructions": experiment_instructions
        },
        "samples": samples
    }

    # Make the POST request
    response = requests.post(url, headers=headers, data=json.dumps(data))

    return response

st.image('melodi_transparent.png', width=70)
st.title("Create Melodi Evaluation")
st.caption("Enter your Melodi API key in the sidebar to begin. Ideal for copy/pasting directly from ChatGPT or for CSV upload of LLM responses. To create experiments via API, [see instructions here](https://melodi.notion.site/Melodi-Experiments-API-08b6d362277d49e9aa167c75bce153a0)")


experiment_name = st.text_input(
        "**Experiment name** *(reviewers won't see this)*",
    )

experiment_instructions = st.text_input(
        "**Instructions for reviewers**",
    )

if 'disabled' not in st.session_state:
    st.session_state['disabled'] = False

col_radio, col_images = st.columns(2)


with col_radio: 
    st.session_state['eval_type'] = st.radio("**Evaluation Type**", ['Bake-off','Binary'], captions=['A/B test responses from two models', 'Pass/fail assessment of responses from one model'])

with col_images:

    if st.session_state['eval_type'] == 'Bake-off':
        st.image('/workspaces/swipellm-tools-public/images/bakeoff.png')
    else:
        st.image('/workspaces/swipellm-tools-public/images/binary.png')

tab1, tab2 = st.tabs(["Manual Entry", "CSV Upload"])

with tab1:

    col1, col2 = st.columns(2)

    if 'help' not in st.session_state:
        st.session_state.help = "Reviewers won't see this"

    if st.session_state['eval_type'] == 'Bake-off':
        with col1:
            prompt_1 = st.text_input(
                "Model or Prompt Name (A)",
                key="placeholder",
                value="Original Prompt",
                disabled=st.session_state.disabled,
                help="Names can't be updated if data is present in the table."
            )

        with col2:
            prompt_2 = st.text_input(
                "Model or Prompt Name (B)",
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
            "Responses" : st.column_config.TextColumn(f"Generated content examples", width='large', required=True),
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
            samples = file.iloc[:,0:2]
            prompt_1, prompt_2 = samples.columns.tolist()
            st.dataframe(samples, hide_index=True)
    else:
        uploaded_file = st.file_uploader(label="For **binary evaluations**, a header row is required, but the header value is not used to create the evaluation. The file should be one column: a list of example LLM responses from one prompt/model.", type=['csv'])
        if uploaded_file is not None:
            st.session_state['file'] = True
            file = pd.read_csv(uploaded_file,header=0)
            samples = file.iloc[:,0:1]
            st.dataframe(samples, hide_index=True)

if st.button('Create Experiment'):
    if st.session_state['eval_type'] == 'Bake-off':
        comparisons = []
        for index, sample in samples.iterrows(): # Loop through rows with iterrows()
            
            comparisons.append({"samples":[
                {"response": sample[prompt_1],"promptLabel": prompt_1}, 
                {"response": sample[prompt_2], "promptLabel": prompt_2} 
            ]})
        
        
        melodi_response = create_experiment(experiment_name, experiment_instructions, comparisons)
    else:
        binary_samples = []
        for index, row in samples.iterrows():
            binary_samples.append({"response": row.iloc[0]})
        melodi_response = create_experiment_binary(experiment_name, experiment_instructions, binary_samples)
        

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

