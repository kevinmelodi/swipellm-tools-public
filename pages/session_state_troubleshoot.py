import json
from openai import OpenAI
import pandas as pd
import requests
import streamlit as st
import time

st.session_state

if 'key' not in st.session_state:
    st.session_state['key'] = 'value'

st.session_state