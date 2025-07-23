# Misc imports
import os
import json
import pandas as pd
import torch

import streamlit as st
from pypdf import PdfReader

# Import Minions + Minions
from minions.minions import Minions
from minions.minion import Minion

# Import Minion Clients
from minions.clients.ollama import OllamaClient
from minions.clients.openai import OpenAIClient


# Import Pydantic
from pydantic import BaseModel

class StructuredLocalOutput(BaseModel):
    explanation: str
    citation: str | None
    answer: str | None

torch.classes.__path__ = []

st.title("Echo Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


remote_client = OpenAIClient(model_name="llama3.2:3b", temperature=0.0, base_url="http://localhost:11434/v1")

# Option 1: Ollama
local_client = OllamaClient(
    model_name="qwen3:0.6b", 
    temperature=0.0, 
    structured_output_schema=StructuredLocalOutput,
    base_url="http://192.168.64.14:31434"
    
)

protocol = Minions(local_client=local_client, remote_client=remote_client)


def pdf_reader(file):
    #Upload PDF
    global text_data
    pdf_reader = PdfReader(file) # read your PDF file
    # extract the text data from your PDF file after looping through its pages with the .extract_text() method
    text_data= ""
    for page in pdf_reader.pages: # for loop method
        text_data+= page.extract_text()    
        # st.write(text_data) # view the text data
    return text_data
    # st.write(pdf_reader(pdf))

pdf = st.file_uploader("Upload your PDF", type="pdf")
if pdf is not None:
    text_data = pdf_reader(pdf)
    context = text_data

# doc_metadata = "Patient Visit Notes"
doc_metadata = ""

if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    task=prompt
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    output = protocol(
        task=task,
        doc_metadata=doc_metadata,
        context=[context],
        max_rounds=5,  # you can adjust rounds as needed for testing
    )

    response = output['final_answer']
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

