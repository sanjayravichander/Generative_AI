import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Streamlit UI
st.title("Simple Chatbot using Gemini AI")
st.sidebar.title("Chatbot Configuration")

api_key = st.sidebar.text_input("Enter your Gemini AI API Key", type="password")

# Mapping user-friendly names to API model names
model_mapping = {
    "2.0 Flash": "gemini-2.0-flash",
    "2.0 Flash-Lite": "gemini-2.0-flash-lite",
    "2.0 Pro": "gemini-2.0-pro"
}

# Dropdown for selecting model
selected_model = st.sidebar.selectbox("Select Model", list(model_mapping.keys()))
model_name = model_mapping[selected_model]  # Convert to API-compatible name

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5)
max_tokens = st.sidebar.slider("Max Tokens", 50, 300, 150)

# Ensure API key is configured before making requests
if api_key:
    genai.configure(api_key=api_key)

def get_gemini_response(prompt, model_name, temperature, max_tokens):
    """Fetch response from Gemini AI."""
    try:
        model = genai.GenerativeModel(model_name)  # Use mapped model names
        response = model.generate_content(prompt, generation_config={
            "temperature": temperature,
            "max_output_tokens": max_tokens
        })
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

user_input = st.text_input("You: ", "")
if user_input and api_key:
    response = get_gemini_response(user_input, model_name, temperature, max_tokens)
    st.write(response)
elif not api_key:
    st.warning("Please enter a valid Gemini AI API key.")
else:
    st.write("Please enter a message to start the conversation.")
