import streamlit as st
import sys
import os

# Ensure src can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from src.chatbot import ChatBot
except ModuleNotFoundError:
    st.error("Module 'src' not found. Ensure you are running this from the project root.")
    st.stop()

st.set_page_config(page_title="Cognevance AI Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 Cognevance AI Chatbot")
st.markdown("An NLP intent-classification chatbot built for the Cognevance ML Internship.")

# Load bot only once per session using Streamlit caching
@st.cache_resource
def load_bot():
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'intents.json')
    try:
        return ChatBot(models_dir=models_dir, dataset_path=data_path)
    except Exception as e:
        return str(e)

bot = load_bot()

if isinstance(bot, str):
    st.error(f"Error loading Chatbot Engine: {bot}")
    st.info("Did you run the model training script? (`python src/model.py`)")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am the Cognevance AI assistant. How can I help you today?"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Type your message here..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get bot response using our NLP engine
    response = bot.get_response(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
        
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
