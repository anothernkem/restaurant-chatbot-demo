import streamlit as st
import os
import sys
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

# --- Page Configuration ---
st.set_page_config(
    page_title="B-West Grill",
    page_icon="🍔",
    layout="centered"
)

# --- 1. GET API KEY FROM STREAMLIT SECRETS ---
# This is the new, secure way.
# It reads the secret you just saved in the Streamlit Cloud settings.
if "GOOGLE_API_KEY" in st.secrets:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("API Key not found. Please add it to your Streamlit Secrets.")
    st.stop()

# --- 2. CONFIGURE LLM & EMBEDDING MODEL ---
try:
    Settings.llm = GoogleGenAI(
        model_name="models/gemini-1.5-flash", 
        api_key=google_api_key
    )
    Settings.embed_model = GoogleGenAIEmbedding(
        model_name="models/embedding-001", 
        api_key=google_api_key
    )
except Exception as e:
    st.error(f"Error setting up Google AI. Check your API key. Details: {e}")
    st.stop()

# --- 3. SYSTEM PROMPT (The Bot's "Rules") ---
# (This is unchanged)
SYSTEM_PROMPT = """
You are "RestaurantBot," an expert, friendly, and helpful waiter for our restaurant.
Your primary goal is to help customers find the perfect dish.

You must follow these rules at all times:
1.  **Use ONLY the provided data.** Never make up a dish, price, or ingredient.
2.  **Be Friendly and Conversational:** Do not just list data. Talk like a helpful waiter.
3.  **Answer Allergen Questions:** Use the `tags` data to answer questions about allergens (`contains_gluten`, `contains_nuts`, etc.) and dietary needs (`vegan`, `is_halal`).
4.  **Answer "What's Good?":** Use the customer reviews to answer questions like "What's good?" or "What's popular?"
5.  **ADVANCED PAIRING LOGIC:** When asked for a pairing or recommendation, you must use the following rules based on the `tags`:
    * **Rule 1 (Spice):** Spicy dishes (`spicy`) pair well with cooling, sweet drinks (`cooling`, `sweet`) or light, crisp beers (`light`, `crisp`).
    * **Rule 2 (Intensity):** Rich, heavy dishes (`rich`, `heavy`, `red_meat`) pair with full-bodied drinks (like a `red_wine` with `tannic` tags).
    * **Rule 3 (Intensity):** Light, delicate dishes (`light`, `delicate`, `fish`, `poultry`) pair with light-bodied, acidic drinks (like a `white_wine` with `crisp`, `acidic` tags).
    * **Rule 4 (Contrast):** Acidic drinks (`acidic`) can cut through rich, creamy dishes (`rich`, `creamy`).
6.  **Explain Your Recommendations:** When you recommend a pairing, *always explain why* using the rules.
"""

# --- 4. LOAD THE KNOWLEDGE BASE (THE "INDEX") ---
@st.cache_resource(show_spinner="Loading knowledge base... 🧠")
def load_index():
    PERSIST_DIR = "./storage"
    if not os.path.exists(PERSIST_DIR):
        # If storage isn't on GitHub, we need to rebuild it
        # This assumes the CSV and reviews folder ARE on GitHub
        print("Storage not found. Rebuilding index...")
        
        # Load the CSV file directly
        menu_docs = SimpleDirectoryReader(
            input_files=["./menu_data.csv"]
        ).load_data()
        
        # Load the reviews folder directly
        review_docs = SimpleDirectoryReader(
            input_dir="./reviews"
        ).load_data()
        
        documents = menu_docs + review_docs
        print(f"Loaded {len(documents)} documents. Indexing...")
        
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        print("Index created and saved.")
    else:
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
    
    return index

index = load_index()
if index is None:
    st.stop()

# --- 5. INITIALIZE THE CHAT ENGINE ---
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="context",
        system_prompt=SYSTEM_PROMPT
    )

# --- 6. MANAGE & DISPLAY CHAT HISTORY ---
st.title("🍽️ RestaurantBot")
st.caption("Ask me about the menu, allergens, or what's good!")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm RestaurantBot. Ask me anything about the menu."}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 7. THE CHAT INPUT BOX ---
if user_query := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(user_query)
            st.markdown(response.response)
    
    st.session_state.messages.append({"role": "assistant", "content": response.response})