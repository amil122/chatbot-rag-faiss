import streamlit as st
import requests

# FastAPI backend URL
API_URL = "http://localhost:8000/chat"

st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("RAG Chatbot")

# Input text box
query = st.text_input("Ask a question about your documents:")

# Button to send request
if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            try:
                # Send POST request to FastAPI backend
                response = requests.post(API_URL, json={"query": query})
                
                if response.status_code == 200:
                    data = response.json()
                    st.markdown("### Answer:")
                    st.write(data["answer"])

                    # Display sources
                    if data["sources"]:
                        st.markdown("### Sources:")
                        for src in data["sources"]:
                            st.markdown(f"- `{src}`")
                    else:
                        st.info("No sources returned.")
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to API: {e}")
