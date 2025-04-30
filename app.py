import os
import streamlit as st

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

os.environ["USE_TF"] = "0"

# Constants
HUGGINGFACE_REPO_ID = "google/flan-t5-base"
#HF_TOKEN = os.environ.get("HF_TOKEN")
HF_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))
DB_FAISS_PATH = "vectorstore/db_faiss"

# Custom CSS for styling
st.markdown("""
    <style>
        /* Gradient background */
        body {
            background: linear-gradient(to right, #f3e8ff, #fde2ff);
        }
        .stApp {
            background: linear-gradient(to right, #f3e8ff, #fde2ff);
        }

        .edex-title {
            font-size: 2.5em;
            font-weight: 800;
            color: #5A189A;
            text-align: center;
            margin-bottom: 0.2em;
        }
        .edex-tagline {
            font-size: 1em;
            text-align: center;
            color: #2563eb;
            margin-bottom: 1em;
        }
        .stChatMessage.user div {
            background: #E9D5FF !important;
            color: black;
            border-radius: 20px;
            padding: 12px 16px;
        }
        .stChatMessage.assistant div {
            background: #F0ABFC !important;
            color: black;
            border-radius: 20px;
            padding: 12px 16px;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def load_llm(huggingface_repo_id):
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",
        temperature=0.5,
        huggingfacehub_api_token=HF_TOKEN,
        model_kwargs={"max_length": 512}
    )

def main():
    # Heading
    st.markdown('<div class="edex-title">EDEX-Ai</div>', unsafe_allow_html=True)
    st.markdown('<div class="edex-tagline">Your friendly AI for judgment-free sexual education</div>', unsafe_allow_html=True)

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    prompt = st.chat_input("Ask your question here...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer user's question.
        If you don't know the answer, just say that you don't know. Don't try to make up an answer.
        Don't provide anything out of the given context.

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.
        """

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})
            result = response["result"]

            st.chat_message('assistant').markdown(result)
            st.session_state.messages.append({'role': 'assistant', 'content': result})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
