import streamlit as st
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import numpy as np
import hmac
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

# Authentication (reuse from uploader)
def check_password():
    def login_form():
        with st.form("Credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)
    def password_entered():
        if (
            st.session_state.get("username") in st.secrets.get("passwords", {})
            and hmac.compare_digest(
                st.session_state.get("password", ""),
                st.secrets.passwords.get(st.session_state.get("username", ""), ""),
            )
        ):
            st.session_state["password_correct"] = True
            del st.session_state["username"]
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False
    if st.session_state.get("password_correct", False):
        return True
    login_form()
    if "password_correct" in st.session_state:
        st.error("Username or password incorrect")
    return False

if not check_password():
    st.stop()

# Load secrets & init Pinecone and embeddings
OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
PINECONE_API_KEY = st.secrets['PINECONE_API_KEY']
index_name = st.secrets['INDEX_NAME']
host = st.secrets['HOST']
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=host)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model="gpt-4")
chain = load_qa_chain(llm, chain_type="stuff")

# Main UI
st.set_page_config(page_title="QA & Compare PDFs", page_icon=":page_facing_up:")
st.title("📑 QA & Compare PDFs")

# Ensure files and vector_ids exist
if 'files' not in st.session_state or 'vector_ids' not in st.session_state:
    st.error("Please run the uploader app first to upload and index documents.")
    st.stop()

# Load vectorstore once
def get_vectorstore():
    if 'vs' not in st.session_state:
        st.session_state.vs = PineconeVectorStore.from_existing_index(
            embedding=embeddings, index_name=index_name, namespace="default"
        )
    return st.session_state.vs

# Tabs
tab1, tab2 = st.tabs(["Q&A", "Compare Sections"])

with tab1:
    st.header("Ask questions over your PDFs")
    vs = get_vectorstore()
    query = st.text_input("Enter your question...")
    if query:
        docs = vs.similarity_search(query)
        answer = chain.run(input_documents=docs, question=query)
        st.markdown(f"**Answer:** {answer}")

with tab2:
    st.header("Compare Specific Sections in Two Documents")
    names = list(st.session_state['vector_ids'].keys())
    if len(names) < 2:
        st.info("Index at least two documents in the uploader first.")
    else:
        a = st.selectbox("First document", names, key="comp_a")
        b = st.selectbox("Second document", names, key="comp_b")
        section_a = st.text_input(f"Section in '{a}' to compare (e.g. 'Chapter 1')", key="sec_a")
        section_b = st.text_input(f"Section in '{b}' to compare (e.g. 'Chapter 1')", key="sec_b")
        if st.button("Compare Sections") and a != b and section_a and section_b:
            vs = get_vectorstore()
            # retrieve top chunks matching each section
            docs_a = vs.similarity_search(section_a, filter={"filename": a}, k=5)
            docs_b = vs.similarity_search(section_b, filter={"filename": b}, k=5)
            # embed the retrieved chunk texts
            texts_a = [d.page_content for d in docs_a]
            texts_b = [d.page_content for d in docs_b]
            embs_a = embeddings.embed_documents(texts_a)
            embs_b = embeddings.embed_documents(texts_b)
            avg_a = np.mean(np.array(embs_a), axis=0)
            avg_b = np.mean(np.array(embs_b), axis=0)
            # cosine similarity
            sim = np.dot(avg_a, avg_b) / (np.linalg.norm(avg_a) * np.linalg.norm(avg_b))
            st.metric("Section Cosine Similarity", f"{sim:.3f}")
            # summary of section comparison
            prompt = (
                f"Compare the section '{section_a}' in document '{a}' "
                f"with the section '{section_b}' in document '{b}'. "
                "What are their main similarities and differences?"
            )
            merged_docs = [type('D',(),{'page_content': ' '.join(texts_a)})(),
                           type('D',(),{'page_content': ' '.join(texts_b)})()]
            summary = chain.run(input_documents=merged_docs, question=prompt)
            st.subheader("Section Comparison Summary")
            st.write(summary)
