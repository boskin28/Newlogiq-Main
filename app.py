import streamlit as st
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import numpy as np
import hmac
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import difflib

# --- Authentication ---
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

# --- Load secrets & init clients ---
OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
PINECONE_API_KEY = st.secrets['PINECONE_API_KEY']
index_name = st.secrets['INDEX_NAME']
host = st.secrets['HOST']
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=host)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model="gpt-4")
chain = load_qa_chain(llm, chain_type="stuff")

# --- Helpers ---
def get_vectorstore(namespace: str):
    """Load or cache PineconeVectorStore for a given namespace"""
    key = f"vs_{namespace}"
    if key not in st.session_state:
        st.session_state[key] = PineconeVectorStore.from_existing_index(
            embedding=embeddings, index_name=index_name, namespace=namespace
        )
    return st.session_state[key]


def list_namespaces():
    stats = index.describe_index_stats()
    return list(stats.get('namespaces', {}).keys())


def list_filenames(namespace: str):
    stats = index.describe_index_stats(namespace=namespace)
    meta = stats['namespaces'].get(namespace, {}).get('metadata', {})
    return list(meta.get('filename', {}).keys())

# --- UI ---
st.set_page_config(page_title="QA & Compare PDFs", page_icon=":page_facing_up:")
st.title("📑 QA & Compare PDFs")

# Namespace selection
namespaces = list_namespaces()
namespace = st.selectbox("Select namespace", namespaces, index=0)

# Tabs
tab1, tab2 = st.tabs(["Q&A", "Compare Sections"]);

with tab1:
    st.header("Ask questions over your PDFs")
    vs = get_vectorstore(namespace)
    query = st.text_input("Enter your question...")
    if query:
        docs = vs.similarity_search(query)
        answer = chain.run(input_documents=docs, question=query)
        st.markdown(f"**Answer:** {answer}")

with tab2:
    st.header("Compare Specific Sections in Two Documents")
    filenames = list_filenames(namespace)
    if len(filenames) < 2:
        st.info("No or only one document found in this namespace. Use the uploader app to index more.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            doc_a = st.selectbox("First document", filenames, key="comp_a")
            section_a = st.text_input(f"Section in '{doc_a}' (e.g. 'Chapter 1')", key="sec_a")
        with col2:
            doc_b = st.selectbox("Second document", filenames, key="comp_b")
            section_b = st.text_input(f"Section in '{doc_b}' (e.g. 'Chapter 1')", key="sec_b")
        if st.button("Compare Sections"):
            if doc_a == doc_b:
                st.warning("Select two different documents.")
            elif not section_a or not section_b:
                st.warning("Enter both section identifiers.")
            else:
                vs = get_vectorstore(namespace)
                docs_a = vs.similarity_search(section_a, filter={"filename": doc_a}, k=5)
                docs_b = vs.similarity_search(section_b, filter={"filename": doc_b}, k=5)
                text_a = "\n".join(d.page_content for d in docs_a)
                text_b = "\n".join(d.page_content for d in docs_b)
                embs_a = embeddings.embed_documents([text_a])
                embs_b = embeddings.embed_documents([text_b])
                sim = np.dot(embs_a[0], embs_b[0]) / (np.linalg.norm(embs_a[0]) * np.linalg.norm(embs_b[0]))
                st.metric("Section Cosine Similarity", f"{sim:.3f}")
                prompt = (
                    f"Compare the section '{section_a}' in '{doc_a}' with '{section_b}' in '{doc_b}'. "
                    "Describe their key similarities and differences."
                )
                merged = [type('D',(),{'page_content': text_a})(), type('D',(),{'page_content': text_b})()]
                summary = chain.run(input_documents=merged, question=prompt)
                st.subheader("Summary")
                st.write(summary)
                diff = difflib.unified_diff(
                    text_a.splitlines(), text_b.splitlines(),
                    fromfile=f"{doc_a}:{section_a}", tofile=f"{doc_b}:{section_b}", lineterm=""
                )
                st.subheader("Detailed Diff")
                st.code("\n".join(diff), language='diff')
