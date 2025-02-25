import streamlit as st
import pandas as pd
import google.generativeai as genai
import warnings
import tempfile
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
warnings.filterwarnings("ignore")

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

if GEMINI_API_KEY is None:
    raise ValueError("Gemini API Key is missing. Please add it to the .env file.")

def load_and_summarize_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        return df.to_string(index=False)
    except Exception as e:
        return f"Error reading CSV: {e}"


def csv_data(query, file_path):
    try:
        csv_summary = load_and_summarize_csv(file_path)
        prompt = f"Here is some data:\n{csv_summary}\n\nNow, answer the following question: {query}"
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text if response else "No response received from the AI model."
    except Exception as e:
        return f"Error processing file: {e}"


# PDF Processing Functions
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def query_gemini(prompt, context):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(f"Context: {context}\nQuestion: {prompt}")
    return response.text if response else "No response."


def handle_pdf_query(user_question):
    retriever = st.session_state.vectorstore.as_retriever()
    relevant_docs = retriever.get_relevant_documents(user_question)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    return query_gemini(user_question, context)


def main():
    st.set_page_config(page_title=" The Querier")

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "csv_file" not in st.session_state:
        st.session_state.csv_file = None

    st.title("The Querier")

    option = st.radio("Choose file type to upload:", ("CSV", "PDF"))

    if option == "CSV":
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
                file_path = temp_file.name
                temp_file.write(uploaded_file.getbuffer())
                st.session_state.csv_file = file_path

            df = pd.read_csv(file_path)
            st.write("CSV Data Preview:")
            st.dataframe(df.head())

            query = st.text_input("Ask a question about the CSV data")
            if query:
                response = csv_data(query, file_path)
                st.write("Response for your query:")
                st.write(response)

    elif option == "PDF":
        pdf_docs = st.file_uploader("Upload PDFs and click 'Process'", accept_multiple_files=True)
        if st.button("Process") and pdf_docs:
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                st.session_state.vectorstore = get_vectorstore(text_chunks)
                st.success("PDFs processed successfully!")

        user_question = st.text_input("Ask a question about your PDFs")
        if user_question and st.session_state.vectorstore:
            response = handle_pdf_query(user_question)
            st.write("Response:")
            st.write(response)

if __name__ == "__main__":
    main()
