import streamlit as st
import pandas as pd
import google.generativeai as genai
import warnings
import tempfile
from dotenv import load_dotenv
import os

warnings.filterwarnings("ignore")

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
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

def main():
    st.title("chatCSV")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            file_path = temp_file.name
            temp_file.write(uploaded_file.getbuffer())

        df = pd.read_csv(file_path)
        st.write("CSV Data Preview:")
        st.dataframe(df.head())

        query = st.text_input("Ask a question about the data")

        if query:
            response = csv_data(query, file_path)
            st.write("Response for your query:")
            st.write(response)

if __name__ == "__main__":
    main()
