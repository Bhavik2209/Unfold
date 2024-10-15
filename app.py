import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Retrieve Google API key from environment variable for secure storage
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

genai.configure(api_key=GOOGLE_API_KEY)

# Constants
MAX_FILE_SIZE = 2 * 1024 * 1024  # 2 MB
CHUNK_SIZE = 10000
CHUNK_OVERLAP = 1000

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.error(f"Error reading PDF {pdf.name}: {str(e)}")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, say "I'm sorry, but I don't have enough information to answer that question based on the provided context."

    Context: {context}

    Question: {question}

    Answer:
    """

    try:
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        return load_qa_chain(model, chain_type="stuff", prompt=prompt)
    except Exception as e:
        st.error(f"Error creating conversational chain: {str(e)}")
        return None

def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        
        if chain:
            response = chain(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True
            )
            st.write("Reply:", response["output_text"])
        else:
            st.error("Failed to create conversational chain. Please try again.")
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")

def main():
    st.set_page_config(page_title="Unfold", page_icon="📄", layout="wide")
    st.title("Welcome to Unfold QNA Bot 📜🤖")

    user_question = st.text_input("Ask a Question")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Welcome:")
        pdf_docs = st.file_uploader("Upload your PDF Files (Max 2MB each)", 
                                    accept_multiple_files=True, 
                                    help="Upload PDF files to process")

        if pdf_docs:
            if st.button("Submit & Process"):
                with st.spinner("Processing..."):
                    # Check file sizes
                    valid_files = [pdf for pdf in pdf_docs if pdf.size <= MAX_FILE_SIZE]
                    if len(valid_files) != len(pdf_docs):
                        st.warning(f"Some files exceeded the 2MB limit and were skipped. Processing {len(valid_files)} valid files.")
                    
                    if valid_files:
                        raw_text = get_pdf_text(valid_files)
                        if raw_text:
                            text_chunks = get_text_chunks(raw_text)
                            get_vector_store(text_chunks)
                            st.success("Processing complete!")
                        else:
                            st.warning("No text could be extracted from the PDFs. Please check your files and try again.")
                    else:
                        st.error("No valid files to process. Please upload PDF files under 2MB each.")

if __name__ == "__main__":
    main()
