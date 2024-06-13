# Imports necessary libraries
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

# Retrieve Google API key from environment variable for secure storage
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Retrieve API key from environment variable
genai.configure(api_key=GOOGLE_API_KEY)



def get_pdf_text(pdf_docs):
    """
  This function extracts text content from a list of PDF documents.

  Args:
      pdf_documents (list): A list containing paths to the PDF documents.

  Returns:
      str: The combined text content extracted from all the PDFs.
    """
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    """
  This function splits a large chunk of text into smaller, more manageable pieces.

  Args:
      text (str): The large text content to be split.

  Returns:
      list: A list containing the smaller text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    """
  This function creates a vector store for efficient searching of similar text data.

  Args:
      text_chunks (list): The list of text chunks obtained from the PDFs.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    """
  This function defines a conversational question-answering chain using a generative AI model.

  Returns:
      langchain.llms.question_answering.QuestionAnsweringChain: The question-answering chain object.
    """


    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    """
  This function processes the user's question based on the uploaded PDFs.

  Args:
      user_question (str): The question entered by the user.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])




def main():
    """
  This function is the entry point for the application. It creates the Streamlit user interface and handles user interactions.
    """

    # Set Streamlit page configuration for layout and title
    st.set_page_config(
    page_title="Unfold",
    page_icon="📄",
    layout="wide",
    )
    st.title("Welcome to Unfold QNA Bot 📜🤖")

    # Create a text input box for the user to enter their question
    user_question = st.text_input("Ask a Question")

    # Check if the user entered a question
    if user_question:
        user_input(user_question)

    # Create a sidebar for uploading PDFs
    with st.sidebar:
        st.title("Welcome :")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)

        # Button to trigger processing when clicked
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                # Extract text from uploaded PDFs
                raw_text = get_pdf_text(pdf_docs)
                # Split text into chunks
                text_chunks = get_text_chunks(raw_text)
                # Create the vector store (optional, can be pre-generated)
                get_vector_store(text_chunks)
                # Show success message after processing
                st.success("Done")

# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()
