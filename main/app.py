import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from PyPDF2 import PdfReader

load_dotenv()

# Set your Groq API key
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


def get_pdf_text(pdf):
    text = ""
    for pdf in pdf:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(docs)
    return chunks


def get_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    vectorstore.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatOpenAI(model="gpt-5-nano", api_key=OPENAI_API_KEY)
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=['context', 'question'])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(question):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")

    new_db = FAISS.load_local(
        "faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(question)
    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": question},
        return_only_outputs=True
    )
    print(response)
    st.write("Reply: \n", response['output_text'])


def main():
    st.set_page_config(
        page_title="Chat with PDF (Groq LLaMA3)", layout="centered")
    st.title("📄 Chat with PDF using LLaMA 3 (Groq)")

    user_question = st.text_input("💭 Ask something from the PDF")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.header("📁 Upload PDF")
        pdf_docs = st.file_uploader(
            "Upload PDF files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("⏳ Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(raw_text)
                get_vector_store(chunks)
                st.success("✅ Done! Ask your questions in the main window.")


if __name__ == "__main__":
    main()
