import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def summarize_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load_and_split()
    llm = ChatGroq(model='gemma2-9b-it', temperature=0, api_key=GROQ_API_KEY)
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.invoke(docs)

    return summary


if __name__ == "__main__":
    summary = summarize_pdf("annual_xchanging.pdf")

    print("Summary: \n")
    print(summary)
