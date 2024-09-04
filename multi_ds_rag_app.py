import os
import time
import streamlit as st

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.llms import Ollama
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool

# Load environment variables
load_dotenv()

# Set environment variables for Langchain
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"

# Initialize Streamlit app
st.title("RAG System Based on Multiple Data Sources")

# Initialize LLM and prompt
llm = Ollama(model="gemma2:2b")
prompt = hub.pull("hwchase17/structured-chat-agent")
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Initialize Arxiv API Wrapper and Tool
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=512)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

# Initialize Wikipedia API Wrapper and Tool
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=512)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)


# Initialize Web Base Loader and Langsmith Tool
web_loader = WebBaseLoader("https://docs.smith.langchain.com/")
web_docs = web_loader.load()
web_documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(web_docs)
web_vectordb = FAISS.from_documents(web_documents, embed_model)
web_retriever = web_vectordb.as_retriever()
langsmith_retriever_tool = create_retriever_tool(
    web_retriever, 
    "langsmith_search",
    "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!"
)


# Initialize PDF Loader and Attention Paper Tool
pdf_loader = PyPDFLoader("attention.pdf")
pdf_docs = pdf_loader.load()
pdf_documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(pdf_docs)
# print(pdf_documents)
pdf_vectordb = FAISS.from_documents(pdf_documents, embed_model)
# print(pdf_vectordb)
pdf_retriever = pdf_vectordb.as_retriever()
# print(pdf_retriever)
attention_retriever_tool = create_retriever_tool(
    pdf_retriever, 
    "attention_paper",
    "Search for information about 'Attention is all you need' paper. For any questions about the attention paper, you must use this tool!"
)


# Combine all tools
tools = [arxiv, attention_retriever_tool, langsmith_retriever_tool, wiki]

# Create agent and executor
agent = create_structured_chat_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

prompt=st.text_input("Search the topic you want", placeholder="Ask your query here ...........")

if prompt:
    start=time.process_time()
    response=agent_executor.invoke({"input":prompt})
    print("Response time :",time.process_time()-start)
    st.write(response['output'])

