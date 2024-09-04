import io
import json
import os
import urllib
import streamlit as st
from pathlib import Path
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from pymongo import MongoClient
from bson.json_util import dumps


st.title("LangChain with Databases")

LOCALDB = "USE_LOCALDB"
SQLDB = "USE_SQLDB"
MONGODB = "USE_MONGODB"

radio_opt = [
    "Use SQLite 3 Database: Student.db",
    "Connect to your SQL Database",
    "Connect to your NoSQL (MongoDB) database"
]

selected_opt = st.sidebar.radio(label="Choose the DB which you want to chat", options=radio_opt)

if radio_opt.index(selected_opt) == 1:
    db_uri = SQLDB
    sql_db_type = st.sidebar.selectbox("Choose SQL Database Type", ["MySQL", "PostgreSQL"])
    sql_host = st.sidebar.text_input(f"Provide {sql_db_type} Host")
    sql_user = st.sidebar.text_input(f"{sql_db_type} User")
    sql_password = st.sidebar.text_input(f"{sql_db_type} Password", type="password")
    sql_db = st.sidebar.text_input(f"{sql_db_type} Database")
elif radio_opt.index(selected_opt) == 2:
    db_uri = MONGODB
    mongo_cluster = st.sidebar.text_input("MongoDB Cluster")
    mongo_user = st.sidebar.text_input("MongoDB User", "")
    mongo_password = st.sidebar.text_input("MongoDB Password", type="password")
    mongo_db = st.sidebar.text_input("MongoDB Database")
    mongo_col = st.sidebar.text_input("MongoDB Collection")
else:
    db_uri = LOCALDB

groq_api_key = st.sidebar.text_input(label="Groq API Key", type="password")

if not db_uri:
    st.info("Please enter the database information and URI")

if not groq_api_key:
    st.info("Please add the Groq API key")

## LLM model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant", streaming=True)

## Mongo db prompt
with io.open("/home/sr/Documents/genAIStuff/6-Chat SQL/sample.txt","r",encoding="utf-8") as f1:
    sample=f1.read()
    f1.close()
with io.open("/home/sr/Documents/genAIStuff/6-Chat SQL/prompt.txt","r",encoding="utf-8") as f1:
    prompt=f1.read()
    f1.close()

mongo_prompt = PromptTemplate(
    template=prompt,
    input_variables=["question", "sample"]
)

## JSON to text conversion
json_to_text_prompt = """
    You are a highly intelligent AI that can convert JSON data into a human-readable text format. 
    Take the following JSON data and provide an engaging textual representation of it. 
    answer it in friendly way and you can use a few emojis too in the answer. 
    Do not use too many emojis in the answer
    
    JSON:
    {json_data}
    
    Text:
"""
json_to_text_template = PromptTemplate(
    template=json_to_text_prompt,
    input_variables=["json_data"]
)
json_to_text_chain = LLMChain(llm=llm, prompt=json_to_text_template)

@st.cache_resource(ttl="2h")
def configure_db(db_uri, sql_db_type=None, sql_host=None, sql_user=None, sql_password=None, sql_db=None, 
                 mongo_cluster=None, mongo_user=None, mongo_password=None, mongo_db=None):
    if db_uri == LOCALDB:
        dbfilepath = (Path(__file__).parent.parent / "sqlFiles/student.db").absolute()
        print(dbfilepath)
        creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))
    elif db_uri == SQLDB:
        if not (sql_host and sql_user and sql_password and sql_db):
            st.error("Please provide all SQL connection details.")
            st.stop()
        if sql_db_type == "MySQL":
            return SQLDatabase(create_engine(f"mysql+mysqlconnector://{sql_user}:{sql_password}@{sql_host}/{sql_db}"))
        elif sql_db_type == "PostgreSQL":
            return SQLDatabase(create_engine(f"postgresql+psycopg2://{sql_user}:{sql_password}@{sql_host}/{sql_db}"))
    elif db_uri == MONGODB:
        if not mongo_db:
            st.error("Please provide MongoDB connection details.")
            st.stop()
        mongo_uri = f"mongodb+srv://{mongo_user}:{mongo_password}@{mongo_cluster}.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
        
        client = MongoClient(mongo_uri)
        return client[mongo_db]

if db_uri == SQLDB:
    db = configure_db(db_uri, sql_db_type, sql_host, sql_user, sql_password, sql_db)
elif db_uri == MONGODB:
    db = configure_db(db_uri, mongo_cluster="cluster0.9ypzt", mongo_user=mongo_user, mongo_password=mongo_password, mongo_db=mongo_db)
else:
    db = configure_db(db_uri)

## toolkit
if db_uri in [SQLDB, LOCALDB]:
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=False,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True
    )
else:
    mongo_llm_chain = LLMChain(llm=llm, prompt=mongo_prompt, verbose=True)

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask anything from the database")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    if db_uri == MONGODB:
        with st.chat_message("assistant"):
            st.write("Querying MongoDB...")
            collection = db[mongo_col]
            response=mongo_llm_chain.invoke({"question":user_query, "sample":sample})
            query=json.loads(response["text"])
            results=collection.aggregate(query)
            # print(query)
            for result in results:
                result_text = json_to_text_chain.invoke({"json_data": json.dumps(result, indent=2)})
                st.write(result_text["text"])
                st.session_state.messages.append({"role": "assistant", "content": result_text['text']})
    else:
        with st.chat_message("assistant"):
            streamlit_callback = StreamlitCallbackHandler(st.container())
            response = agent.run(user_query, callbacks=[streamlit_callback])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
