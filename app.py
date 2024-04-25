import streamlit as st
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from IPython.display import Markdown as md
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Setup API Key
f = open(r"C:\Users\errav\genai\RAG_app\gem_key\gem_api_key.txt")
gem_api_key = f.read()

chat_model = ChatGoogleGenerativeAI(google_api_key=gem_api_key, model="gemini-1.5-pro-latest")

st.header("🙇‍♂️✌️QnA using RAG System")

embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=gem_api_key, model="models/embedding-001")

# Setting a Connection with the ChromaDB
db_connection = Chroma(persist_directory=r"C:\Users\errav\genai\RAG_app\chromadb1", embedding_function=embedding_model)

# Converting CHROMA db_connection to Retriever Object
retriever = db_connection.as_retriever(search_kwargs={"k": 5})

print(type(retriever))

chat_template = ChatPromptTemplate.from_messages([
    # System Message Prompt Template
    SystemMessage(content="""You are a Helpful AI Bot. 
    You take the context and question from user. Your answer should be based on the specific context."""),
    # Human Message Prompt Template
    HumanMessagePromptTemplate.from_template("""Aswer the question based on the given context.
    Context:
    {context}
    
    Question: 
    {question}
    
    Answer: """)
])

output_parser = StrOutputParser()

from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | chat_template
    | chat_model
    | output_parser
)

user_question = st.text_area("Enter your query here:")
if st.button("Ask"):
    response = rag_chain.invoke(user_question)
    st.markdown(response)