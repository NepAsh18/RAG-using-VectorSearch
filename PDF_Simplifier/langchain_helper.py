import os
from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

CHROMA_PATH = "chroma"

embeddings = GPT4AllEmbeddings(
    model_name="all-MiniLM-L6-v2.gguf2.f16.gguf",
    gpt4all_kwargs={"allow_download": "True"}
)

db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
retriever = db.as_retriever(search_type="similarity")

llm = GoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)


contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question."
)

qa_system_prompt = """Use ONLY the following context to answer the question. 
Respond in 3–7 sentences. If you don't know the answer, just say that you don't know.

Context: {context}"""


contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


contextualize_chain = contextualize_q_prompt | llm | StrOutputParser()


def contextualized_retrieval(input_data):
    if input_data.get("chat_history"):
        
        reformulated_question = contextualize_chain.invoke({
            "chat_history": input_data["chat_history"],
            "input": input_data["input"]
        })
        
        docs = retriever.invoke(reformulated_question)
    else:
       
        docs = retriever.invoke(input_data["input"])
    
    return format_docs(docs)

rag_chain = (
    RunnablePassthrough.assign(
        context=contextualized_retrieval
    )
    | qa_prompt
    | llm
    | StrOutputParser()
)

def execute_user_query(query, chat_history=[]):
    response = rag_chain.invoke({
        "input": query,
        "chat_history": chat_history
    })
    
    return response
