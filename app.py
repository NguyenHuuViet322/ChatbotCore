from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langchain.tools import Tool
from vector_store import load_and_process_documents, get_vectorstore
from langchain_chroma import Chroma
import os
import torch
from dotenv import load_dotenv

# --- Load .env variables ---
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Configuration ---
DATA_FOLDER_PATH = "./data"
VECTORSTORE_PERSIST_DIR = "./vectorstore"
EMBEDDING_MODEL_NAME = "sentence-transformers/static-similarity-mrl-multilingual-v1"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
RETRIEVAL_K = 4
TAVILY_MAX_RESULTS = 2

# --- Initialize embedding model ---
hf_embedding = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": False}
)

# --- Process documents and load vectorstore ---
document_splits = load_and_process_documents(
    data_folder=DATA_FOLDER_PATH,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)
vectorstore = get_vectorstore(
    splits=document_splits,
    embedding_model=hf_embedding,
    persist_directory=VECTORSTORE_PERSIST_DIR
)

# --- Define LangChain-compatible tool ---
def retrieve_documents(query: str) -> str:
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    results = retriever.invoke(query)
    if not results:
        return "No relevant documents found."
    formatted = []
    for doc in results:
        source = os.path.basename(doc.metadata.get("source", "N/A"))
        formatted.append(f"Source: {source}\n---\n{doc.page_content}")
    return "\n\n".join(formatted)

retrieve_tool = Tool.from_function(
    name="retrieve_company_documents",
    description="Search internal company documents, policies, or rules.",
    func=retrieve_documents
)

web_search_tool = TavilySearch(
    max_results=TAVILY_MAX_RESULTS,
    topic="general",
    api_key=TAVILY_API_KEY
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.5,
    google_api_key=GOOGLE_API_KEY
)

agent = create_react_agent(llm, [retrieve_tool, web_search_tool])

# --- FastAPI app setup ---
app = FastAPI(title="Company Chatbot API", description="Microservice for internal document Q&A via LangChain + Gemini", version="1.0.0")

class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str

class ChatRequest(BaseModel):
    session_id: str
    messages: List[ChatMessage]

class ChatResponse(BaseModel):
    answer: str

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        messages: List[BaseMessage] = []
        for msg in req.messages:
            if msg.role == 'user':
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == 'assistant':
                messages.append(AIMessage(content=msg.content))

        result = agent.invoke({
            "messages": messages,
            "contents": [msg.content for msg in req.messages]
        })

        for msg in reversed(result["messages"]):
            if isinstance(msg, BaseMessage) and hasattr(msg, "content"):
                return {"answer": msg.content}

        return {"answer": "Không tìm thấy phản hồi phù hợp."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

# Run with: uvicorn app:app --reload
