from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import os

# LangChain updated imports
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline

app = FastAPI()

# --- CORS setup ---
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # restrict to your frontend origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Embeddings ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Local LLM ---
generator = pipeline(
    "text-generation",
    model="HuggingFaceH4/zephyr-7b-alpha",  # replace with smaller model for testing
    device_map="auto",
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
)
llm = HuggingFacePipeline(pipeline=generator)

# Store QA chains per document
qa_chains = {}


@app.post("/upload")
async def upload_file(file: UploadFile):
    """Upload and ingest PDF into FAISS vectorstore"""
    file_id = file.filename  # use filename as file_id
    temp_path = f"temp_{file_id}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # PDF ingestion
    loader = PyPDFLoader(temp_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(f"vectorstore_{file_id}")

    # Create retriever + chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
    )
    qa_chains[file_id] = qa_chain

    os.remove(temp_path)

    # ✅ Return file_id along with status
    return {"status": "Ingestion done", "file_id": file_id}

@app.post("/ask")
async def ask_question(question: str = Form(...), file_id: str = Form(...)):
    """Ask questions from ingested PDF"""
    if file_id not in qa_chains:
        db = FAISS.load_local(f"vectorstore_{file_id}", embeddings, allow_dangerous_deserialization=True)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=db.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
        )
        qa_chains[file_id] = qa_chain
    else:
        qa_chain = qa_chains[file_id]

    result = qa_chain({"query": question})
    return {
        "answer": result["result"],
        "sources": [doc.metadata.get("source") for doc in result["source_documents"]],
    }
