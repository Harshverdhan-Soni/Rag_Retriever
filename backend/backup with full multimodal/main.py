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

# --- CORS setup for frontend ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Embeddings (latest package) ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Local LLM (via Transformers + HuggingFacePipeline) ---
generator = pipeline(
    "text-generation",
    model="HuggingFaceH4/zephyr-7b-alpha",   # replace with GPU model if needed
    device_map="auto",
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
)
llm = HuggingFacePipeline(pipeline=generator)

qa_chain = None


@app.post("/upload")
async def upload_file(file: UploadFile):
    """Upload and ingest PDF into FAISS vectorstore"""
    global qa_chain
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    loader = PyPDFLoader(temp_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local("vectorstore")

    # Create retriever + chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
    )

    os.remove(temp_path)
    return {"status": "Ingestion done"}


@app.post("/ask")
async def ask_question(question: str = Form(...)):
    """Ask questions from ingested PDF"""
    global qa_chain
    if qa_chain is None:
        db = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=db.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
        )

    result = qa_chain({"query": question})
    return {
        "answer": result["result"],
        "sources": [doc.metadata.get("source") for doc in result["source_documents"]],
    }
