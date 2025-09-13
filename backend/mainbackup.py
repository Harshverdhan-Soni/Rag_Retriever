from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import time
import uuid

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from transformers import pipeline

app = FastAPI()

# CORS setup
origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    print("FastAPI server started on http://0.0.0.0:8000")

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"batch_size": 64}
)

# Local LLM
generator = pipeline(
    task="text2text-generation",
    model="google/flan-t5-base",
    device_map="auto",
    max_new_tokens=128
)
llm = HuggingFacePipeline(pipeline=generator)

QA_TEMPLATE = """You are a concise, factual assistant.
Use ONLY the context below. If the answer is not in the context, say "I don't know".
Context:
{context}
Question: 
{question}
Answer:"""
qa_prompt = PromptTemplate(template=QA_TEMPLATE, input_variables=["context", "question"])

# Store QA chains per PDF
qa_chains = {}
progress = {}   # <--- NEW dictionary to track progress


@app.post("/upload")
async def upload_file(file: UploadFile):
    file_id = str(uuid.uuid4())  # unique ID
    progress[file_id] = "Starting upload..."
    print(f"Started ingestion for {file.filename} with file_id={file_id}")

    start = time.time()
    temp_path = f"temp_{file.filename}"

    data = await file.read()
    with open(temp_path, "wb") as f:
        f.write(data)
    progress[file_id] = "PDF uploaded"
    print(f"File uploaded ({len(data)/1e6:.2f} MB in {time.time()-start:.2f}s)")

    # Load and split PDF
    progress[file_id] = "Loading PDF..."
    loader = PyPDFLoader(temp_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages")

    progress[file_id] = "Splitting into chunks..."
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    # Create FAISS DB
    progress[file_id] = "Creating FAISS index..."
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(f"vectorstore_{file_id}")
    print("FAISS index saved")

    # QA chain
    progress[file_id] = "Building QA chain..."
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={"prompt": qa_prompt},
    )
    qa_chains[file_id] = qa_chain

    os.remove(temp_path)
    progress[file_id] = "Done"
    print("Ingestion done")

    return {"status": "Ingestion done", "file_id": file_id}


@app.get("/status/{file_id}")
async def get_status(file_id: str):
    return {"progress": progress.get(file_id, "unknown file_id")}


@app.post("/ask")
async def ask_question(question: str = Form(...), file_id: str = Form(...)):
    try:
        print("\nEntered /ask endpoint")
        print(f"Question: {question}, file_id: {file_id}")

        if file_id not in qa_chains:
            db = FAISS.load_local(
                f"vectorstore_{file_id}",
                embeddings,
                allow_dangerous_deserialization=True
            )
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=db.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type="stuff",
                chain_type_kwargs={"prompt": qa_prompt},
            )
            qa_chains[file_id] = qa_chain
        else:
            qa_chain = qa_chains[file_id]

        start = time.time()
        result = qa_chain.invoke({"query": question})
        print(f"Query done in {time.time()-start:.2f}s")

        answer = result.get("result", "").strip()
        sources = [doc.metadata.get("source") for doc in result["source_documents"]]

        if not sources or not answer:
            answer = "I don't know"

        return {"answer": answer, "sources": sources}

    except Exception as e:
        print(f"Error: {e}")
        return {"answer": "Error occurred", "sources": []}
