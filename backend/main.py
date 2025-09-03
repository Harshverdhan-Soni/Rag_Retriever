from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import os

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
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

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Local LLM
generator = pipeline(
    "text-generation",
    model="HuggingFaceH4/zephyr-7b-alpha",  # change if GPU available
    device_map="auto",
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
)
llm = HuggingFacePipeline(pipeline=generator)

# Store QA chains per PDF
qa_chains = {}

@app.post("/upload")
async def upload_file(file: UploadFile):
    file_id = file.filename
    temp_path = f"temp_{file_id}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # Load and split PDF
    loader = PyPDFLoader(temp_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Create FAISS DB
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(f"vectorstore_{file_id}")

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
    )
    qa_chains[file_id] = qa_chain

    os.remove(temp_path)

    # Return file_id so frontend can query
    return {"status": "Ingestion done", "file_id": file_id}


@app.post("/ask")
async def ask_question(question: str = Form(...), file_id: str = Form(...)):
    try:
        # If chain not already created for this file_id, build it
        if file_id not in qa_chains:
            print(f"Loading vectorstore for file_id={file_id}...")
            db = FAISS.load_local(
                f"vectorstore_{file_id}",
                embeddings,
                allow_dangerous_deserialization=True
            )
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=db.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
            )
            print("QA chain created")
            qa_chains[file_id] = qa_chain
        else:
            qa_chain = qa_chains[file_id]
            print(f"ℹUsing cached QA chain for file_id={file_id}")

        # Run the chain
        print(f"Running query: {question}")
        result = qa_chain.invoke({"query": question})
        print("Query completed")

        return {
            "answer": result["result"],
            "sources": [doc.metadata.get("source") for doc in result["source_documents"]],
        }

    except Exception as e:
        print(f"Error in /ask endpoint: {e}")
        return {
            "answer": "Error occurred while processing your query.",
            "sources": []
        }

