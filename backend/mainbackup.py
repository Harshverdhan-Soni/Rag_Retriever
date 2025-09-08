from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import os

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline
import time
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

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"batch_size": 64}   # default is 32 or lower
)
# Local LLM
generator = pipeline(
    task="text2text-generation",
    model="google/flan-t5-base",   # or "google/flan-t5-large" if you have RAM
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

@app.post("/upload")
async def upload_file(file: UploadFile):
    file_id = file.filename
    temp_path = f"temp_{file_id}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # Load and split PDF
    loader = PyPDFLoader(temp_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Create FAISS DB
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(f"vectorstore_{file_id}")

    # Create QA chain with strict prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={"prompt": qa_prompt},
    )
    qa_chains[file_id] = qa_chain

    os.remove(temp_path)

    return {"status": "Ingestion done", "file_id": file_id}

@app.post("/ask")
async def ask_question(question: str = Form(...), file_id: str = Form(...)):
    try:
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
                chain_type="stuff",
                chain_type_kwargs={"prompt": qa_prompt},
            )
            qa_chains[file_id] = qa_chain
        else:
            qa_chain = qa_chains[file_id]

        print(f"Running query: {question}")
        start = time.time()
        result = qa_chain.invoke({"query": question})
        print(f"Query completed in {time.time() - start:.2f} seconds")

        answer = result.get("result", "").strip()
        sources = [doc.metadata.get("source") for doc in result["source_documents"]]

        # If no sources OR answer looks empty => return "I don't know"
        if not sources or not answer or answer.lower() in ["", "unknown", "i don't know"]:
            answer = "I don't know"

        return {
            "answer": answer,
            "sources": sources,
        }

    except Exception as e:
        print(f"Error in /ask endpoint: {e}")
        return {
            "answer": "Error occurred while processing your query.",
            "sources": []
        }