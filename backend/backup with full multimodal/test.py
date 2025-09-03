from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"msg": "FastAPI is working"}
