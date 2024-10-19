from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from llama_cpp import Llama

from config import llm_config
from rag import generate_rag, index_directory

app = FastAPI()

llm = Llama(**llm_config)

@app.get("/query/rag")
async def query_rag(user_input: str):
  return StreamingResponse(generate_rag(user_input, llm), media_type="text/event-stream")

@app.post("/index_directory")
async def index_directory_endpoint(directory_path: str, recursive: bool = False):
  try:
    index_directory(directory_path, recursive)
    return {"message": f"Successfully indexed directory: {directory_path}"}
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=8000)
