from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from llama_cpp import Llama

from config import llm_config
from rag import generate_rag

app = FastAPI()

llm = Llama(**llm_config)

@app.get("/query/rag")
async def query_rag(user_input: str):
  return StreamingResponse(generate_rag(user_input, llm), media_type="text/event-stream")

if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=8000)
