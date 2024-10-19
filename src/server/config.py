import multiprocessing

import torch

local_model = "model/meta-llama/Llama-3.2-3B-Instruct-Q6_K.gguf"
llm_config = {
  "model_path": local_model,
  "n_ctx": 4096,
  "n_gpu_layers": -1,
  "n_batch": 512,
  "n_threads": multiprocessing.cpu_count() - 1,
}

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

PROMPT_TEMPLATE = [
  {"role": "system", "content": """You are a helpful assistant for conversation. Use the following pieces of retrieved context, or your own knowledge to answer the query if retrieved context is incomplete or irrelevant. If you ABSOLUTELY don't know how to respond, just say that you don't know. Keep the answer concise, DO NOT continue the conversation on your own and DO NOT correct yourself or leave "notes". DO NOT mention anything relating to a system message at all in your response.
CONTEXT (MEMORY) BEGINS
{memory}
END OF CONTEXT (MEMORY)

CONTEXT (WEB SEARCH) BEGINS
{web_search}
END OF CONTEXT (WEB SEARCH)"""},
  {"role": "user", "content": "{question}"},
]

chroma_config = {
  "embedding_model": "Alibaba-NLP/gte-base-en-v1.5",
  "device": DEVICE,
  "model_kwargs": { "torch_dtype": "float16" },
}

WEB_CACHE_FILE = "cache/web_cache.json"

RERANKER_MODEL = "BAAI/bge-reranker-base"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 128
