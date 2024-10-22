import multiprocessing

import torch

local_model = "model/Qwen/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
llm_config = {
  "model_path": local_model,
  "n_ctx": 8192,
  "n_gpu_layers": -1,
  "n_batch": 512,
  "n_threads": multiprocessing.cpu_count() - 1,
}

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

chroma_config = {
  "embedding_model": "Alibaba-NLP/gte-base-en-v1.5",
  "device": DEVICE,
  "model_kwargs": { "torch_dtype": "float16" },
}

WEB_CACHE_FILE = "cache/web_cache.json"

RERANKER_MODEL = "BAAI/bge-reranker-base"

CHUNK_SIZE = 1024
CHUNK_OVERLAP = 128
