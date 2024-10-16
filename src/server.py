import asyncio
import json
import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from pprint import pprint
from typing import Iterator

import chromadb
import trafilatura
import uvicorn
from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import SentenceTransformerEmbeddingFunction
from chromadb.api.models.Collection import Collection
from docx import Document
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.table import Table
from docx.text.paragraph import Paragraph
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from llama_cpp import CreateChatCompletionStreamResponse, Llama, ChatCompletionRequestMessage, CreateChatCompletionResponse, LogitsProcessorList
from pypdf import PdfReader
from semantic_text_splitter import TextSplitter
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from sentence_transformers import CrossEncoder

from entropix_sampler import SamplerConfig, EntropixLogitsProcessor

app = FastAPI()

# LLM setup
local_model = "model/Qwen/Qwen2.5-3B-Instruct.Q6_K.gguf"
llm = Llama(
	model_path=local_model,
	n_ctx=4096,
	n_gpu_layers=-1,
	n_batch=512,
	n_threads=multiprocessing.cpu_count() - 1,
)
entropix_config = SamplerConfig()
logits_processor = EntropixLogitsProcessor(entropix_config)
logits_processor_list = LogitsProcessorList([logits_processor])

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
reranker = CrossEncoder("BAAI/bge-reranker-base", device="cuda", trust_remote_code=True, max_length=512)
embedding_function = SentenceTransformerEmbeddingFunction(model_name="Alibaba-NLP/gte-base-en-v1.5", device="cuda", model_kwargs={ "torch_dtype": "float16" }, trust_remote_code=True)

chroma_client = chromadb.PersistentClient()
memory_collection = chroma_client.get_or_create_collection("memory", embedding_function=embedding_function)
web_collection = chroma_client.get_or_create_collection("web", embedding_function=embedding_function)
file_collection = chroma_client.get_or_create_collection("file", embedding_function=embedding_function)

splitter = TextSplitter(512, 128)

WEB_CACHE_FILE = Path("cache/web_cache.json")
web_cache = {}
if WEB_CACHE_FILE.exists():
	with open(WEB_CACHE_FILE, "r") as f:
		web_cache = json.load(f)

def rerank(query: str, docs: list[dict], top_k: int = 5) -> list[dict]:
	scores = reranker.predict([(query, doc['content']) for doc in docs])
	sorted_docs = sorted(docs, key=lambda x: scores[docs.index(x)], reverse=True)[:top_k]
	return sorted_docs

def add_to_chroma(collection: Collection, text: str, metadata: dict, max_workers: int = 4):
	chunks = splitter.chunks(text)

	def add_chunk(chunk, index):
		chunk_metadata = metadata.copy()
		chunk_metadata['chunk_index'] = index
		collection.add(
			documents=[chunk],
			metadatas=[chunk_metadata],
			ids=[f"{time.time()}_{index}"],
		)

	with ThreadPoolExecutor(max_workers=max_workers) as executor:
		futures = [executor.submit(add_chunk, chunk, i) for i, chunk in enumerate(chunks)]
		for future in as_completed(futures):
			future.result()

def search_chroma(collection: Collection, query: str, n_results: int = 10) -> list[dict]:
	results = collection.query(
		query_texts=[query],
		n_results=n_results
	)
	docs = [{"content": doc, "metadata": meta} for doc, meta in zip(results['documents'][0], results['metadatas'][0])]
	if len(docs) == 0:
		return []
	return rerank(query, docs)


def format_context(docs: list[dict]) -> str:
	formatted_docs = []
	for doc in docs:
		formatted_doc = f"Chunk {doc['metadata']['chunk_index']}:\n{doc['content']}"
		formatted_docs.append(formatted_doc)
	return "\n\n".join(formatted_docs)

def generate_prompt(template: list[ChatCompletionRequestMessage], **kwargs) -> list[ChatCompletionRequestMessage]:
	prompt = deepcopy(template)
	for message in prompt:
		if 'content' in message:
			message['content'] = message['content'].format(**kwargs)

	return prompt

def fetch_url(url: str) -> dict[str, str]:
	downloaded = trafilatura.fetch_url(url)
	text = trafilatura.extract(downloaded, include_links=False, include_images=False, favor_precision=True, output_format="markdown")
	return { "url": url, "content": text }

async def load_urls(urls: list[str]) -> list[dict[str, str]]:
	with ThreadPoolExecutor(max_workers=10) as executor:
		loop = asyncio.get_event_loop()
		tasks = [loop.run_in_executor(executor, fetch_url, url) for url in urls]
		return await asyncio.gather(*tasks)

def convert_docx_to_text(file_path: Path):
	doc = Document(file_path)
	md = []

	def parse_paragraph(paragraph: Paragraph):
		text = paragraph.text.strip()
		if not text:
			return ""

		style = paragraph.style.name
		if style.startswith('Heading'):
			level = style[-1]
			return f"{'#' * int(level)} {text}\n\n"
		else:
			return f"{text}\n\n"

	def parse_table(table: Table):
		md_table = "| " + " | ".join(cell.text for cell in table.rows[0].cells) + " |\n"
		md_table += "|" + "|".join(["---"] * len(table.columns)) + "|\n"
		for row in table.rows[1:]:
			md_table += "| " + " | ".join(cell.text for cell in row.cells) + " |\n"
		return md_table + "\n"

	for element in doc.element.body:
		if isinstance(element, CT_P):
			md.append(parse_paragraph(Paragraph(element, doc)))
		elif isinstance(element, CT_Tbl):
			md.append(parse_table(Table(element, doc)))

	return "".join(md)

def convert_pdf_to_text(file_path: Path):
	reader = PdfReader(file_path)
	return '\n'.join([page.extract_text() for page in reader.pages])

def get_file_content(file_path: Path):
	file_extension = file_path.suffix.lower()
	if file_extension == '.docx':
		return convert_docx_to_text(file_path)
	elif file_extension == '.pdf':
		return convert_pdf_to_text(file_path)
	else:
		with open(file_path, 'r', encoding='utf-8') as file:
			return file.read()

def index_directory(directory_path: str, recursive: bool = False):
	path = Path(directory_path)
	if not path.is_dir():
		raise ValueError(f"{directory_path} is not a valid directory")

	pattern = '**/*' if recursive else '*'
	for file_path in path.glob(pattern):
		if file_path.is_file():
			try:
				content = get_file_content(file_path)
				add_to_chroma(file_collection, content, {
					"file_path": str(file_path),
					"file_name": file_path.name,
					"file_type": file_path.suffix,
					"timestamp": time.time()
				})
				print(f"Indexed: {file_path}")
			except Exception as e:
					print(f"Error indexing {file_path}: {str(e)}")

def add_to_web_cache(url: str, content: str):
	web_cache[url] = {
		"content": content,
		"timestamp": time.time()
	}
	if WEB_CACHE_FILE.exists():
		with open(WEB_CACHE_FILE, "w") as f:
			json.dump(web_cache, f)

def get_from_web_cache(urls: list[str]) -> dict[str, str | None]:
	return {url: web_cache[url]["content"] if url in web_cache else None for url in urls}

async def perform_web_search(query: str, n_results: int = 10) -> list[dict]:
  print(f"Performing search for '{query}'")
  
  options = webdriver.ChromeOptions()
  options.add_argument('--headless=new')
  options.add_argument("--window-position=-2400,-2400")
  options.add_argument('--no-sandbox')
  options.add_argument('--disable-dev-shm-usage')
  driver = webdriver.Chrome(options=options)

  try:
    driver.get("https://www.google.com")
    
    try:
      accept_button = WebDriverWait(driver, 5).until(
        EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Accept all')]"))
      )
      accept_button.click()
    except:
      pass

    search_box = WebDriverWait(driver, 10).until(
      EC.presence_of_element_located((By.NAME, "q"))
    )
    search_box.send_keys(query)
    search_box.send_keys(Keys.RETURN)

    WebDriverWait(driver, 10).until(
      EC.presence_of_element_located((By.CSS_SELECTOR, "div.g"))
    )

    results = driver.find_elements(By.CSS_SELECTOR, "div.g")
    search_results = []

    for result in results[:n_results]:
      try:
        title_element = result.find_element(By.CSS_SELECTOR, "h3")
        url_element = result.find_element(By.CSS_SELECTOR, "a")
        
        title = title_element.text
        url = url_element.get_attribute("href")
        
        if title and url:
          search_results.append({
            'title': title,
            'url': url,
            'content': ''
          })
      except:
        continue

      if len(search_results) >= n_results:
        break

  finally:
    driver.quit()

  urls = [result['url'] for result in search_results]
  cached_contents = get_from_web_cache(urls)
  urls_to_fetch = [url for url, content in cached_contents.items() if content is None]

  if urls_to_fetch:
    fetched_contents = await load_urls(urls_to_fetch)
    
    for fetched in fetched_contents:
      for result in search_results:
        if result['url'] == fetched['url']:
          result['content'] = fetched['content']
          add_to_web_cache(fetched['url'], fetched['content'])
          break

  for result in search_results:
    if result['content'] == '' and cached_contents[result['url']]:
      result['content'] = cached_contents[result['url']]

  return search_results

@app.get("/query/rag")
async def query_rag(user_input: str):
	async def generate():
		try:
			memory_context_docs = search_chroma(memory_collection, user_input)
			memory_context = format_context(memory_context_docs)
			messages = generate_prompt(PROMPT_TEMPLATE, memory=memory_context, web_search="", question=user_input)
			tool_call: CreateChatCompletionResponse = llm.create_chat_completion(
				messages=messages,
				temperature=0.25,
				tool_choice={ "type": "function", "function": { "name": "search" } },
				tools=[{
					"type": "function",
					"function": {
						"name": "search",
						"description": "Search the local vector store for relevant documents or search the web using Google for relevant results.",
						"parameters": {
							"type": "object",
							"properties": {
								"query": {
									"type": "string",
									"description": "The search query. Make sure to include the specific topic of the conversation (e.g. names, events, etc.), and make sure to be concise, specific and direct."
								},
								"collections": {
									"type": "array",
									"description": "The names of the ChromaDB collections to search in a string array. Available collections are 'memory', 'file', and 'web', or none at all (empty array). Please use 'web' for queries that require up-to-date, precise or obscure information.",
								},
								"n_results": {
									"type": "number",
									"description": "The number of search results to return. Default value is 10. Adjust value depending on the scope of the search."
								}
							},
							"required": ["query", "collections", "n_results"]
						}
					}
				}]
			)
			tool_call_params = tool_call["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
			tool_call_args = json.loads(tool_call_params)

			pprint(tool_call_args)

			web_search_context, file_context, memory_context = "", "", ""

			for collection in tool_call_args["collections"]:
				match collection:
					case "web":
						yield f"Searching for '{tool_call_args['query']}'...\n\n"
						web_search_results = await perform_web_search(tool_call_args["query"], tool_call_args["n_results"])
						for web_search_result in web_search_results:
							if web_search_result["content"]:
								add_to_chroma(web_collection, web_search_result['content'], {"source": web_search_result['url'], "title": web_search_result['title']})
						web_search_docs = search_chroma(web_collection, tool_call_args["query"], tool_call_args["n_results"])
						web_search_context = format_context(web_search_docs)
					case "file":
						file_context_docs = search_chroma(file_collection, tool_call_args["query"], tool_call_args["n_results"])
						file_context = format_context(file_context_docs)
					case "memory":
						memory_context_docs = search_chroma(memory_collection, tool_call_args["query"], tool_call_args["n_results"])
						memory_context = format_context(memory_context_docs)

			messages = generate_prompt(PROMPT_TEMPLATE, memory=memory_context, web_search=web_search_context, question=user_input)
			pprint(messages)
			result: Iterator[CreateChatCompletionStreamResponse] = llm.create_chat_completion(
				messages=messages,
				logits_processor=logits_processor_list,
				temperature=0.666,
        top_p=0.9,
        top_k=27,
        min_p=0.03,
				stream=True,
			)
			response = ""
			for chunk in result:
				yield chunk['choices'][0]['delta'].get('content', '')
				await asyncio.sleep(0)
				response += chunk['choices'][0]['delta'].get('content', '')
			if web_search_context:
				yield "\n\nSources:\n" + "\n".join([f"- {web_search_result['url']}" for web_search_result in web_search_results])
			add_to_chroma(memory_collection, f"Human: {user_input}\n\nAI: {response}", {"type": "qa", "timestamp": time.time()})
		except Exception as e:
			yield f"An error occurred: {str(e)}"
			raise e

	return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/index_directory")
async def index_directory_endpoint(directory_path: str, recursive: bool = False):
	try:
		index_directory(directory_path, recursive)
		return {"message": f"Successfully indexed directory: {directory_path}"}
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
	uvicorn.run(app, host="0.0.0.0", port=8000)