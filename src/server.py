import asyncio
import json
import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from pprint import pprint

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
from llama_cpp import Llama, ChatCompletionRequestMessage, CreateChatCompletionResponse
from pypdf import PdfReader
from semantic_text_splitter import TextSplitter
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from sentence_transformers import CrossEncoder

app = FastAPI()

# LLM setup
local_model = "model/meta-llama/Llama-3.2-3B-Instruct-Q6_K.gguf"
llm = Llama(
	model_path=local_model,
	n_ctx=8192,
	n_gpu_layers=-1,
	n_batch=300,
	n_threads=multiprocessing.cpu_count() - 1,
)

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
reranker = CrossEncoder("BAAI/bge-reranker-base", device="cuda", trust_remote_code=True, max_length=1024)
embedding_function = SentenceTransformerEmbeddingFunction(model_name="Alibaba-NLP/gte-base-en-v1.5", device="cuda", model_kwargs={ "torch_dtype": "float16" }, trust_remote_code=True)

chroma_client = chromadb.PersistentClient()
memory_collection = chroma_client.get_or_create_collection("memory", embedding_function=embedding_function)
web_collection = chroma_client.get_or_create_collection("web", embedding_function=embedding_function)
file_collection = chroma_client.get_or_create_collection("file", embedding_function=embedding_function)

splitter = TextSplitter(1024, 128)

def rerank(query: str, docs: list[dict], top_k: int = 3) -> list[dict]:
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

def fetch_url(url: str) -> str:
	downloaded = trafilatura.fetch_url(url)
	text = trafilatura.extract(downloaded, include_links=False, include_images=False, favor_precision=True, output_format="markdown")
	return text if text else ""

async def load_urls(urls: list[str]) -> list[str]:
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
		
		# Accept cookies if the dialog appears
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
						'content': ''  # We'll fetch content separately
					})
			except:
				continue  # Skip this result if we can't extract title or URL

			if len(search_results) >= n_results:
				break

	finally:
		driver.quit()

	# Fetch content for each result
	contents = await load_urls([result['url'] for result in search_results])
	for result, content in zip(search_results, contents):
		result['content'] = content

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
				mirostat_mode=2,
				mirostat_tau=2.0,
				tool_choice={ "type": "function", "function": { "name": "search_chroma" } },
				tools=[{
					"type": "function",
					"function": {
						"name": "search_chroma",
						"description": "Search the local ChromaDB vector store for relevant documents, modifying the user's query to be less vague, which is interpreted easier by the retriever.",
						"parameters": {
							"type": "object",
							"properties": {
								"query": {
									"type": "string",
									"description": "The search query. Make sure to include the specific topic of the conversation (e.g. names, events, etc.)."
								},
								"collections": {
									"type": "array",
									"description": "The names of the ChromaDB collections to search in a string array. Available collections are 'memory', 'file', and 'web'.",
								},
								"n_results": {
									"type": "number",
									"description": "The number of search results to return. Default value is 10. Adjust value depending on the scope of the search."
								}
							},
							"required": ["query", "collections"]
						}
					}
				}]
			)
			tool_call_params = tool_call["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
			tool_call_args = json.loads(tool_call_params)
			pprint(tool_call_args)
			for collection in tool_call_args["collections"]:
				match collection:
					case "web":
						web_search_docs = search_chroma(web_collection, tool_call_args["query"], tool_call_args.get("n_results", 10))
						web_search_context = format_context(web_search_docs)
					case "file":
						file_context_docs = search_chroma(file_collection, tool_call_args["query"], tool_call_args.get("n_results", 10))
						file_context = format_context(file_context_docs)
					case "memory":
						memory_context_docs = search_chroma(memory_collection, tool_call_args["query"], tool_call_args.get("n_results", 10))
						memory_context = format_context(memory_context_docs)
			messages = generate_prompt(PROMPT_TEMPLATE, memory=memory_context, web_search=web_search_context, question=user_input)
			result = llm.create_chat_completion(
				messages=messages,
				stream=True,
				temperature=0.25,
				mirostat_mode=2,
				mirostat_tau=2.0
			)
			response = ""
			for chunk in result:
				yield chunk['choices'][0]['delta'].get('content', '')
				response += chunk['choices'][0]['delta'].get('content', '')

			add_to_chroma(memory_collection, f"Human: {user_input}\n\nAI: {response}", {"type": "qa", "timestamp": time.time()})
		except Exception as e:
			yield f"An error occurred: {str(e)}"
			raise e

	return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/query/web")
async def query_web(user_input: str):
	async def generate():
		try:
			memory_context_docs = search_chroma(memory_collection, user_input)
			web_context_docs = search_chroma(web_collection, user_input)
			memory_context = format_context(memory_context_docs)
			web_context = format_context(web_context_docs)
			messages = generate_prompt(PROMPT_TEMPLATE, memory=memory_context, web_search=web_context, question=user_input)
			tool_call: CreateChatCompletionResponse = llm.create_chat_completion(
				messages=messages,
				temperature=0.25,
				mirostat_mode=2,
				mirostat_tau=2.0,
				tool_choice={ "type": "function", "function": { "name": "search" } },
				tools=[{
					"type": "function",
					"function": {
						"name": "search",
						"description": "Search the web using Google for more information on a topic.",
						"parameters": {
							"type": "object",
							"properties": {
								"query": {
									"type": "string",
									"description": "The search query. Make sure to include the specific topic of the conversation (e.g. names, events, etc.)."
								},
								"n_results": {
									"type": "number",
									"description": "The number of search results to return. Default value is 5. Adjust value depending on the scope of the search."
								},
							},
							"required": ["query"]
						}
					}
				}]
			)
			tool_call_params = tool_call["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
			query = json.loads(tool_call_params)["query"]

			yield f"Searching for '{query}'...\n\n"

			search_results = await perform_web_search(query)
			formatted_results = [result["content"] for result in search_results]

			add_to_chroma(web_collection, "\n\n".join(formatted_results), {"type": "search", "timestamp": time.time()})

			web_context_docs = search_chroma(web_collection, query)
			web_context = format_context(web_context_docs)
			messages = generate_prompt(PROMPT_TEMPLATE, memory=memory_context, web_search=web_context, question=user_input)
			pprint(messages, indent=1)
			result = llm.create_chat_completion(
				messages=messages,
				stream=True,
				temperature=0.25,
				mirostat_mode=2,
				mirostat_tau=2.0
			)

			response = ""
			for chunk in result:
				yield chunk['choices'][0]['delta'].get('content', '')
				response += chunk['choices'][0]['delta'].get('content', '')
			sources = "\n\nSources:\n" + "\n".join([f"- {result['url']}" for result in search_results])
			yield sources
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