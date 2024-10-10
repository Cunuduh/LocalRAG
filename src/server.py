import asyncio
import json
import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pprint import pprint

import chromadb
import trafilatura
import uvicorn
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from chromadb.api.models.Collection import Collection
from duckduckgo_search import AsyncDDGS
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from html2text import HTML2Text
from llama_cpp import Llama, ChatCompletionRequestMessage, CreateChatCompletionResponse
from semantic_text_splitter import TextSplitter

app = FastAPI()

# LLM setup
local_model = "model/meta-llama/Llama-3.2-3B-Instruct-Q6_K.gguf"
llm = Llama(
	model_path=local_model,
	n_ctx=100000,
	n_gpu_layers=8,
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

# Chroma setup
chroma_client = chromadb.PersistentClient()
memory_collection = chroma_client.get_or_create_collection("memory", embedding_function=SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2", device="cuda"))
web_collection = chroma_client.get_or_create_collection("web", embedding_function=SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2", device="cuda"))

h = HTML2Text()
splitter = TextSplitter(1024, 128)

def add_to_chroma(collection: Collection, text: str, metadata: dict, max_workers: int =4):
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
            future.result()  # This will raise any exceptions that occurred during execution


def search_chroma(collection: Collection, query: str, n_results: int = 4) -> list[dict]:
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return [{"content": doc, "metadata": meta} for doc, meta in zip(results['documents'][0], results['metadatas'][0])]

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

async def search(query: str, n_results: int = 1):
	results = await AsyncDDGS(proxy=None).atext(query, max_results=n_results)
	return results

def fetch_url(url: str) -> str:
    downloaded = trafilatura.fetch_url(url)
    text = trafilatura.extract(downloaded, include_links=False, include_images=False)
    return text if text else ""

async def load_urls(urls: list[str]) -> list[str]:
    with ThreadPoolExecutor(max_workers=10) as executor:
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(executor, fetch_url, url) for url in urls]
        return await asyncio.gather(*tasks)

async def perform_web_search(query: str, n_results: int = 3) -> list[dict]:
	"""
	Perform a web search using DuckDuckGo and return the search results as a list of dictionaries.

	Args:
		query (str): The search query.
		n_results (int): The number of search results to return. Default value is 3. Adjust value depending on the scope of the search.

	Returns:
		A list of dictionaries containing the title, URL, and content of each search result.
	"""
	print(f"Performing search for '{query}'")
	results = await search(query, n_results)
	urls = [result['href'] for result in results]
	contents = await load_urls(urls)
	
	search_results = []
	for result, content in zip(results[:5], contents):
		search_results.append({
			'title': result['title'],
			'url': result['href'],
			'content': content
		})
	
	return search_results

def format_search_results(results: list[dict]) -> list[str]:
	return [f"Title: {result['title']}\n{result['url']}\n{h.handle(result['content'])}" for result in results]

@app.get("/query/raw")
async def query(user_input: str):
	async def generate():
		try:
			memory_context_docs = search_chroma(memory_collection, user_input)
			web_context_docs = search_chroma(web_collection, user_input)
			memory_context = format_context(memory_context_docs)
			web_context = format_context(web_context_docs)
			messages = generate_prompt(PROMPT_TEMPLATE, memory=memory_context, web_search=web_context, question=user_input)
			
			result = llm.create_chat_completion(
				messages=messages,
				stream=True,
				temperature=0.5,
				mirostat_mode=2,
				mirostat_tau=3.0
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
				mirostat_tau=3.0,
				tool_choice={ "type": "function", "function": { "name": "search" } },
				tools=[{
					"type": "function",
					"function": {
						"name": "search",
						"description": "Search the web using DuckDuckGo for more information on a topic.",
						"parameters": {
							"type": "object",
							"properties": {
								"query": {
									"type": "string",
									"description": "The search query. Make sure to include the specific topic of the conversation (e.g. names, events, etc.)."
								},
								"n_results": {
									"type": "number",
									"description": "The number of search results to return. Default value is 3. Adjust value depending on the scope of the search."
								},
							},
							"required": ["query"]
						}
					}
				}]
			)
			tool_call_params = tool_call["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
			query = json.loads(tool_call_params)["query"]

			yield query

			search_results = await perform_web_search(query)
			formatted_results = format_search_results(search_results)

			add_to_chroma(web_collection, "\n\n".join(formatted_results), {"type": "search", "timestamp": time.time()})

			memory_context_docs = search_chroma(memory_collection, user_input)
			web_context_docs = search_chroma(web_collection, query)
			memory_context = format_context(memory_context_docs)
			web_context = format_context(web_context_docs)
			messages = generate_prompt(PROMPT_TEMPLATE, memory=memory_context, web_search=web_context, question=user_input)
			pprint(messages, indent=1)
			result = llm.create_chat_completion(
				messages=messages,
				stream=True,
				temperature=0.5,
				mirostat_mode=2,
				mirostat_tau=3.0
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

if __name__ == "__main__":
	uvicorn.run(app, host="0.0.0.0", port=8000)