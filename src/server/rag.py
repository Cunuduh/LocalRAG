import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Iterator

import chromadb
import trafilatura
from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import SentenceTransformerEmbeddingFunction
from chromadb.api.models.Collection import Collection
from duckduckgo_search import AsyncDDGS
from llama_cpp import CreateChatCompletionStreamResponse, Llama, ChatCompletionRequestMessage, CreateChatCompletionResponse
from semantic_text_splitter import MarkdownSplitter
from sentence_transformers import CrossEncoder

from config import chroma_config, DEVICE, PROMPT_TEMPLATE, RERANKER_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, WEB_CACHE_FILE

reranker = CrossEncoder(RERANKER_MODEL, device=DEVICE, trust_remote_code=True, max_length=CHUNK_SIZE)
embedding_function = SentenceTransformerEmbeddingFunction(model_name=chroma_config["embedding_model"], device=chroma_config["device"], model_kwargs=chroma_config["model_kwargs"], trust_remote_code=True)

chroma_client = chromadb.PersistentClient()
memory_collection = chroma_client.get_or_create_collection("memory", embedding_function=embedding_function)
web_collection = chroma_client.get_or_create_collection("web", embedding_function=embedding_function)

splitter = MarkdownSplitter(CHUNK_SIZE, CHUNK_OVERLAP)

WEB_CACHE_FILE = Path("cache/web_cache.json")
web_cache = {}
if WEB_CACHE_FILE.exists():
	with open(WEB_CACHE_FILE, "r") as f:
		web_cache = json.load(f)

def rerank(query: str, docs: list[dict], top_k: int = 20) -> list[dict]:
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

def search_chroma(collection: Collection, query: str, n_results: int = 20) -> list[dict]:
	results = collection.query(
		query_texts=[query],
		n_results=n_results*5
	)
	docs = [{"content": doc, "metadata": meta} for doc, meta in zip(results['documents'][0], results['metadatas'][0])]
	if len(docs) == 0:
		return []
	return rerank(query, docs, n_results)

def format_context(docs: list[dict]) -> str:
	formatted_docs = []
	for doc in docs:
		formatted_doc = f"""\t<chunk src="{doc['metadata']['source']}", timestamp="{doc['metadata']['timestamp']}">
	{doc['content']}
	</chunk>"""
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
	text = trafilatura.extract(downloaded, include_links=True, include_images=False, favor_precision=True, deduplicate=True, output_format="markdown")
	return { "url": url, "content": text }

async def load_urls(urls: list[str]) -> list[dict[str, str]]:
	with ThreadPoolExecutor(max_workers=10) as executor:
		loop = asyncio.get_event_loop()
		tasks = [loop.run_in_executor(executor, fetch_url, url) for url in urls]
		return await asyncio.gather(*tasks)

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

async def perform_web_search(query: str, n_results: int = 20) -> list[dict]:
	print(f"Performing search for '{query}'")
	results = await AsyncDDGS().atext(query, max_results=n_results)
	hrefs = [result['href'] for result in results]

	content = await load_urls(hrefs)
	return content

def get_formatted_time() -> str:
	return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

SEARCH_PROMPT_TEMPLATE = [{"role": "system", "content":
"""<task>
Provide extra context to use in your next response using the search tool.
</task>
<instruction>
Given the below context within the <context> tag, craft an effective search query that can be used with the search tool to answer the user query effectively.
</instruction>
<context type="memory">
{memory}
</context>
<instruction>
Make sure to include the specific topic of the conversation (names, events, etc.), and make sure to include important, relevant keywords that expand the query (e.g. synonyms, related concepts).
Do not be too vague with the search query, or else the search tool may not return relevant results.
If the user specifically requests you to search the web, just include 'web' in the 'collections' parameter of the search tool. Do not modify the search query to include the mention of web search.
</instruction>
"""},
	{"role": "user", "content": "{question}"},
]

async def generate_rag(user_input: str, llm: Llama):
	try:
		memory_context_docs = search_chroma(memory_collection, user_input)
		memory_context = format_context(memory_context_docs)
		messages = generate_prompt(SEARCH_PROMPT_TEMPLATE, memory=memory_context, question=user_input)
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
								"description": "The search query to be sent to the search engine. This can be a question, a statement, or a search term."
							},
							"collections": {
								"type": "array",
								"description": "The names of the ChromaDB collections to search in a string array. Available collections are 'memory' and 'web', or none at all (empty array). Please use 'web' for queries that require up-to-date, precise or obscure information.",
							},
							"n_results": {
								"type": "number",
								"description": "The number of search results to return. Default value is 20. Minimum value is 10 for better results and larger context."
							}
						},
						"required": ["query", "collections"]
					}
				}
			}]
		)
		tool_call_params = tool_call["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
		tool_call_args: dict = json.loads(tool_call_params)
		tool_call_args["n_results"] = max(tool_call_args.get("n_results", 20), 10)
		pprint(tool_call_args)

		web_search_context, memory_context = "", ""

		for collection in tool_call_args["collections"]:
			match collection:
				case "web":
					yield f"Searching for '{tool_call_args['query']}'...\n\n"
					web_search_results = await perform_web_search(tool_call_args["query"], tool_call_args["n_results"])
					for web_search_result in web_search_results:
						if web_search_result['content']:
							add_to_chroma(web_collection, web_search_result['content'], {"source": web_search_result['url'], "timestamp": get_formatted_time()})
					web_search_docs = search_chroma(web_collection, tool_call_args["query"], tool_call_args["n_results"])
					web_search_context = format_context(web_search_docs)
				case "memory":
					memory_context_docs = search_chroma(memory_collection, tool_call_args['query'], tool_call_args['n_results'])
					memory_context = format_context(memory_context_docs)

		messages = generate_prompt(PROMPT_TEMPLATE, memory=memory_context, web_search=web_search_context, question=user_input)
		for message in messages:
			print(message["content"])
		result: Iterator[CreateChatCompletionStreamResponse] = llm.create_chat_completion(
			messages=messages,
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
			yield "\n\nSearched:\n" + "\n".join([f"- {web_search_result['url']}" for web_search_result in web_search_results])
		add_to_chroma(memory_collection, user_input, {"source": "user", "timestamp": get_formatted_time()})
		add_to_chroma(memory_collection, response, {"source": "assistant", "timestamp": get_formatted_time()})
	except Exception as e:
		yield f"An error occurred: {str(e)}"
		raise e