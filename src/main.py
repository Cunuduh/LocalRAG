import sys
import requests

def main(tool: str, user_input: str):
	if tool == "idx":
		index_dir(user_input)
		return
	url = f"http://localhost:8000/query/{tool}"
	params = { "user_input": user_input }

	print(f"\033[95mAI response:\n\033[0m", end='', flush=True)

	try:
		with requests.get(url, params=params, stream=True) as response:
			response.raise_for_status()
			for line in response.iter_content(chunk_size=None, decode_unicode=True):
				print(line, end='', flush=True)
			print()
			
	except requests.RequestException as e:
		print(f"Error: {e}")
def index_dir(directory_path: str, recursive: bool = True):
	url = f"http://localhost:8000/index_directory"
	params = { "directory_path": directory_path, "recursive": recursive }
	try:
		response = requests.post(url, params=params)
		response.raise_for_status()
		print(response.json()["message"])
	except requests.RequestException as e:
		print(f"Error: {e}")

if __name__ == "__main__":
	if len(sys.argv) != 3:
		print("Usage: python main.py 'tool (raw|web)' 'Your question or prompt here'")
	else:
		tool = sys.argv[1]
		user_input = sys.argv[2]
		main(tool, user_input)

