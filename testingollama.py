import ollama

response = ollama.chat(
    model='llama3.2-vision',
    messages=[{'role': 'user', 'content': 'Hello'}]
)

print(response['message']['content'])