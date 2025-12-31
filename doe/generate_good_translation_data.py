from openai import OpenAI

fp_openai_key='/home/hungphd/'

# Initialize the client (make sure you set your API key in the environment variable OPENAI_API_KEY)
client = OpenAI()

# Ask a question
question = "What is the capital of France?"

# Send request to ChatGPT
response = client.chat.completions.create(
    model="gpt-5",  # or "gpt-5-pro" if you have access
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question}
    ]
)

# Print the answer
print(response.choices[0].message["content"])
