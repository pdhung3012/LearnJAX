import os
from google import genai

# 1. Point explicitly to your downloaded JSON key file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/hungphd/hung-gemini-project-0c4f3f87a126.json"

# 2. Initialize the client for Vertex AI
# The client will automatically detect the environment variable and authenticate as the service account.
client = genai.Client(
    vertexai=True,
    project="hung-gemini-project",  # Replace with your actual GCP project ID
    location="global"      # Replace with your preferred region if different
)

# 3. Make the inference call
response = client.models.generate_content(
    model="gemini-3.1-pro-preview",
    contents="Explain the fundamentals of machine learning."
)

print(response.text)