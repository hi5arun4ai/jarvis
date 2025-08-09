from fastapi import FastAPI, Request
from pydantic import BaseModel
import requests
import os

app = FastAPI()

HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-small"
HF_API_KEY = os.getenv("HF_API_KEY")  # Set this in Render dashboard

headers = {"Authorization": f"Bearer {HF_API_KEY}"}

class Prompt(BaseModel):
    prompt: str

@app.post("/chat")
async def chat(data: Prompt):
    payload = {"inputs": data.prompt}
    response = requests.post(HF_API_URL, headers=headers, json=payload)
    result = response.json()
    return {"response": result[0]['generated_text'] if isinstance(result, list) else result}

@app.get("/")
def home():
    return {"status": "Jarvis API is running!"}
