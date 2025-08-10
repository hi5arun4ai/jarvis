from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os

app = FastAPI()

HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-small"
HF_API_KEY = os.getenv("HF_API_KEY")

if not HF_API_KEY:
    raise RuntimeError("HF_API_KEY is missing. Set it in Render environment variables.")

headers = {"Authorization": f"Bearer {HF_API_KEY}"}

class Prompt(BaseModel):
    prompt: str

@app.post("/chat")
async def chat(data: Prompt):
    payload = {"inputs": data.prompt}
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.json())
        result = response.json()
        return {
            "response": result[0]['generated_text'] if isinstance(result, list) else result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"status": "Jarvis API is running!"}
