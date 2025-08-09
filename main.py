from fastapi import FastAPI, Request
import requests
import os

app = FastAPI()

HF_API_KEY = os.environ.get("HF_API_KEY")
MODEL = "google/flan-t5-small"  # you can change this to other free models

@app.get("/")
def home():
    return {"status": "Jarvis API running"}

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_input = data.get("prompt", "")

    if not user_input:
        return {"error": "No prompt provided"}

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": user_input}

    response = requests.post(
        f"https://api-inference.huggingface.co/models/{MODEL}",
        headers=headers,
        json=payload
    )

    if response.status_code != 200:
        return {"error": f"HF API error {response.status_code}", "details": response.text}

    result = response.json()
    output_text = result[0]["generated_text"] if isinstance(result, list) else result
    return {"response": output_text}
