from fastapi import FastAPI, Request
import requests
import os

app = FastAPI()

HF_API_KEY = os.getenv("HF_API_KEY")
MODEL_NAME = "google/flan-t5-small"

@app.get("/")
def home():
    return {"message": "Jarvis API is running"}

@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        prompt = data.get("prompt", "")

        if not prompt:
            return {"error": "No prompt provided"}

        # Hugging Face API call
        headers = {
            "Authorization": f"Bearer {HF_API_KEY}"
        }
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 100}
        }

        response = requests.post(
            f"https://api-inference.huggingface.co/models/{MODEL_NAME}",
            headers=headers,
            json=payload
        )

        if response.status_code != 200:
            return {"error": "Hugging Face API call failed", "details": response.text}

        result = response.json()

        # Handle text output format
        if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
            output_text = result[0]["generated_text"]
        else:
            output_text = result

        return {"response": output_text}

    except Exception as e:
        return {"error": str(e)}
