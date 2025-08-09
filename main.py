from fastapi import FastAPI, Request
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

app = FastAPI()

# Load Model
model_name = "microsoft/phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    result = generator(prompt, max_length=200, do_sample=True, temperature=0.7)[0]["generated_text"]
    return {"response": result}

@app.post("/summarize")
async def summarize(request: Request):
    data = await request.json()
    text = data.get("text", "")
    summary_prompt = f"Summarize this in 3 sentences:\n\n{text}"
    result = generator(summary_prompt, max_length=150)[0]["generated_text"]
    return {"summary": result}
