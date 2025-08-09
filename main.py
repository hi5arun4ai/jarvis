from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

app = FastAPI()

model_name = "google/flan-t5-small"  # Light model for low RAM
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

@app.get("/")
def home():
    return {"message": "Jarvis (light) is online"}

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    result = generator(prompt, max_new_tokens=150)[0]["generated_text"]
    return {"response": result}

@app.post("/summarize")
async def summarize(request: Request):
    data = await request.json()
    text = data.get("text", "")
    summary_prompt = f"Summarize this in 3 sentences: {text}"
    result = generator(summary_prompt, max_new_tokens=100)[0]["generated_text"]
    return {"summary": result}
