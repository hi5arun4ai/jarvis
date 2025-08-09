# jarvis
Personal Assistant (experimental)
# Jarvis AI (Free Cloud-Hosted AI Assistant)

A lightweight AI agent powered by Microsoft's Phi-3-mini model, deployed on Railway, accessible via REST API.

## Endpoints
- `/` → Health check
- `/chat` → Send prompt and get AI response
- `/summarize` → Summarize any text in 3 sentences

## Example Request
```bash
curl -X POST https://<your-railway-url>.up.railway.app/chat \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Hello Jarvis"}'
