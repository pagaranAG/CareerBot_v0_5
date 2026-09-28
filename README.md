# CareerBot web UI

A minimal, responsive browser interface for the attached CareerBot prototype. It now includes the original notebook, its `centroids.pkl` artifact, and a FastAPI server that loads the artifact and runs the embedding/classification flow.

## Run it

Install the Python dependencies once:

```powershell
python -m pip install -r requirements.txt
```

Set the server-side Groq key in your shell:

```powershell
$env:GROQ_API_KEY = 'your-key-here'
```

Then run:

```powershell
uvicorn server:app --reload --port 8000
```

Open http://localhost:8000 in a browser. The API health check is available at http://localhost:8000/api/health.

The first run downloads `all-MiniLM-L6-v2` from Hugging Face. The copied `centroids.pkl` is loaded at startup, so the server does not recompute centroids.

## Connect the real model

The browser is already configured to call `/api/chat`. The endpoint accepts:

```js
{
	"message": "What skills do I need for data science?",
	"history": []
}
```

and return:

```json
{
	"answer": "...",
	"intent": "skill_recommendation",
	"score": 0.82
}
```

Keep the Groq key and model-loading code on the Python server. Do not place either in `app.js` or expose the key in a deployed frontend. The notebook's Google Drive persistence can remain server-side where `centroids.pkl` is loaded at startup.

