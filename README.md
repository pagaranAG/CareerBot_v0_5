# CareerBot v0.5

CareerBot is a web chatbot that helps Computer Studies students with career paths, skills, resumes, internships, interviews, further studies, role comparisons, and salaries.

## Quick start on Windows

Run these commands in PowerShell from the project folder (`D:\chatbot-sample`).

### 1. Check Python

Python 3.10 or newer is required:

```powershell
python --version
```

If Python is not installed, install it from [python.org](https://www.python.org/downloads/). During installation, enable **Add Python to PATH**.

### 2. Create the virtual environment

Run this once:

```powershell
python -m venv .venv
```

Activate it every time you open a new terminal:

```powershell
.\.venv\Scripts\Activate.ps1
```

You should see `(.venv)` at the beginning of the terminal prompt.

If PowerShell blocks activation, use this once for the current terminal window, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Install the requirements

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The first installation may take a few minutes.

### 4. Add the Groq API key

Create your private environment file:

```powershell
Copy-Item .env.example .env
notepad .env
```

In Notepad, replace this line:

```text
GROQ_API_KEY=your-groq-api-key-here
```

with your real key from the [Groq Console](https://console.groq.com/keys), then save and close Notepad.

Keep the key only in `.env`. Do not paste it into `server.py`, `app.js`, the notebook, screenshots, or GitHub. `.env` is ignored by Git. If a key was exposed previously, revoke it in the Groq Console and create a new one.

You can run the chatbot without a key, but it will use built-in answers instead of live Groq answers.

### 5. Start CareerBot

Make sure the virtual environment is active, then run:

```powershell
python -m uvicorn backend.server:app --reload --port 8000
```

When the terminal shows that Uvicorn is running, open:

- Website: [http://localhost:8000](http://localhost:8000)
- Health check: [http://localhost:8000/api/health](http://localhost:8000/api/health)

The health check should show:

```json
"groq_configured": true
```

To stop the chatbot, return to the terminal and press `Ctrl+C`.

## Everyday run

After the first setup, only these commands are needed:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.server:app --reload --port 8000
```

## Common problems

**`python` is not recognized**

Install Python and enable **Add Python to PATH**, then restart VS Code.

**The virtual environment does not activate**

Run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

**`GROQ_API_KEY` is not configured**

Confirm that the file is named exactly `.env`, is in `D:\chatbot-sample`, and contains the real key without quotation marks or extra spaces. Restart Uvicorn after changing `.env`.

**Port 8000 is already in use**

Use another port:

```powershell
python -m uvicorn backend.server:app --reload --port 8001
```

Then open [http://localhost:8001](http://localhost:8001).

**The first startup is slow**

The embedding model downloads from Hugging Face the first time. Wait for it to finish; later starts should be faster.

## Project files

- `backend/server.py` - FastAPI backend, v0.5 router, Groq integration, and fallback answers
- `frontend/` - responsive web interface (`index.html`, `styles.css`, and `app.js`)
- `artifacts/centroids.pkl` - saved intent-classification data
- `.env.example` - safe configuration template
- `.env` - your private local configuration; never commit it
- `notebooks/` - optional notebook experiments and prototypes

## API example

The browser calls `POST /api/chat` with:

```json
{
  "message": "What skills do I need for data science?",
  "history": []
}
```

The response contains the answer, detected intent, similarity score, route, and whether typo correction was applied.
