# clinical-qa

Flask web app for VividMedi clinical documentation workflows.

## Local run (Windows)

```powershell
cd C:\Users\micha\Documents\dev\clinical-qa
.\.venv\Scripts\Activate.ps1
python app.py
```

Set the required environment variables first. At minimum, local authentication
now requires `AUTH_CODE`; AI, calling, and transcription features also need
their corresponding API keys from `.env.example`.
