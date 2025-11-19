from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pathlib import Path
from backend.orchestrator import Orchestrator
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title='Kitchen AI Compass')
orch = Orchestrator()

class PromptRequest(BaseModel):
    prompt: str
    model: str = "gpt-3.5-turbo"  # Default to GPT-3.5-turbo

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent.parent / 'frontend' / 'index.html'
    with open(html_path, encoding='utf-8') as f:
        return f.read()

@app.post('/api/generate')
async def generate(req: PromptRequest):
    try:
        out = orch.generate_with_context(req.prompt, model=req.model)
        return {'ok': True, 'output': out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/api/ingest_csv')
async def ingest_csv(file: UploadFile = File(...)):
    try:
        import pandas as pd
        from io import BytesIO
        
        # Read file content
        content = await file.read()
        
        # Convert to DataFrame
        df = pd.read_csv(BytesIO(content))
        
        # Ingest the DataFrame
        result = orch.ingest_csv(df)
        
        return {
            'ok': True, 
            'filename': file.filename,
            'rows_ingested': result.get('rows_ingested', 0),
            'total_chunks': result.get('total_chunks', 0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/stats')
async def stats():
    try:
        return {'ok': True, **orch.get_stats()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optional: Endpoint to compare both models
@app.post('/api/compare')
async def compare(req: PromptRequest):
    try:
        result = orch.compare_models(req.prompt)
        return {'ok': True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))