"""
Enhanced FastAPI App for Kitchen Compass AI
Includes agent status tracking, streaming responses, and comprehensive error handling
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from pathlib import Path
from backend.orchestrator import Orchestrator
from dotenv import load_dotenv
import asyncio
import json
import time

# Load environment variables
load_dotenv()

app = FastAPI(
    title='Kitchen AI Compass - Enhanced',
    description='Multi-agent AI system for NYC restaurant entrepreneurs',
    version='2.0.0'
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize orchestrator
orch = Orchestrator()

# Request Models
class PromptRequest(BaseModel):
    prompt: str
    model: str = "gpt-4o-mini"
    stream: bool = False

class AgentExecutionRequest(BaseModel):
    prompt: str
    agents: List[str] = []

# ========== Main Routes ==========

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the frontend HTML"""
    html_path = Path(__file__).parent.parent / 'frontend' / 'index.html'
    
    # Fallback to enhanced version if original doesn't exist
    if not html_path.exists():
        html_path = Path(__file__).parent.parent / 'frontend' / 'index_enhanced.html'
    
    try:
        with open(html_path, encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return """
        <html>
            <body>
                <h1>Kitchen Compass AI</h1>
                <p>Frontend not found. Please ensure index.html exists in the frontend directory.</p>
            </body>
        </html>
        """

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "2.0.0"
    }

# ========== Generation Endpoints ==========

@app.post('/api/generate')
async def generate(req: PromptRequest):
    """
    Generate AI response with context
    Supports both streaming and non-streaming modes
    """
    try:
        if req.stream:
            return StreamingResponse(
                stream_response(req.prompt, req.model),
                media_type="text/event-stream"
            )
        else:
            start_time = time.time()
            output = orch.generate_with_context(req.prompt, model=req.model)
            execution_time = time.time() - start_time
            
            return {
                'ok': True,
                'output': output,
                'execution_time': round(execution_time, 2),
                'model': req.model
            }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                'error': str(e),
                'message': 'Failed to generate response'
            }
        )

async def stream_response(prompt: str, model: str):
    """
    Stream response generation with agent updates
    """
    try:
        # Send initial status
        yield f"data: {json.dumps({'type': 'status', 'message': 'Starting generation...'})}\n\n"
        await asyncio.sleep(0.1)
        
        # Route query
        yield f"data: {json.dumps({'type': 'agent', 'agent': 'Intent Router', 'status': 'active'})}\n\n"
        await asyncio.sleep(0.3)
        
        # Simulate agent execution (replace with actual agent calls)
        agents = ['Compliance Guide', 'Location Risk', 'Strategic Advisor']
        for agent in agents:
            yield f"data: {json.dumps({'type': 'agent', 'agent': agent, 'status': 'active'})}\n\n"
            await asyncio.sleep(0.5)
            yield f"data: {json.dumps({'type': 'agent', 'agent': agent, 'status': 'complete'})}\n\n"
        
        # Generate response
        output = orch.generate_with_context(prompt, model=model)
        
        # Stream the response in chunks
        words = output.split()
        for i in range(0, len(words), 5):
            chunk = ' '.join(words[i:i+5])
            yield f"data: {json.dumps({'type': 'content', 'content': chunk + ' '})}\n\n"
            await asyncio.sleep(0.1)
        
        # Send completion
        yield f"data: {json.dumps({'type': 'done', 'message': 'Generation complete'})}\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

@app.post('/api/compare')
async def compare(req: PromptRequest):
    """Compare responses from different models"""
    try:
        result = orch.compare_models(req.prompt)
        return {'ok': True, **result}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                'error': str(e),
                'message': 'Failed to compare models'
            }
        )

# ========== Agent Management Endpoints ==========

@app.get('/api/agents/status')
async def get_agent_status():
    """
    Get current status and usage statistics for all agents
    """
    try:
        status = orch.get_agent_status()
        return {
            'ok': True,
            **status
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                'error': str(e),
                'message': 'Failed to retrieve agent status'
            }
        )

@app.post('/api/agents/execute')
async def execute_specific_agents(req: AgentExecutionRequest):
    """
    Execute specific agents for a given prompt
    Useful for testing individual agent capabilities
    """
    try:
        # This would be implemented to call specific agents
        # For now, return a placeholder
        return {
            'ok': True,
            'message': f'Executing agents: {req.agents}',
            'prompt': req.prompt
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                'error': str(e),
                'message': 'Failed to execute agents'
            }
        )

# ========== Data Management Endpoints ==========

@app.post('/api/ingest_csv')
async def ingest_csv(file: UploadFile = File(...)):
    """
    Ingest CSV data into vector store
    """
    try:
        import pandas as pd
        from io import BytesIO
        
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(
                status_code=400,
                detail='Only CSV files are supported'
            )
        
        # Read file content
        content = await file.read()
        
        # Convert to DataFrame
        df = pd.read_csv(BytesIO(content))
        
        # Validate DataFrame
        if df.empty:
            raise HTTPException(
                status_code=400,
                detail='CSV file is empty'
            )
        
        # Ingest the DataFrame
        result = orch.ingest_csv(df)
        
        return {
            'ok': True,
            'filename': file.filename,
            'rows_ingested': result.get('rows_ingested', 0),
            'total_chunks': result.get('total_chunks', 0),
            'columns': list(df.columns),
            'status': result.get('status', 'unknown')
        }
    except pd.errors.EmptyDataError:
        raise HTTPException(
            status_code=400,
            detail='CSV file is empty or invalid'
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                'error': str(e),
                'message': 'Failed to ingest CSV data'
            }
        )

@app.get('/api/stats')
async def stats():
    """
    Get comprehensive system statistics
    """
    try:
        stats_data = orch.get_stats()
        return {
            'ok': True,
            **stats_data
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                'error': str(e),
                'message': 'Failed to retrieve statistics'
            }
        )

@app.get('/api/vector_store/info')
async def vector_store_info():
    """
    Get information about the vector store
    """
    try:
        return {
            'ok': True,
            'total_chunks': orch.vector_store.get_collection_size(),
            'embedding_model': 'text-embedding-ada-002',
            'vector_dimensions': 1536
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                'error': str(e),
                'message': 'Failed to retrieve vector store info'
            }
        )

# ========== Analytics Endpoints ==========

@app.get('/api/analytics/overview')
async def analytics_overview():
    """
    Get comprehensive analytics overview
    """
    try:
        stats_data = orch.get_stats()
        agent_status = orch.get_agent_status()
        
        return {
            'ok': True,
            'overview': {
                'total_queries': stats_data.get('total_queries', 0),
                'total_chunks': stats_data.get('total_chunks', 0),
                'avg_execution_time': stats_data.get('avg_execution_time', 0),
                'agent_usage': stats_data.get('agent_usage', {}),
                'total_agent_executions': agent_status.get('total_executions', 0)
            },
            'agents': agent_status.get('agents', []),
            'recent_activity': stats_data.get('recent_executions', [])
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                'error': str(e),
                'message': 'Failed to generate analytics'
            }
        )

# ========== Error Handlers ==========

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            'ok': False,
            'error': 'Endpoint not found',
            'path': str(request.url)
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            'ok': False,
            'error': 'Internal server error',
            'message': 'An unexpected error occurred'
        }
    )

# ========== Startup/Shutdown Events ==========

@app.on_event("startup")
async def startup_event():
    print("🚀 Kitchen Compass AI - Enhanced Version Starting...")
    print("📡 API Endpoints:")
    print("   - GET  /              (Frontend)")
    print("   - POST /api/generate  (Generate response)")
    print("   - GET  /api/agents/status (Agent status)")
    print("   - GET  /api/stats     (System statistics)")
    print("   - POST /api/ingest_csv (Upload data)")
    print("   - GET  /docs          (API Documentation)")
    print("✅ Server ready!")

@app.on_event("shutdown")
async def shutdown_event():
    print("👋 Kitchen Compass AI shutting down...")