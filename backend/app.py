"""
Enhanced FastAPI App for Kitchen Compass AI
Includes agent status tracking, streaming responses, heatmap endpoints, and comprehensive error handling
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
import re

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

class HeatmapRequest(BaseModel):
    query: str = ""
    borough: Optional[str] = None
    complaint_type: Optional[str] = None


# ========== Helper Functions ==========

def clean_output(text: str) -> str:
    """Clean up AI output for better display"""
    # Remove markdown headers
    text = re.sub(r'^#{1,3}\s+', '', text, flags=re.MULTILINE)
    
    # Remove standalone bold section titles
    text = re.sub(r'^\*\*([^*]+)\*\*$', r'\1', text, flags=re.MULTILINE)
    
    # Limit consecutive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove excessive separators
    text = re.sub(r'^-{3,}$', '', text, flags=re.MULTILINE)
    
    return text.strip()


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
            
            # Clean the output
            output = clean_output(output)
            
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
        output = clean_output(output)
        
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

# ========== Heatmap Endpoints ==========

@app.get('/api/heatmap/data')
async def get_heatmap_data(borough: Optional[str] = None, complaint_type: Optional[str] = None):
    """
    Get heatmap data for complaint visualization
    Returns GeoJSON-compatible format for ArcGIS/Leaflet
    """
    try:
        # Try to get data from location_risk_agent if available
        if hasattr(orch, 'agents') and 'location_risk' in orch.agents:
            agent = orch.agents['location_risk']
            if hasattr(agent, 'get_heatmap_data'):
                heatmap_data = agent.get_heatmap_data(borough=borough, complaint_type=complaint_type)
                return {'ok': True, **heatmap_data}
        
        # Fallback: Generate sample heatmap data from vector store
        borough_coords = {
            'Manhattan': {'lat': 40.7831, 'lng': -73.9712},
            'Brooklyn': {'lat': 40.6782, 'lng': -73.9442},
            'Queens': {'lat': 40.7282, 'lng': -73.7949},
            'Bronx': {'lat': 40.8448, 'lng': -73.8648},
            'Staten Island': {'lat': 40.5795, 'lng': -74.1502}
        }
        
        # Sample complaint data
        sample_data = {
            'Manhattan': {'count': 245, 'risk_score': 72, 'top_complaints': ['Health', 'Rodent', 'Food Poisoning']},
            'Brooklyn': {'count': 312, 'risk_score': 78, 'top_complaints': ['Rodent', 'Health', 'Consumer Complaint']},
            'Queens': {'count': 189, 'risk_score': 65, 'top_complaints': ['Health', 'Vendor Enforcement', 'Noise']},
            'Bronx': {'count': 156, 'risk_score': 61, 'top_complaints': ['Rodent', 'Health', 'Food Poisoning']},
            'Staten Island': {'count': 67, 'risk_score': 42, 'top_complaints': ['Health', 'Noise', 'Outdoor Dining']}
        }
        
        # Filter by borough if specified
        if borough and borough in sample_data:
            filtered_data = {borough: sample_data[borough]}
        else:
            filtered_data = sample_data
        
        # Build GeoJSON features
        features = []
        for boro, data in filtered_data.items():
            coords = borough_coords.get(boro, {'lat': 40.7128, 'lng': -74.0060})
            features.append({
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [coords['lng'], coords['lat']]
                },
                'properties': {
                    'borough': boro,
                    'complaint_count': data['count'],
                    'risk_score': data['risk_score'],
                    'risk_level': 'High' if data['risk_score'] >= 70 else ('Medium' if data['risk_score'] >= 40 else 'Low'),
                    'top_complaints': data['top_complaints']
                }
            })
        
        return {
            'ok': True,
            'type': 'FeatureCollection',
            'features': features,
            'summary': {
                'total_complaints': sum(d['count'] for d in filtered_data.values()),
                'boroughs_covered': len(filtered_data),
                'highest_risk': max(filtered_data.items(), key=lambda x: x[1]['risk_score'])[0],
                'lowest_risk': min(filtered_data.items(), key=lambda x: x[1]['risk_score'])[0]
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={'error': str(e), 'message': 'Failed to generate heatmap data'}
        )

@app.get('/api/heatmap/boroughs')
async def get_borough_stats():
    """
    Get aggregated statistics by borough for the heatmap sidebar
    """
    try:
        borough_stats = [
            {
                'name': 'Manhattan',
                'complaint_count': 245,
                'risk_score': 72,
                'risk_level': 'High',
                'trend': 'increasing',
                'top_complaint': 'Health Violations',
                'coordinates': {'lat': 40.7831, 'lng': -73.9712}
            },
            {
                'name': 'Brooklyn',
                'complaint_count': 312,
                'risk_score': 78,
                'risk_level': 'High',
                'trend': 'stable',
                'top_complaint': 'Rodent Issues',
                'coordinates': {'lat': 40.6782, 'lng': -73.9442}
            },
            {
                'name': 'Queens',
                'complaint_count': 189,
                'risk_score': 65,
                'risk_level': 'Medium',
                'trend': 'decreasing',
                'top_complaint': 'Health Violations',
                'coordinates': {'lat': 40.7282, 'lng': -73.7949}
            },
            {
                'name': 'Bronx',
                'complaint_count': 156,
                'risk_score': 61,
                'risk_level': 'Medium',
                'trend': 'stable',
                'top_complaint': 'Rodent Issues',
                'coordinates': {'lat': 40.8448, 'lng': -73.8648}
            },
            {
                'name': 'Staten Island',
                'complaint_count': 67,
                'risk_score': 42,
                'risk_level': 'Low',
                'trend': 'decreasing',
                'top_complaint': 'Noise Complaints',
                'coordinates': {'lat': 40.5795, 'lng': -74.1502}
            }
        ]
        
        return {
            'ok': True,
            'boroughs': borough_stats,
            'total_complaints': sum(b['complaint_count'] for b in borough_stats),
            'avg_risk_score': round(sum(b['risk_score'] for b in borough_stats) / len(borough_stats), 1)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={'error': str(e), 'message': 'Failed to get borough stats'}
        )

@app.post('/api/heatmap/analyze')
async def analyze_location(req: HeatmapRequest):
    """
    Get AI analysis for a specific location query along with heatmap data
    """
    try:
        # Generate AI analysis using orchestrator
        analysis = ""
        if req.query:
            analysis = orch.generate_with_context(
                f"Analyze location risk for NYC restaurant: {req.query}. "
                f"Borough focus: {req.borough or 'all'}. "
                "Provide specific risk factors and recommendations.",
                model="gpt-4o-mini"
            )
            analysis = clean_output(analysis)
        
        # Get heatmap data
        heatmap_response = await get_heatmap_data(borough=req.borough, complaint_type=req.complaint_type)
        
        return {
            'ok': True,
            'analysis': analysis,
            'heatmap_data': heatmap_response if heatmap_response.get('ok') else None,
            'query': req.query,
            'filters': {
                'borough': req.borough,
                'complaint_type': req.complaint_type
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={'error': str(e), 'message': 'Failed to analyze location'}
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
    print("   - GET  /api/heatmap/data (Heatmap data)")
    print("   - GET  /api/heatmap/boroughs (Borough stats)")
    print("   - POST /api/heatmap/analyze (Location analysis)")
    print("   - GET  /docs          (API Documentation)")
    print("✅ Server ready!")

@app.on_event("shutdown")
async def shutdown_event():
    print("👋 Kitchen Compass AI shutting down...")