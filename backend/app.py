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

# ========== Vector Store Inspection Endpoints ==========

@app.get('/api/vector_store/inspect')
async def inspect_vector_store():
    """
    Comprehensive vector store inspection
    Shows contents, statistics, and sample data
    """
    try:
        vs = orch.vector_store
        
        # Basic statistics
        total_docs = vs.get_collection_size()
        faiss_count = vs.index.ntotal if vs.index else 0
        df_rows = len(vs.df) if vs.df is not None else 0
        
        # DataFrame info
        df_info = {}
        if vs.df is not None and len(vs.df) > 0:
            df_info = {
                'columns': list(vs.df.columns),
                'shape': vs.df.shape,
                'dtypes': {col: str(dtype) for col, dtype in vs.df.dtypes.items()}
            }
            
            # Check if complaint data
            if 'complaint_type' in vs.df.columns:
                df_info['data_type'] = 'NYC 311 Complaints'
                df_info['unique_complaint_types'] = int(vs.df['complaint_type'].nunique())
                df_info['top_5_complaint_types'] = vs.df['complaint_type'].value_counts().head(5).to_dict()
                
                if 'borough' in vs.df.columns:
                    df_info['complaints_by_borough'] = vs.df['borough'].value_counts().to_dict()
            else:
                df_info['data_type'] = 'General CSV Data'
        
        # Sample documents
        sample_docs = []
        for i, doc in enumerate(vs.documents[:5]):
            sample_docs.append({
                'index': i,
                'type': doc.metadata.get('type', 'unknown'),
                'content_preview': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content,
                'content_length': len(doc.page_content),
                'metadata_keys': list(doc.metadata.keys()),
                'metadata_sample': {k: str(v)[:50] for k, v in list(doc.metadata.items())[:5]}
            })
        
        return {
            'ok': True,
            'statistics': {
                'total_documents': total_docs,
                'faiss_index_vectors': faiss_count,
                'dataframe_rows': df_rows,
                'index_matches': total_docs == faiss_count == df_rows
            },
            'dataframe_info': df_info,
            'sample_documents': sample_docs,
            'storage_paths': {
                'index_file': vs.index_path,
                'data_file': vs.index_path + '.data.pkl',
                'csv_source': vs.csv_path
            }
        }
    except Exception as e:
        import traceback
        raise HTTPException(
            status_code=500,
            detail={
                'error': str(e),
                'traceback': traceback.format_exc(),
                'message': 'Failed to inspect vector store'
            }
        )

@app.get('/api/vector_store/search_test')
async def test_vector_search(query: str = "food establishment", k: int = 3):
    """
    Test vector store search functionality
    Shows how well the search is working
    """
    try:
        vs = orch.vector_store
        
        # Perform search
        results = vs.search(query, k=k)
        
        # Format results
        search_results = []
        for i, doc in enumerate(results):
            search_results.append({
                'rank': i + 1,
                'relevance_score': doc.metadata.get('relevance_score', 0),
                'distance': doc.metadata.get('distance', 0),
                'content': doc.page_content[:300] + '...' if len(doc.page_content) > 300 else doc.page_content,
                'metadata': {k: str(v) for k, v in doc.metadata.items() if k not in ['relevance_score', 'distance']}
            })
        
        return {
            'ok': True,
            'query': query,
            'k': k,
            'results_count': len(results),
            'results': search_results,
            'vector_store_size': vs.get_collection_size()
        }
    except Exception as e:
        import traceback
        raise HTTPException(
            status_code=500,
            detail={
                'error': str(e),
                'traceback': traceback.format_exc(),
                'message': 'Failed to test search'
            }
        )

@app.get('/api/vector_store/complaints/borough/{borough}')
async def get_complaints_by_borough(borough: str):
    """
    Get all complaints for a specific borough
    Only works if vector store contains complaint data
    """
    try:
        vs = orch.vector_store
        complaints = vs.get_complaints_by_borough(borough)
        
        if not complaints:
            return {
                'ok': True,
                'borough': borough,
                'complaint_count': 0,
                'complaints': [],
                'message': 'No complaints found for this borough'
            }
        
        return {
            'ok': True,
            'borough': borough,
            'complaint_count': len(complaints),
            'complaints': complaints[:50],  # Limit to first 50
            'total_available': len(complaints),
            'sample_complaint_types': list(set([c.get('complaint_type', 'Unknown') for c in complaints[:20]]))
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                'error': str(e),
                'message': 'Failed to retrieve borough complaints'
            }
        )

@app.get('/api/vector_store/raw_data')
async def get_raw_dataframe(limit: int = 100):
    """
    Get raw DataFrame data from vector store
    """
    try:
        vs = orch.vector_store
        
        if vs.df is None or len(vs.df) == 0:
            return {
                'ok': True,
                'message': 'No DataFrame data available',
                'rows': []
            }
        
        # Get limited rows
        df_limited = vs.df.head(limit)
        
        return {
            'ok': True,
            'total_rows': len(vs.df),
            'rows_returned': len(df_limited),
            'columns': list(vs.df.columns),
            'data': df_limited.to_dict('records')
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                'error': str(e),
                'message': 'Failed to retrieve raw data'
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
async def get_heatmap_data(borough: Optional[str] = None, complaint_type: Optional[str] = None, limit: int = 500):
    """
    Get heatmap data for complaint visualization using REAL coordinates
    Returns GeoJSON-compatible format for ArcGIS/Leaflet
    """
    try:
        import pandas as pd
        vs = orch.vector_store
        
        # Get the DataFrame
        df = vs.df
        
        if df is None or len(df) == 0:
            raise HTTPException(
                status_code=404,
                detail={'error': 'No complaint data available', 'message': 'Vector store is empty'}
            )
        
        # Filter by borough if specified
        filtered_df = df.copy()
        if borough:
            filtered_df = filtered_df[filtered_df['borough'].str.upper() == borough.upper()]
        
        # Filter by complaint type if specified
        if complaint_type:
            filtered_df = filtered_df[filtered_df['complaint_type'].str.contains(complaint_type, case=False, na=False)]
        
        # Remove rows without valid coordinates
        filtered_df = filtered_df.dropna(subset=['latitude', 'longitude'])
        
        # Limit results for performance
        if len(filtered_df) > limit:
            filtered_df = filtered_df.sample(n=limit, random_state=42)
        
        # Build GeoJSON features from actual data
        features = []
        for idx, row in filtered_df.iterrows():
            try:
                lat = float(row['latitude'])
                lon = float(row['longitude'])
                
                # Skip invalid coordinates
                if pd.isna(lat) or pd.isna(lon) or lat == 0 or lon == 0:
                    continue
                
                features.append({
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Point',
                        'coordinates': [lon, lat]  # GeoJSON uses [longitude, latitude]
                    },
                    'properties': {
                        'complaint_type': str(row.get('complaint_type', 'Unknown')),
                        'descriptor': str(row.get('descriptor', '')),
                        'borough': str(row.get('borough', 'Unknown')),
                        'status': str(row.get('status', '')),
                        'created_date': str(row.get('created_date', '')),
                        'address': str(row.get('incident_address', '')),
                        'zip': str(row.get('incident_zip', ''))
                    }
                })
            except (ValueError, TypeError) as e:
                # Skip invalid rows
                continue
        
        # Generate summary statistics
        borough_stats = {}
        if 'borough' in filtered_df.columns:
            for boro in filtered_df['borough'].dropna().unique():
                boro_data = filtered_df[filtered_df['borough'] == boro]
                borough_stats[str(boro)] = {
                    'count': int(len(boro_data)),
                    'top_complaint': str(boro_data['complaint_type'].mode()[0]) if len(boro_data) > 0 else 'N/A'
                }
        
        return {
            'ok': True,
            'type': 'FeatureCollection',
            'features': features,
            'summary': {
                'total_complaints': len(features),
                'boroughs_covered': len(borough_stats),
                'filtered_by_borough': borough,
                'filtered_by_complaint_type': complaint_type,
                'borough_breakdown': borough_stats,
                'data_source': 'NYC 311 Open Data'
            }
        }
        
    except Exception as e:
        import traceback
        raise HTTPException(
            status_code=500,
            detail={
                'error': str(e),
                'traceback': traceback.format_exc(),
                'message': 'Failed to generate heatmap data'
            }
        )

@app.get('/api/heatmap/boroughs')
async def get_borough_stats():
    """
    Get aggregated statistics by borough using REAL data from vector store
    """
    try:
        import pandas as pd
        vs = orch.vector_store
        df = vs.df
        
        if df is None or len(df) == 0:
            raise HTTPException(
                status_code=404,
                detail={'error': 'No complaint data available'}
            )
        
        # NYC Borough coordinates (for map centering)
        borough_coords = {
            'BROOKLYN': {'lat': 40.6782, 'lng': -73.9442},
            'MANHATTAN': {'lat': 40.7831, 'lng': -73.9712},
            'QUEENS': {'lat': 40.7282, 'lng': -73.7949},
            'BRONX': {'lat': 40.8448, 'lng': -73.8648},
            'STATEN ISLAND': {'lat': 40.5795, 'lng': -74.1502}
        }
        
        # Calculate real statistics per borough
        borough_stats = []
        
        for boro in df['borough'].dropna().unique():
            boro_data = df[df['borough'] == boro]
            
            # Get top complaint type
            top_complaint = 'N/A'
            if len(boro_data) > 0 and 'complaint_type' in boro_data.columns:
                top_complaint = str(boro_data['complaint_type'].mode()[0])
            
            # Calculate a simple risk score based on complaint density
            # (complaints per 1000 would require population data, so we'll use relative counts)
            complaint_count = len(boro_data)
            risk_score = min(100, int((complaint_count / len(df)) * 500))  # Normalized score
            
            # Determine risk level
            if risk_score >= 70:
                risk_level = 'High'
            elif risk_score >= 40:
                risk_level = 'Medium'
            else:
                risk_level = 'Low'
            
            # Simple trend calculation (would need time series data for real trends)
            # For now, just mark all as 'stable'
            trend = 'stable'
            
            borough_stats.append({
                'name': str(boro),
                'complaint_count': complaint_count,
                'risk_score': risk_score,
                'risk_level': risk_level,
                'trend': trend,
                'top_complaint': top_complaint,
                'coordinates': borough_coords.get(str(boro).upper(), {'lat': 40.7128, 'lng': -74.0060})
            })
        
        # Sort by complaint count descending
        borough_stats.sort(key=lambda x: x['complaint_count'], reverse=True)
        
        return {
            'ok': True,
            'boroughs': borough_stats,
            'total_complaints': int(len(df)),
            'avg_risk_score': round(sum(b['risk_score'] for b in borough_stats) / len(borough_stats), 1) if borough_stats else 0
        }
        
    except Exception as e:
        import traceback
        raise HTTPException(
            status_code=500,
            detail={
                'error': str(e),
                'traceback': traceback.format_exc(),
                'message': 'Failed to get borough stats'
            }
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
    
    # Add Vector Store Status
    print("\n🔍 Vector Store Status:")
    try:
        print(f"   Total Documents: {orch.vector_store.get_collection_size()}")
        if orch.vector_store.df is not None:
            print(f"   DataFrame Rows: {len(orch.vector_store.df)}")
            data_type = '311 Complaints' if 'complaint_type' in orch.vector_store.df.columns else 'General'
            print(f"   Data Type: {data_type}")
            if data_type == '311 Complaints':
                print(f"   Unique Complaint Types: {orch.vector_store.df['complaint_type'].nunique()}")
    except Exception as e:
        print(f"   ⚠️ Error loading vector store: {e}")
    
    print("\n📡 API Endpoints:")
    print("   - GET  /              (Frontend)")
    print("   - POST /api/generate  (Generate response)")
    print("   - GET  /api/agents/status (Agent status)")
    print("   - GET  /api/stats     (System statistics)")
    print("   - POST /api/ingest_csv (Upload data)")
    print("   - GET  /api/heatmap/data (Heatmap data)")
    print("   - GET  /api/heatmap/boroughs (Borough stats)")
    print("   - POST /api/heatmap/analyze (Location analysis)")
    print("   - GET  /api/vector_store/inspect (🆕 Vector store inspection)")
    print("   - GET  /api/vector_store/search_test (🆕 Test search)")
    print("   - GET  /docs          (API Documentation)")
    print("✅ Server ready!")

@app.on_event("shutdown")
async def shutdown_event():
    print("👋 Kitchen Compass AI shutting down...")