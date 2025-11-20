"""
311 Complaint Analyzer Agent
"""

from backend.agents.base_agent import BaseAgent
from typing import Dict, Any, Optional

class ComplaintAnalyzerAgent(BaseAgent):
    """Analyzes 311 complaint data"""
    
    def __init__(self, openai_client, vector_store):
        super().__init__("Complaint Analyzer", "Analyzes NYC 311 complaint data")
        self.openai_client = openai_client
        self.vector_store = vector_store
    
    def process(self, query: str, context: str = "", **kwargs) -> Dict:
        results = self.vector_store.search(query, k=5)
        response_text = f"Found {len(results)} relevant complaints for: {query}"
        return self.format_response(response_text, {'results_count': len(results)})
    
    def get_prompt(self, query: str, context: str = "") -> str:
        return f"Analyze 311 complaints for: {query}"