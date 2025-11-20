from backend.agents.base_agent import BaseAgent

class LocationRiskAgent(BaseAgent):
    def __init__(self, openai_client, vector_store):
        super().__init__("Location Risk Analyzer", "Analyzes location-based risks for restaurants")
        self.openai_client = openai_client
        self.vector_store = vector_store
    
    def process(self, query: str, context: str = "", **kwargs):
        """Process location risk query"""
        return self.format_response("Location risk analysis functionality coming soon...")
    
    def get_prompt(self, query: str, context: str = "") -> str:
        """Generate location risk prompt"""
        return f"Analyze location risks for: {query}"
