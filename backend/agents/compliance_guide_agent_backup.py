from backend.agents.base_agent import BaseAgent

class ComplianceGuideAgent(BaseAgent):
    def __init__(self, openai_client, vector_store):
        super().__init__("Compliance Guide", "Provides NYC restaurant compliance guidance")
        self.openai_client = openai_client
        self.vector_store = vector_store
    
    def process(self, query: str, context: str = "", **kwargs):
        """Process compliance query"""
        return self.format_response("Compliance guidance functionality coming soon...")
    
    def get_prompt(self, query: str, context: str = "") -> str:
        """Generate compliance prompt"""
        return f"Provide compliance guidance for: {query}"
