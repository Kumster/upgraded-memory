from backend.agents.base_agent import BaseAgent

class ViolationPreventionAgent(BaseAgent):
    def __init__(self, openai_client, vector_store):
        super().__init__("Violation Prevention", "Helps prevent health code violations")
        self.openai_client = openai_client
        self.vector_store = vector_store
    
    def process(self, query: str, context: str = "", **kwargs):
        """Process violation prevention query"""
        return self.format_response("Violation prevention guidance functionality coming soon...")
    
    def get_prompt(self, query: str, context: str = "") -> str:
        """Generate violation prevention prompt"""
        return f"Provide violation prevention guidance for: {query}"
