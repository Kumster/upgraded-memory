from backend.agents.base_agent import BaseAgent

class StrategicAdvisorAgent(BaseAgent):
    def __init__(self, openai_client, vector_store):
        super().__init__("Strategic Advisor", "Provides strategic business advice for restaurants")
        self.openai_client = openai_client
        self.vector_store = vector_store
    
    def process(self, query: str, context: str = "", **kwargs):
        """Process strategic advisory query"""
        return self.format_response("Strategic advisory functionality coming soon...")
    
    def get_prompt(self, query: str, context: str = "") -> str:
        """Generate strategic advisory prompt"""
        return f"Provide strategic advice for: {query}"
