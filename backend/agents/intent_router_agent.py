from backend.agents.base_agent import BaseAgent

class IntentRouterAgent(BaseAgent):
    def __init__(self, openai_client):
        super().__init__("Intent Router", "Routes queries to appropriate specialized agents")
        self.openai_client = openai_client
    
    def route(self, query: str):
        """Route query to appropriate agents"""
        # Simple routing logic - default to complaint_analyzer
        return {
            'agents': ['complaint_analyzer'],
            'execution_plan': 'sequential'
        }
    
    def process(self, query: str, context: str = "", **kwargs):
        """Process routing request"""
        routing = self.route(query)
        return self.format_response("Intent routing completed", {'routing': routing})
    
    def get_prompt(self, query: str, context: str = "") -> str:
        """Generate routing prompt"""
        return f"Route this query to appropriate agents: {query}"
