"""
Enhanced Orchestrator for Kitchen Compass AI
Supports multi-agent tracking, streaming responses, and detailed execution flow
"""

import os
import time
from typing import Dict, List, Optional, Any
from openai import OpenAI
from backend.vector_store import VectorStore
from backend.agents.intent_router_agent import IntentRouterAgent
from backend.agents.complaint_analyzer_agent import ComplaintAnalyzerAgent
from backend.agents.compliance_guide_agent import ComplianceGuideAgent
from backend.agents.location_risk_agent import LocationRiskAgent
from backend.agents.strategic_advisor_agent import StrategicAdvisorAgent
from backend.agents.violation_prevention_agent import ViolationPreventionAgent

class Orchestrator:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.vector_store = VectorStore(csv_path="data/erm2-nwe9.csv")
        
        # Initialize all agents
        self.intent_router = IntentRouterAgent(self.openai_client)
        self.complaint_analyzer = ComplaintAnalyzerAgent(self.openai_client, self.vector_store)
        self.compliance_guide = ComplianceGuideAgent(self.openai_client, self.vector_store)
        self.location_risk = LocationRiskAgent(self.openai_client, self.vector_store)
        self.strategic_advisor = StrategicAdvisorAgent(self.openai_client, self.vector_store)
        self.violation_prevention = ViolationPreventionAgent(self.openai_client, self.vector_store)
        
        # Agent mapping
        self.agents = {
            'intent_router': self.intent_router,
            'complaint_analyzer': self.complaint_analyzer,
            'compliance_guide': self.compliance_guide,
            'location_risk': self.location_risk,
            'strategic_advisor': self.strategic_advisor,
            'violation_prevention': self.violation_prevention
        }
        
        # Execution tracking
        self.execution_history = []
        self.agent_usage_count = {}
    
    def generate_with_context(self, prompt: str, model: str = "gpt-4o-mini") -> str:
        """
        Generate response with context from vector store
        Returns formatted response with agent tracking
        """
        try:
            start_time = time.time()
            
            # Step 1: Route query to appropriate agents
            routing_result = self.intent_router.route(prompt)
            selected_agents = routing_result.get('agents', ['compliance_guide'])
            execution_plan = routing_result.get('execution_plan', 'sequential')
            
            # Track agent usage
            for agent_name in selected_agents:
                self.agent_usage_count[agent_name] = self.agent_usage_count.get(agent_name, 0) + 1
            
            # Step 2: Retrieve relevant context from vector store
            context_docs = self.vector_store.search(prompt, k=5)
            context = "\n\n".join([doc.page_content for doc in context_docs])
            
            # Step 3: Execute agents based on routing plan
            agent_responses = []
            for agent_name in selected_agents:
                agent = self.agents.get(agent_name)
                if agent:
                    try:
                        response = agent.process(prompt, context)
                        agent_responses.append({
                            'agent': agent_name,
                            'response': response,
                            'status': 'success'
                        })
                    except Exception as e:
                        agent_responses.append({
                            'agent': agent_name,
                            'response': f"Error: {str(e)}",
                            'status': 'error'
                        })
            
            # Step 4: Synthesize final response
            final_response = self._synthesize_responses(prompt, agent_responses, context)
            
            # Step 5: Track execution
            execution_time = time.time() - start_time
            self.execution_history.append({
                'prompt': prompt,
                'agents_used': selected_agents,
                'execution_time': execution_time,
                'timestamp': time.time(),
                'model': model
            })
            
            return final_response
            
        except Exception as e:
            return f"Error processing request: {str(e)}"
    
    def _synthesize_responses(self, prompt: str, agent_responses: List[Dict], context: str) -> str:
        """
        Synthesize multiple agent responses into a coherent answer
        """
        if not agent_responses:
            return "No agent responses available."
        
        # If only one agent, return its response directly
        if len(agent_responses) == 1:
            return agent_responses[0]['response']
        
        # Combine multiple agent responses
        synthesis_prompt = f"""
You are synthesizing insights from multiple specialized agents to answer this question:

**User Question:** {prompt}

**Agent Responses:**
"""
        for i, agent_resp in enumerate(agent_responses, 1):
            agent_name = agent_resp['agent'].replace('_', ' ').title()
            synthesis_prompt += f"\n{i}. **{agent_name}:**\n{agent_resp['response']}\n"
        
        synthesis_prompt += """
**Context from NYC Data:**
""" + context[:1000] + """

Please synthesize these responses into a comprehensive, well-structured answer. 
Include relevant details from each agent's perspective, and cite specific data points when available.
Format your response with clear sections using markdown.
"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that synthesizes information from multiple sources."},
                    {"role": "user", "content": synthesis_prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            # Fallback to simple concatenation
            combined = "\n\n---\n\n".join([f"**{r['agent'].replace('_', ' ').title()}:**\n{r['response']}" for r in agent_responses])
            return combined
    
    def get_agent_status(self) -> Dict[str, Any]:
        """
        Get current status of all agents
        """
        return {
            'agents': [
                {
                    'name': 'Intent Router',
                    'id': 'intent_router',
                    'status': 'idle',
                    'usage_count': self.agent_usage_count.get('intent_router', 0)
                },
                {
                    'name': 'Complaint Analyzer',
                    'id': 'complaint_analyzer',
                    'status': 'idle',
                    'usage_count': self.agent_usage_count.get('complaint_analyzer', 0)
                },
                {
                    'name': 'Compliance Guide',
                    'id': 'compliance_guide',
                    'status': 'idle',
                    'usage_count': self.agent_usage_count.get('compliance_guide', 0)
                },
                {
                    'name': 'Location Risk',
                    'id': 'location_risk',
                    'status': 'idle',
                    'usage_count': self.agent_usage_count.get('location_risk', 0)
                },
                {
                    'name': 'Strategic Advisor',
                    'id': 'strategic_advisor',
                    'status': 'idle',
                    'usage_count': self.agent_usage_count.get('strategic_advisor', 0)
                },
                {
                    'name': 'Violation Prevention',
                    'id': 'violation_prevention',
                    'status': 'idle',
                    'usage_count': self.agent_usage_count.get('violation_prevention', 0)
                }
            ],
            'total_executions': len(self.execution_history)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive orchestrator statistics
        """
        total_chunks = self.vector_store.get_collection_size()
        
        avg_execution_time = 0
        if self.execution_history:
            avg_execution_time = sum(e['execution_time'] for e in self.execution_history) / len(self.execution_history)
        
        return {
            'total_chunks': total_chunks,
            'total_queries': len(self.execution_history),
            'agent_usage': self.agent_usage_count,
            'avg_execution_time': round(avg_execution_time, 2),
            'recent_executions': self.execution_history[-10:] if self.execution_history else []
        }
    
    def ingest_csv(self, df) -> Dict[str, Any]:
        """
        Ingest CSV data into vector store
        """
        try:
            result = self.vector_store.ingest_dataframe(df)
            return {
                'rows_ingested': len(df),
                'total_chunks': self.vector_store.get_collection_size(),
                'status': 'success'
            }
        except Exception as e:
            return {
                'rows_ingested': 0,
                'total_chunks': 0,
                'status': 'error',
                'error': str(e)
            }
    
    def compare_models(self, prompt: str) -> Dict[str, Any]:
        """
        Compare responses from different models
        """
        gpt_response = self.generate_with_context(prompt, model="gpt-4o-mini")
        
        # For Ollama, you'd call your ollama_client here
        # ollama_response = self.ollama_client.generate(prompt)
        ollama_response = "Ollama integration coming soon..."
        
        return {
            'gpt_response': gpt_response,
            'ollama_response': ollama_response
        }