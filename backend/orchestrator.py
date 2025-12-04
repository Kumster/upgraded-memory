"""
Enhanced Orchestrator for Kitchen Compass AI
Supports multi-agent tracking, streaming responses, and detailed execution flow
"""

import os
import re
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
            
            # Step 2: Retrieve relevant context from vector store with metadata
            context_docs = self.vector_store.search(prompt, k=8)
            
            # Extract context with source tracking
            context_with_sources = []
            for i, doc in enumerate(context_docs):
                source_info = {
                    'id': i + 1,
                    'content': doc.page_content,
                    'metadata': getattr(doc, 'metadata', {})
                }
                context_with_sources.append(source_info)
            
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
            
            # Step 4: Synthesize final response with improved formatting
            final_response = self._synthesize_responses(
                prompt, 
                agent_responses, 
                context_with_sources,
                model
            )
            
            # Step 5: Clean up the output
            final_response = self._clean_output(final_response)
            
            # Step 6: Track execution
            execution_time = time.time() - start_time
            self.execution_history.append({
                'prompt': prompt,
                'agents_used': selected_agents,
                'execution_time': execution_time,
                'timestamp': time.time(),
                'model': model,
                'sources_used': len(context_docs)
            })
            
            return final_response
            
        except Exception as e:
            return f"Error processing request: {str(e)}"
    
    def _clean_output(self, text: str) -> str:
        """Clean up AI output for better display"""
        # Remove markdown headers (##, ###)
        text = re.sub(r'^#{1,3}\s+', '', text, flags=re.MULTILINE)
        
        # Remove ** around section titles that are on their own line
        text = re.sub(r'^\*\*([^*]+)\*\*$', r'\1', text, flags=re.MULTILINE)
        
        # Limit consecutive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove excessive dashes/separators
        text = re.sub(r'^-{3,}$', '', text, flags=re.MULTILINE)
        
        # Trim whitespace
        text = text.strip()
        
        return text
    
    def _synthesize_responses(
        self, 
        prompt: str, 
        agent_responses: List[Dict], 
        context_sources: List[Dict],
        model: str = "gpt-4o-mini"
    ) -> str:
        """
        Synthesize multiple agent responses into a coherent, well-structured answer
        """
        if not agent_responses:
            return "No agent responses available."
        
        # If only one agent, clean and return its response
        if len(agent_responses) == 1:
            return self._clean_output(agent_responses[0]['response'])
        
        # Build context summary
        context_summary = ""
        for source in context_sources[:5]:
            metadata = source.get('metadata', {})
            borough = metadata.get('borough', 'NYC')
            complaint_type = metadata.get('complaint_type', 'General')
            context_summary += f"• {borough}: {complaint_type} - {source['content'][:150]}...\n"
        
        # Collect agent insights
        agent_insights = ""
        for agent_resp in agent_responses:
            agent_name = agent_resp['agent'].replace('_', ' ').title()
            # Truncate long responses
            response_text = agent_resp['response'][:800] if len(agent_resp['response']) > 800 else agent_resp['response']
            agent_insights += f"{agent_name}:\n{response_text}\n\n"
        
        # Concise synthesis prompt
        synthesis_prompt = f"""You are Kitchen Compass AI - a friendly NYC restaurant consultant.

USER QUESTION: {prompt}

AGENT INSIGHTS:
{agent_insights}

DATA CONTEXT:
{context_summary}

RESPOND WITH:
1. A direct 2-3 sentence answer first
2. 3-5 key bullet points with specific data (numbers, boroughs, percentages)
3. One practical "Pro Tip" at the end

RULES:
- NO headers or titles (no ##, ###, **)
- NO "Executive Summary" or "Key Findings" labels
- Keep total response under 250 words
- Be conversational, not formal
- Use bullet points (•) sparingly
- Include specific data when available
- End with: 💡 Pro Tip: [actionable advice]

Write your response now:"""

        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful NYC restaurant consultant. Give concise, practical advice. Never use markdown headers. Be conversational."
                    },
                    {"role": "user", "content": synthesis_prompt}
                ],
                temperature=0.7,
                max_tokens=600
            )
            return response.choices[0].message.content
        except Exception as e:
            # Fallback to simple concatenation
            combined = "\n\n".join([
                f"{r['agent'].replace('_', ' ').title()}: {r['response'][:300]}..." 
                for r in agent_responses
            ])
            return combined
    
    def route_query(self, prompt: str) -> List[str]:
        """
        Route query and return list of agent names that will be used
        Useful for streaming UI updates
        """
        try:
            routing_result = self.intent_router.route(prompt)
            return routing_result.get('agents', ['compliance_guide'])
        except Exception:
            return ['compliance_guide']
    
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
                    'usage_count': self.agent_usage_count.get('intent_router', 0),
                    'description': 'Routes queries to appropriate specialist agents'
                },
                {
                    'name': 'Complaint Analyzer',
                    'id': 'complaint_analyzer',
                    'status': 'idle',
                    'usage_count': self.agent_usage_count.get('complaint_analyzer', 0),
                    'description': 'Analyzes NYC 311 complaint patterns'
                },
                {
                    'name': 'Compliance Guide',
                    'id': 'compliance_guide',
                    'status': 'idle',
                    'usage_count': self.agent_usage_count.get('compliance_guide', 0),
                    'description': 'Health code & permit guidance'
                },
                {
                    'name': 'Location Risk',
                    'id': 'location_risk',
                    'status': 'idle',
                    'usage_count': self.agent_usage_count.get('location_risk', 0),
                    'description': 'Neighborhood risk assessment'
                },
                {
                    'name': 'Strategic Advisor',
                    'id': 'strategic_advisor',
                    'status': 'idle',
                    'usage_count': self.agent_usage_count.get('strategic_advisor', 0),
                    'description': 'Business strategy insights'
                },
                {
                    'name': 'Violation Prevention',
                    'id': 'violation_prevention',
                    'status': 'idle',
                    'usage_count': self.agent_usage_count.get('violation_prevention', 0),
                    'description': 'Proactive violation prevention'
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
        
        # Calculate most used agents
        most_used_agent = None
        if self.agent_usage_count:
            most_used_agent = max(self.agent_usage_count, key=self.agent_usage_count.get)
        
        return {
            'total_chunks': total_chunks,
            'total_queries': len(self.execution_history),
            'agent_usage': self.agent_usage_count,
            'most_used_agent': most_used_agent,
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
        ollama_response = "Ollama integration coming soon..."
        
        return {
            'gpt_response': gpt_response,
            'ollama_response': ollama_response
        }