"""
Compliance Guide Agent
Provides NYC restaurant compliance guidance
"""

from backend.agents.base_agent import BaseAgent
from typing import Dict, Any

class ComplianceGuideAgent(BaseAgent):
    """Provides NYC restaurant compliance guidance"""
    
    def __init__(self, openai_client, vector_store):
        super().__init__("Compliance Guide", "Provides NYC restaurant compliance guidance")
        self.openai_client = openai_client
        self.vector_store = vector_store
    
    def process(self, query: str, context: str = "", **kwargs) -> str:
        """
        Process compliance query
        Returns AI-generated compliance guidance
        """
        try:
            # Search for relevant context
            results = self.vector_store.search(query, k=5)
            
            # Build context from search results
            if results:
                data_context = "\n\n".join([
                    f"Reference {i+1}: {doc.page_content}" 
                    for i, doc in enumerate(results)
                ])
            else:
                data_context = "No specific data available, providing general guidance."
            
            # Generate AI response using OpenAI
            prompt = f"""You are a NYC restaurant compliance expert helping restaurant owners understand permits, licenses, health codes, and regulations.

User Question: {query}

Relevant NYC Data:
{data_context}

Provide comprehensive compliance guidance that includes:
1. Required permits and licenses
2. Key regulations and requirements
3. Step-by-step instructions when applicable
4. Common pitfalls to avoid
5. Resources and links to official NYC government sites

Be specific, accurate, and helpful. If the question relates to the data provided, cite specific examples."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert in NYC restaurant compliance, providing clear and actionable guidance on permits, licenses, health codes, and regulations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating compliance guidance: {str(e)}"
    
    def get_prompt(self, query: str, context: str = "") -> str:
        """Generate compliance prompt"""
        return f"Provide compliance guidance for: {query}"