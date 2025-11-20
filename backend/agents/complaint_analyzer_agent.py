"""
311 Complaint Analyzer Agent
Analyzes NYC 311 complaint data using OpenAI and vector store
"""

from backend.agents.base_agent import BaseAgent
from typing import Dict, Any, Optional

class ComplaintAnalyzerAgent(BaseAgent):
    """Analyzes 311 complaint data"""
    
    def __init__(self, openai_client, vector_store):
        super().__init__("Complaint Analyzer", "Analyzes NYC 311 complaint data")
        self.openai_client = openai_client
        self.vector_store = vector_store
    
    def process(self, query: str, context: str = "", **kwargs) -> str:
        """
        Process complaint analysis query
        Returns AI-generated analysis based on 311 complaint data
        """
        try:
            # Search for relevant complaints
            results = self.vector_store.search(query, k=5)
            
            if not results:
                return "I couldn't find any relevant complaint data for your query. Please try rephrasing your question or ask about a different topic."
            
            # Build context from search results
            complaints_context = "\n\n".join([
                f"Complaint {i+1}: {doc.page_content}" 
                for i, doc in enumerate(results)
            ])
            
            # Generate AI response using OpenAI
            prompt = f"""You are a NYC restaurant compliance expert analyzing 311 complaint data.

User Question: {query}

Relevant 311 Complaint Data:
{complaints_context}

Based on the complaint data above, provide a comprehensive analysis that:
1. Identifies key patterns and trends in the complaints
2. Highlights the most common issues
3. Provides actionable insights for restaurant owners
4. Suggests preventive measures

Be specific and cite examples from the data when relevant."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert analyst of NYC 311 complaint data, helping restaurant owners understand and prevent common issues."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error analyzing complaints: {str(e)}"
    
    def get_prompt(self, query: str, context: str = "") -> str:
        return f"Analyze 311 complaints for: {query}"