"""
Base Agent Class for Kitchen AI Compass Multi-Agent System
"""

from typing import Dict, List, Optional
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """
    Abstract base class for all Kitchen AI Compass agents.
    """
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.agent_type = self.__class__.__name__
        
    @abstractmethod
    def process(self, query: str, context: str = "", **kwargs) -> Dict:
        """Process a user query and return structured response"""
        pass
    
    @abstractmethod
    def get_prompt(self, query: str, context: str = "") -> str:
        """Generate the specialized prompt for this agent"""
        pass
    
    def validate_query(self, query: str) -> bool:
        """Basic query validation"""
        if not query or not query.strip():
            return False
        if len(query) < 3:
            return False
        return True
    
    def format_response(self, content: str, metadata: Dict = None) -> Dict:
        """Format agent response with metadata"""
        response = {
            'agent': self.name,
            'content': content,
            'success': True
        }
        
        if metadata:
            response['metadata'] = metadata
            
        return response
    
    def format_error(self, error_msg: str) -> Dict:
        """Format error response"""
        return {
            'agent': self.name,
            'content': f"Error: {error_msg}",
            'success': False,
            'error': error_msg
        }
    
    def __str__(self) -> str:
        return f"{self.name}: {self.description}"
    
    def __repr__(self) -> str:
        return f"<{self.agent_type}(name='{self.name}')>"
