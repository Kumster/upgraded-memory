import requests
import pandas as pd
from typing import Dict
import os
from openai import OpenAI

class Orchestrator:
    def __init__(self):
        # Load environment variables from .env file
        from dotenv import load_dotenv
        load_dotenv()
        
        # Ollama configuration
        self.ollama_base_url = "http://localhost:11434"
        self.ollama_model_name = "gemma:2b"
        
        # OpenAI configuration
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            print("WARNING: OPENAI_API_KEY not found in environment!")
        else:
            print(f"✅ API Key loaded")
        
        self.openai_model_name = "gpt-4o-mini"
        
        # Data storage
        self.documents = []
        self.response_cache = {}
        
        # ADD COMPLAINT ANALYZER AGENT (lazy import - only loads when __init__ runs)
        self.complaint_agent = None
        self._load_complaint_analyzer()
    
    def _load_complaint_analyzer(self):
        """Load complaint analyzer with lazy import to avoid import errors at startup"""
        try:
            # Install plotly if not available
            try:
                import plotly
            except ImportError:
                print("⚠️ Plotly not installed. Installing now...")
                import subprocess
                subprocess.check_call(['pip', 'install', 'plotly', '-q'])
                print("✅ Plotly installed")
            
            # Direct import of complaint analyzer
            import sys
            agent_dir = os.path.join(os.path.dirname(__file__), 'agents')
            if agent_dir not in sys.path:
                sys.path.insert(0, agent_dir)
            
            from complaint_analyzer_agent import ComplaintAnalyzerAgent
            
            # Load the CSV data
            csv_path = 'backend/data/erm2-nwe9.csv'
            if os.path.exists(csv_path):
                self.complaint_agent = ComplaintAnalyzerAgent(csv_path)
                print(f"✅ Complaint Analyzer loaded: {len(self.complaint_agent.df)} records")
            else:
                print(f"⚠️ CSV not found at {csv_path}")
        except Exception as e:
            print(f"⚠️ Could not load Complaint Analyzer: {e}")
            import traceback
            traceback.print_exc()
        
    def generate_with_context(self, prompt: str, use_rag: bool = True, model: str = "gpt-4o-mini") -> str:
        """
        Generate response using either GPT-4o-mini or Ollama
        """
        cache_key = f"{prompt}_{use_rag}_{model}"
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        # CHECK IF THIS IS A COMPLAINT-RELATED QUERY
        if self.complaint_agent:
            complaint_result = self._check_complaint_query(prompt)
            if complaint_result:
                self.response_cache[cache_key] = complaint_result
                return complaint_result
        
        # Retrieve context if using RAG
        context = ""
        if use_rag and self.documents:
            context = self._simple_retrieve(prompt)
        
        # Build the full prompt
        full_prompt = self._build_prompt(prompt, context)
        
        # Generate response based on selected model
        if model == "gpt-4o-mini":
            response = self._call_openai_direct(prompt, context)
        elif model == "ollama":
            response = self._call_ollama(full_prompt)
        else:
            response = f"Error: Unknown model '{model}'. Use 'gpt-4o-mini' or 'ollama'."
        
        self.response_cache[cache_key] = response
        return response
    
    def _check_complaint_query(self, prompt: str) -> str:
        """Check if query is about complaints and route to complaint agent"""
        if not self.complaint_agent:
            return None
        
        prompt_lower = prompt.lower()
        
        # Keywords that indicate complaint-related queries
        complaint_keywords = [
            'complaint', 'noise', '311', 'violation', 'sanitation',
            'zip code', 'zipcode', 'zip', 'borough', 'neighborhood',
            'area', 'location risk', 'manhattan', 'brooklyn', 
            'queens', 'bronx', 'staten island', 'where should i open'
        ]
        
        # Check if any keyword is in the prompt
        if any(keyword in prompt_lower for keyword in complaint_keywords):
            print(f"🎯 Routing to Complaint Analyzer...")
            
            # Use the complaint agent to answer
            result = self.complaint_agent.answer_question(prompt)
            
            # Format the response nicely
            response = f"📊 {result['answer']}\n\n"
            
            # Add data summary if available
            if 'data' in result and isinstance(result['data'], dict):
                response += "Key Data:\n"
                for key, value in list(result['data'].items())[:5]:
                    response += f"• {key}: {value}\n"
            
            # Add sources
            if 'sources' in result:
                response += f"\n📚 Source: {', '.join(result['sources'])}\n"
            
            return response
        
        return None
    
    def _call_openai_direct(self, prompt: str, context: str) -> str:
        """Call OpenAI API DIRECTLY"""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_api_key)
            
            system_content = """You are Kitchen AI Compass, an expert assistant for NYC restaurant inspections and food safety regulations.

FORMATTING GUIDELINES:
- Use clear section headers with relevant emojis (🍽️, 🏥, 📋, 💡, ⚠️, etc.)
- Group related information into organized sections
- Use bullet points (•) for lists
- Keep paragraphs short and scannable (2-3 sentences max)
- Bold important terms using **text**
- Highlight key warnings or notes
- End with a helpful call-to-action or offer to help further
- Make responses visually organized and easy to read

Keep responses concise but comprehensive. Focus on actionable information."""

            if context:
                system_content += f"\n\nRelevant Information from your knowledge base:\n{context}"
            
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ]
            
            print('--- CALLING OPENAI DIRECTLY ---')
            print(f"Model: {self.openai_model_name}")
            print(f"Prompt: {prompt[:100]}...")
            
            resp = client.chat.completions.create(
                model=self.openai_model_name,
                messages=messages,
                temperature=0.7
            )
            
            result = resp.choices[0].message.content
            print(f"✅ SUCCESS! Response length: {len(result)} chars")
            print('--- END OPENAI CALL ---')
            
            return result
            
        except Exception as e:
            import traceback
            error_type = type(e).__name__
            error_msg = str(e)
            
            print("="*60)
            print(f"❌ ERROR TYPE: {error_type}")
            print(f"❌ ERROR MESSAGE: {error_msg}")
            print("="*60)
            print("FULL TRACEBACK:")
            print(traceback.format_exc())
            print("="*60)
            
            return f"OpenAI API Error ({error_type}): {error_msg}\n\nCheck the terminal for full error details."
    
    def _simple_retrieve(self, query: str, top_k: int = 3) -> str:
        """Simple keyword-based retrieval from documents"""
        query_words = set(query.lower().split())
        scored_docs = []
        for doc in self.documents:
            doc_words = set(doc['text'].lower().split())
            score = len(query_words & doc_words)
            if score > 0:
                scored_docs.append((score, doc))
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        top_docs = [doc for _, doc in scored_docs[:top_k]]
        context = "\n\n".join([doc['text'] for doc in top_docs])
        return context if context else ""
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build enhanced prompt with context"""
        if context:
            return f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        return query
    
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama local model"""
        try:
            print('--- CALLING OLLAMA ---')
            print(f"Model: {self.ollama_model_name}")
            print(f"Prompt: {prompt[:100]}...")
            
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.ollama_model_name,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json().get('response', 'No response generated.')
                print(f"✅ Ollama SUCCESS! Response length: {len(result)} chars")
                return result
            else:
                error_msg = f"Ollama returned status code {response.status_code}"
                print(f"❌ Ollama ERROR: {error_msg}")
                return f"Error calling Ollama: {error_msg}"
                
        except requests.exceptions.ConnectionError:
            return "Error: Cannot connect to Ollama. Make sure Ollama is running (ollama serve) and the gemma:2b model is available."
        except requests.exceptions.Timeout:
            return "Error: Ollama request timed out. The model might be taking too long to respond."
        except Exception as e:
            return f"Error calling Ollama: {str(e)}"
    
    def ingest_csv(self, df: pd.DataFrame) -> Dict:
        """Ingest CSV data into vector store"""
        try:
            self.documents = []
            
            for idx, row in df.iterrows():
                text_parts = []
                for col, val in row.items():
                    if pd.notna(val):
                        text_parts.append(f"{col}: {val}")
                
                doc_text = " | ".join(text_parts)
                
                self.documents.append({
                    'text': doc_text,
                    'metadata': row.to_dict()
                })
            
            print(f"✅ Ingested {len(self.documents)} documents from CSV")
            
            return {
                'success': True,
                'rows_ingested': len(df),
                'total_chunks': len(self.documents)
            }
            
        except Exception as e:
            print(f"❌ CSV Ingestion Error: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        stats = {
            'document_count': len(self.documents),
            'total_chunks': len(self.documents),
            'cache_size': len(self.response_cache),
            'models_available': ['gpt-4o-mini', 'ollama']
        }
        
        # Add complaint agent stats if available
        if self.complaint_agent:
            stats['complaint_records'] = len(self.complaint_agent.df)
            stats['complaint_agent_active'] = True
        else:
            stats['complaint_agent_active'] = False
        
        return stats

    def compare_models(self, prompt: str) -> Dict:
        """Compare responses from both models"""
        gpt_response = self.generate_with_context(prompt, model="gpt-4o-mini")
        ollama_response = self.generate_with_context(prompt, model="ollama")
        return {
            'gpt4o_mini_response': gpt_response,
            'ollama_response': ollama_response,
            'comparison_available': True
        }