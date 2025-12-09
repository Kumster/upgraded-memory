"""
FIXED Orchestrator - Properly initializes vector store and routes queries
"""
import requests
import pandas as pd
from typing import Dict, Optional
import os
from openai import OpenAI


class Orchestrator:
    def __init__(self):
        from dotenv import load_dotenv
        load_dotenv()
        
        # OpenAI configuration
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            print("WARNING: OPENAI_API_KEY not found!")
        else:
            print(f"✅ API Key loaded")
        
        self.openai_model_name = "gpt-4o-mini"
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        
        # Data storage
        self.documents = []
        self.response_cache = {}
        
        # FIXED: Actually initialize vector store!
        self._init_vector_store()
        
        # Initialize agents
        self._init_agents()
    
    def _init_vector_store(self):
        """Initialize the vector store"""
        print("📦 Initializing vector store...")
        
        try:
            # Add current directory to path for imports
            import sys
            if os.path.dirname(__file__) not in sys.path:
                sys.path.insert(0, os.path.dirname(__file__))
            
            from vector_store import VectorStore
            
            # Try multiple possible CSV paths
            possible_paths = [
                'backend/data/erm2-nwe9.csv',
                'data/erm2-nwe9.csv',
                os.path.join(os.path.dirname(__file__), 'data', 'erm2-nwe9.csv')
            ]
            
            csv_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    csv_path = path
                    break
            
            if csv_path:
                self.vector_store = VectorStore(csv_path=csv_path)
                print(f"   ✅ Vector store initialized with {self.vector_store.get_collection_size()} documents")
            else:
                # Initialize empty vector store
                self.vector_store = VectorStore(csv_path="data/erm2-nwe9.csv")  # Will create empty
                print(f"   ⚠️ No CSV found - vector store is empty (you can upload data later)")
                
        except Exception as e:
            print(f"   ⚠️ Vector store initialization error: {e}")
            self.vector_store = None
    
    def _init_agents(self):
        """Initialize all specialized agents"""
        print("🤖 Initializing agents...")
        
        # Import the intent router from agents directory
        import sys
        import os
        agents_dir = os.path.join(os.path.dirname(__file__), 'agents')
        if agents_dir not in sys.path:
            sys.path.insert(0, agents_dir)
        
        # 1. Intent Router
        from intent_router_agent import IntentRouterAgent
        self.intent_router = IntentRouterAgent(self.openai_client)
        print("   ✅ Intent Router loaded")
        
        # 2. Complaint Analyzer
        try:
            from complaint_analyzer_agent import ComplaintAnalyzerAgent
            
            # Check if vector store has complaint data
            if self.vector_store and self.vector_store.df is not None and len(self.vector_store.df) > 0:
                # Use vector store data
                if 'complaint_type' in self.vector_store.df.columns:
                    self.complaint_analyzer = ComplaintAnalyzerAgent(vector_store=self.vector_store)
                    print(f"   ✅ Complaint Analyzer loaded: {len(self.vector_store.df)} records")
                else:
                    self.complaint_analyzer = None
                    print(f"   ⚠️ Vector store doesn't contain complaint data")
            else:
                self.complaint_analyzer = None
                print(f"   ⚠️ No complaint data available")
                
        except Exception as e:
            self.complaint_analyzer = None
            print(f"   ⚠️ Could not load Complaint Analyzer: {e}")
        
        # 3. Violation Prevention Agent
        try:
            from violation_prevention_agent import ViolationPreventionAgent
            self.violation_prevention = ViolationPreventionAgent(self.openai_client, self.vector_store)
            print("   ✅ Violation Prevention Agent loaded")
        except Exception as e:
            self.violation_prevention = None
            print(f"   ⚠️ Could not load Violation Prevention Agent: {e}")
        
        # Other agents would be loaded here
        print("   ℹ️ Other agents will use OpenAI client")
    
    def generate_with_context(self, prompt: str, use_rag: bool = True, model: str = "gpt-4o-mini") -> str:
        """
        Generate response using proper agent routing
        """
        print("\n" + "="*70)
        print(f"📝 QUERY: {prompt}")
        print("="*70)
        
        # Step 1: Route the query using intent router
        routing = self.intent_router.route(prompt)
        
        print(f"\n🧭 INTENT ROUTER RESULT:")
        print(f"   Intent: {routing['intent']}")
        print(f"   Confidence: {routing['confidence']:.2f}")
        print(f"   Agents: {', '.join(routing['agents'])}")
        print(f"   Trigger Heatmap: {routing.get('trigger_heatmap', False)}")
        
        # Step 2: Execute the appropriate agent(s)
        response = self._execute_agents(prompt, routing, model)
        
        print(f"\n✅ Response generated ({len(response)} chars)")
        print("="*70 + "\n")
        
        return response
    
    def _execute_agents(self, prompt: str, routing: Dict, model: str) -> str:
        """
        Execute the agents specified in the routing plan
        """
        agents = routing['agents']
        intent = routing['intent']
        
        # PRIORITY 1: Violation Prevention (for health violation queries)
        if 'violation_prevention' in agents:
            return self._execute_violation_prevention(prompt, model)
        
        # PRIORITY 2: Complaint Analyzer (for 311 data queries)
        elif 'complaint_analyzer' in agents and self.complaint_analyzer:
            return self._execute_complaint_analyzer(prompt)
        
        # PRIORITY 3: Compliance Guide (for permit/license queries)
        elif 'compliance_guide' in agents:
            return self._execute_compliance_guide(prompt, model)
        
        # PRIORITY 4: Location Risk (for location analysis)
        elif 'location_risk' in agents:
            return self._execute_location_risk(prompt, model)
        
        # PRIORITY 5: Strategic Advisor (for business strategy)
        elif 'strategic_advisor' in agents:
            return self._execute_strategic_advisor(prompt, model)
        
        else:
            # General response
            return self._execute_general(prompt, model)
    
    def _execute_complaint_analyzer(self, prompt: str) -> str:
        """Execute complaint analyzer agent"""
        print("   🎯 Executing: COMPLAINT ANALYZER")
        
        result = self.complaint_analyzer.answer_question(prompt)
        
        # Format response
        response = f"{result['answer']}\n\n"
        
        if 'data' in result and isinstance(result['data'], dict):
            response += "📊 Key Data:\n"
            for key, value in list(result['data'].items())[:5]:
                response += f"• {key}: {value}\n"
        
        if 'sources' in result:
            response += f"\n📚 Source: {', '.join(result['sources'])}\n"
        
        return response
    
    def _execute_compliance_guide(self, prompt: str, model: str) -> str:
        """Execute compliance guide agent"""
        print("   🎯 Executing: COMPLIANCE GUIDE")
        
        system_prompt = """You are the Compliance Guide agent for Kitchen Compass AI.
        
Your expertise: NYC restaurant permits, licenses, health codes, and regulatory compliance.

RESPOND WITH:
1. 📋 **Required Permits/Licenses** - List all necessary permits with:
   - Official name
   - Issuing agency
   - Typical cost
   - Processing time
   - Key requirements

2. 📝 **Application Process** - Step-by-step instructions

3. 💡 **Pro Tips** - Common mistakes to avoid

4. 🔗 **Official Resources** - Links to NYC.gov pages

FORMAT:
- Use clear section headers with emojis
- Bullet points for lists
- Bold key terms
- Keep concise but comprehensive
- Always end with an offer to help with specific questions

DO NOT analyze 311 complaint data. Focus on permits, licenses, and regulations.
DO NOT answer questions about health violations - that's a different agent."""

        return self._call_openai_with_system_prompt(prompt, system_prompt, model)
    
    def _execute_location_risk(self, prompt: str, model: str) -> str:
        """Execute location risk agent"""
        print("   🎯 Executing: LOCATION RISK")
        
        # Check if this is a heatmap request
        is_heatmap_request = any(word in prompt.lower() for word in ['heatmap', 'heat map', 'show map', 'visualize'])
        
        if is_heatmap_request:
            system_prompt = """You are the Location Risk agent for Kitchen Compass AI.

NOTE: A visual heatmap is being displayed to the user alongside your response.

Your response should complement the heatmap visualization by providing:
1. 📊 **Borough Risk Summary** - Quick overview of the 5 boroughs
2. 🎯 **Key Insights** - What the heatmap reveals
3. 📍 **Recommendations** - Best areas based on risk levels
4. ⚠️ **Risk Factors** - What causes high/low risk in each area

Keep it concise since users can interact with the heatmap for details.
Start with: "🗺️ Here's what the heatmap reveals about NYC restaurant risks:"

FORMAT with clear sections and emojis."""
        else:
            system_prompt = """You are the Location Risk agent for Kitchen Compass AI.

Your expertise: NYC neighborhood analysis, borough comparison, location-based risk assessment.

RESPOND WITH:
1. 📍 **Borough Overview** - Risk levels across NYC
2. 🎯 **Recommended Areas** - Best neighborhoods for restaurants
3. ⚠️ **Risk Factors** - What to watch out for in each area
4. 💰 **Cost Considerations** - Rent and competition levels
5. 📊 **Data Insights** - Key statistics

Use data from NYC boroughs:
- Manhattan: High foot traffic, expensive, competitive
- Brooklyn: Growing food scene, moderate costs
- Queens: Diverse demographics, affordable
- Bronx: Emerging market, lower costs
- Staten Island: Residential, limited competition

FORMAT with clear sections, emojis, and actionable insights."""

        return self._call_openai_with_system_prompt(prompt, system_prompt, model)
    
    def _execute_strategic_advisor(self, prompt: str, model: str) -> str:
        """Execute strategic advisor agent"""
        print("   🎯 Executing: STRATEGIC ADVISOR")
        
        system_prompt = """You are the Strategic Advisor agent for Kitchen Compass AI.

Your expertise: Restaurant business strategy, operations, marketing, and financial planning.

RESPOND WITH:
1. 💡 **Strategic Recommendations** - Actionable business advice
2. 📈 **Market Analysis** - NYC restaurant landscape insights
3. 💰 **Financial Considerations** - Budget and revenue planning
4. 🎯 **Target Market** - Customer demographics and preferences
5. ⚡ **Quick Wins** - Immediate actions to take

Focus on practical, NYC-specific advice for new restaurant owners."""

        return self._call_openai_with_system_prompt(prompt, system_prompt, model)
    
    def _execute_violation_prevention(self, prompt: str, model: str) -> str:
        """
        Execute violation prevention agent
        """
        print("   🎯 Executing: VIOLATION PREVENTION AGENT")
        
        # Use the actual agent if available
        if self.violation_prevention:
            print("      Using ViolationPreventionAgent.process()")
            return self.violation_prevention.process(prompt)
        
        # Fallback to OpenAI if agent not available
        print("      ⚠️ Falling back to OpenAI system prompt")
        system_prompt = """You are the Violation Prevention agent for Kitchen Compass AI.

Your expertise: NYC health code violations and how to prevent them.

RESPOND WITH:
1. ⚠️ **Most Common Health Violations in NYC Restaurants**
2. ✅ **Prevention Best Practices**
3. 🔍 **Inspection Preparation**
4. 📋 **Daily Compliance Checklist**
5. 🛡️ **Staff Training Areas**

Provide specific, actionable advice with NYC health code citations when relevant."""

        return self._call_openai_with_system_prompt(prompt, system_prompt, model)
    
    def _execute_general(self, prompt: str, model: str) -> str:
        """Execute general response"""
        print("   🎯 Executing: GENERAL")
        
        system_prompt = """You are Kitchen Compass AI, a helpful assistant for NYC restaurant entrepreneurs.

Provide clear, concise answers with:
- Relevant emojis for sections
- Bullet points for clarity
- Specific NYC context when applicable
- Offer to help with more detailed questions"""

        return self._call_openai_with_system_prompt(prompt, system_prompt, model)
    
    def _call_openai_with_system_prompt(self, prompt: str, system_prompt: str, model: str) -> str:
        """Call OpenAI with a specific system prompt"""
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            resp = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7
            )
            
            return resp.choices[0].message.content
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def ingest_csv(self, df: pd.DataFrame) -> Dict:
        """Ingest CSV data into vector store"""
        if self.vector_store:
            return self.vector_store.ingest_dataframe(df)
        else:
            print("⚠️ Vector store not initialized")
            return {
                'success': False,
                'error': 'Vector store not initialized',
                'rows_ingested': 0
            }
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        stats = {
            'document_count': len(self.documents),
            'total_chunks': len(self.documents),
            'cache_size': len(self.response_cache),
            'models_available': ['gpt-4o-mini']
        }
        
        # Add vector store stats
        if self.vector_store:
            stats['vector_store_size'] = self.vector_store.get_collection_size()
            stats['vector_store_active'] = True
        else:
            stats['vector_store_size'] = 0
            stats['vector_store_active'] = False
        
        if self.complaint_analyzer:
            stats['complaint_records'] = len(self.vector_store.df) if self.vector_store and self.vector_store.df is not None else 0
            stats['complaint_agent_active'] = True
        else:
            stats['complaint_agent_active'] = False
        
        if self.violation_prevention:
            stats['violation_prevention_active'] = True
        else:
            stats['violation_prevention_active'] = False
        
        return stats
    
    def get_agent_status(self) -> Dict:
        """Get agent status"""
        agents = [
            {'name': 'Intent Router', 'status': 'active', 'usage_count': 0},
            {'name': 'Complaint Analyzer', 'status': 'active' if self.complaint_analyzer else 'inactive', 'usage_count': 0},
            {'name': 'Compliance Guide', 'status': 'active', 'usage_count': 0},
            {'name': 'Location Risk', 'status': 'active', 'usage_count': 0},
            {'name': 'Strategic Advisor', 'status': 'active', 'usage_count': 0},
            {'name': 'Violation Prevention', 'status': 'active' if self.violation_prevention else 'inactive', 'usage_count': 0},
        ]
        
        return {
            'agents': agents,
            'total_executions': 0
        }