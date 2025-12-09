"""
ENHANCED Intent Router Agent
Routes user queries to the appropriate specialized agent(s)
UPDATED: Better keyword detection for health violation queries
"""

class IntentRouterAgent:
    def __init__(self, openai_client):
        self.openai_client = openai_client
        self.name = "Intent Router"
        
        # Enhanced intent classification patterns with weighted keywords
        self.intent_patterns = {
            'permits_licenses': {
                'keywords': [
                    'permit', 'permits', 'license', 'licenses', 'application',
                    'food service establishment permit', 'liquor license',
                    'certificate of occupancy', 'business license', 'sign permit',
                    'health department permit', 'what do i need', 'requirements',
                    'how to get', 'apply for', 'obtain', 'acquire'
                ],
                'agents': ['compliance_guide'],
                'priority': 'high'
            },
            
            'compliance_health': {
                'keywords': [
                    'health code', 'health department', 'inspection', 'health standards',
                    'food safety', 'sanitation rules', 'hygiene', 'compliance', 'regulations',
                    'grade', 'health grade', 'inspection failures', 'failed inspection'
                ],
                'agents': ['compliance_guide'],
                'priority': 'high'
            },
            
            'violation_prevention': {
                'keywords': [
                    # ENHANCED: More comprehensive violation keywords
                    'violation', 'violations', 'citation', 'citations',
                    'common violations', 'most common violations', 'typical violations',
                    'frequent violations', 'health violations', 'code violations',
                    'what violations', 'list of violations', 'violation types',
                    'avoid violations', 'prevent violations', 'stop violations',
                    'how to avoid', 'what not to do', 'avoid citations',
                    'stay compliant', 'proactive measures', 'best practices',
                    'common mistakes', 'violations should i', 'violations to watch',
                    'violations risk', 'prevent health', 'avoid health',
                    'inspection violations', 'health inspection issues',
                    # NEW: Specific violation type keywords
                    'pest', 'mice', 'roaches', 'rodent', 'vermin',
                    'temperature', 'food temp', 'hot holding', 'cold holding',
                    'cross contamination', 'food storage', 'sanitation',
                    'hand washing', 'hygiene violations'
                ],
                'agents': ['violation_prevention'],
                'priority': 'highest'  # HIGHEST priority for violation queries
            },
            
            'location_analysis': {
                'keywords': [
                    'location', 'area', 'neighborhood', 'where should',
                    'best location', 'safest', 'risk assessment',
                    'brooklyn', 'manhattan', 'queens', 'bronx', 'staten island',
                    'which borough', 'which area', 'where to open',
                    'neighborhood analysis', 'location comparison'
                ],
                'agents': ['location_risk'],
                'priority': 'medium'
            },
            
            'heatmap_visualization': {
                'keywords': [
                    'heatmap', 'heat map', 'show map', 'visualize', 'visualization',
                    'show me map', 'display map', 'map of', 'interactive map',
                    'risk map', 'complaint map'
                ],
                'agents': ['location_risk'],
                'priority': 'high',
                'trigger_heatmap': True
            },
            
            'complaint_data': {
                'keywords': [
                    '311', '311 data', '311 complaints', 'complaint data',
                    'specific complaints', 'complaint history', 'past complaints',
                    'show complaints', 'list complaints', 'get complaints',
                    'address complaints', 'location complaints'
                ],
                'agents': ['complaint_analyzer'],
                'priority': 'high'
            },
            
            'business_strategy': {
                'keywords': [
                    'strategy', 'business plan', 'marketing', 'operations',
                    'financial planning', 'budget', 'revenue', 'profit',
                    'target market', 'customer demographics', 'menu planning',
                    'pricing strategy', 'competitive analysis'
                ],
                'agents': ['strategic_advisor'],
                'priority': 'medium'
            },
            
            'general_info': {
                'keywords': [
                    'hello', 'hi', 'hey', 'help', 'what can you do',
                    'capabilities', 'features', 'about', 'information',
                    'tell me about', 'explain'
                ],
                'agents': ['general'],
                'priority': 'low'
            }
        }
    
    def route(self, query: str) -> dict:
        """
        Route user query to appropriate agent(s)
        ENHANCED: Better scoring and prioritization
        """
        query_lower = query.lower()
        
        # Calculate scores for each intent
        intent_scores = {}
        
        for intent_name, intent_data in self.intent_patterns.items():
            score = 0
            matched_keywords = []
            
            for keyword in intent_data['keywords']:
                if keyword in query_lower:
                    # Longer phrases get higher weight
                    keyword_weight = len(keyword.split())
                    score += keyword_weight
                    matched_keywords.append(keyword)
            
            if score > 0:
                # Apply priority multiplier
                priority_multipliers = {
                    'highest': 2.0,  # NEW: Highest priority
                    'high': 1.5,
                    'medium': 1.0,
                    'low': 0.5
                }
                priority = intent_data.get('priority', 'medium')
                score *= priority_multipliers.get(priority, 1.0)
                
                intent_scores[intent_name] = {
                    'score': score,
                    'agents': intent_data['agents'],
                    'matched_keywords': matched_keywords,
                    'trigger_heatmap': intent_data.get('trigger_heatmap', False)
                }
        
        # If no matches, default to general
        if not intent_scores:
            return {
                'intent': 'general_info',
                'agents': ['general'],
                'confidence': 0.3,
                'trigger_heatmap': False
            }
        
        # Get the highest scoring intent
        best_intent = max(intent_scores.items(), key=lambda x: x[1]['score'])
        intent_name = best_intent[0]
        intent_info = best_intent[1]
        
        # Calculate confidence (0-1 scale)
        max_possible_score = 20  # Rough estimate
        confidence = min(1.0, intent_info['score'] / max_possible_score)
        
        return {
            'intent': intent_name,
            'agents': intent_info['agents'],
            'confidence': confidence,
            'trigger_heatmap': intent_info.get('trigger_heatmap', False),
            'matched_keywords': intent_info.get('matched_keywords', [])
        }


# ========== TESTING ==========

if __name__ == "__main__":
    print("🧪 TESTING ENHANCED INTENT ROUTER\n")
    print("="*70)
    
    router = IntentRouterAgent(None)
    
    test_queries = [
        "What are the most common health violations?",
        "How can I prevent violations?",
        "What permits do I need?",
        "Show me a heatmap of restaurant risks",
        "Where should I open my restaurant in Brooklyn?",
        "What's the best strategy for my restaurant?"
    ]
    
    for query in test_queries:
        result = router.route(query)
        print(f"\n📝 Query: {query}")
        print(f"   → Intent: {result['intent']}")
        print(f"   → Agents: {', '.join(result['agents'])}")
        print(f"   → Confidence: {result['confidence']:.2f}")
        if result.get('matched_keywords'):
            print(f"   → Matched: {', '.join(result['matched_keywords'][:3])}")
        print("-" * 70)