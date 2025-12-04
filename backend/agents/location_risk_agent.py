"""
Enhanced Location Risk Agent with Heatmap Support
Analyzes geographic patterns in NYC 311 complaint data
"""

from backend.agents.base_agent import BaseAgent
from typing import Dict, List, Any, Optional
import json
import re

class LocationRiskAgent(BaseAgent):
    def __init__(self, openai_client, vector_store):
        super().__init__("Location Risk Analyzer", "Analyzes location-based risks for restaurants")
        self.openai_client = openai_client
        self.vector_store = vector_store
        
        # NYC Borough center coordinates
        self.borough_coords = {
            'MANHATTAN': {'lat': 40.7831, 'lng': -73.9712, 'zoom': 13},
            'BROOKLYN': {'lat': 40.6782, 'lng': -73.9442, 'zoom': 12},
            'QUEENS': {'lat': 40.7282, 'lng': -73.7949, 'zoom': 11},
            'BRONX': {'lat': 40.8448, 'lng': -73.8648, 'zoom': 12},
            'STATEN ISLAND': {'lat': 40.5795, 'lng': -74.1502, 'zoom': 12}
        }
        
        # Complaint type severity weights (higher = more risky)
        self.severity_weights = {
            'Food Poisoning': 5,
            'Health': 4,
            'Rodent': 4,
            'Unsanitary Condition': 4,
            'Consumer Complaint': 3,
            'Vendor Enforcement': 2,
            'Outdoor Dining': 2,
            'Mobile Food Vendor': 2,
            'Noise': 1,
            'General': 1
        }
        
        # Neighborhood risk data (sample - would be populated from actual data)
        self.neighborhood_risks = {}
    
    def process(self, query: str, context: str = "", **kwargs) -> str:
        """
        Process location risk query and return analysis
        """
        # Check if user wants visualization
        wants_map = any(word in query.lower() for word in [
            'map', 'heatmap', 'show', 'visualize', 'hotspot', 'where', 'location', 'area'
        ])
        
        # Parse location data from context
        location_data = self._parse_location_data(context)
        
        # Calculate risk scores
        risk_analysis = self._calculate_risk_scores(location_data)
        
        # Generate LLM analysis
        analysis = self._generate_analysis(query, context, risk_analysis)
        
        # If map requested, append structured data for frontend
        if wants_map and location_data:
            map_data = self._prepare_heatmap_data(location_data)
            analysis += f"\n\n<!-- HEATMAP_DATA:{json.dumps(map_data)} -->"
        
        return analysis
    
    def _parse_location_data(self, context: str) -> List[Dict]:
        """
        Parse location information from context
        Extracts borough, address, lat/lng if available
        """
        locations = []
        
        # Split context into individual records
        records = context.split('\n\n')
        
        for record in records:
            location = {}
            record_lower = record.lower()
            
            # Extract borough
            for borough in self.borough_coords.keys():
                if borough.lower() in record_lower:
                    location['borough'] = borough
                    location['lat'] = self.borough_coords[borough]['lat']
                    location['lng'] = self.borough_coords[borough]['lng']
                    break
            
            # Extract complaint type
            for complaint_type in self.severity_weights.keys():
                if complaint_type.lower() in record_lower:
                    location['complaint_type'] = complaint_type
                    location['severity'] = self.severity_weights[complaint_type]
                    break
            
            # Extract coordinates if present (format: lat, lng or latitude/longitude)
            lat_match = re.search(r'lat(?:itude)?[:\s]+(-?\d+\.?\d*)', record_lower)
            lng_match = re.search(r'(?:lng|lon|longitude)[:\s]+(-?\d+\.?\d*)', record_lower)
            
            if lat_match and lng_match:
                location['lat'] = float(lat_match.group(1))
                location['lng'] = float(lng_match.group(1))
            
            # Extract status
            if 'in progress' in record_lower:
                location['status'] = 'In Progress'
            elif 'closed' in record_lower:
                location['status'] = 'Closed'
            
            # Extract address if present
            address_match = re.search(r'address[:\s]+([^\n]+)', record, re.IGNORECASE)
            if address_match:
                location['address'] = address_match.group(1).strip()
            
            if location.get('borough') or location.get('lat'):
                locations.append(location)
        
        return locations
    
    def _calculate_risk_scores(self, location_data: List[Dict]) -> Dict[str, Any]:
        """
        Calculate risk scores by borough and overall
        """
        borough_stats = {}
        
        for loc in location_data:
            borough = loc.get('borough', 'UNKNOWN')
            if borough not in borough_stats:
                borough_stats[borough] = {
                    'count': 0,
                    'total_severity': 0,
                    'complaint_types': {},
                    'in_progress': 0
                }
            
            borough_stats[borough]['count'] += 1
            borough_stats[borough]['total_severity'] += loc.get('severity', 1)
            
            complaint_type = loc.get('complaint_type', 'General')
            borough_stats[borough]['complaint_types'][complaint_type] = \
                borough_stats[borough]['complaint_types'].get(complaint_type, 0) + 1
            
            if loc.get('status') == 'In Progress':
                borough_stats[borough]['in_progress'] += 1
        
        # Calculate risk scores (0-100)
        for borough, stats in borough_stats.items():
            if stats['count'] > 0:
                avg_severity = stats['total_severity'] / stats['count']
                # Risk formula: weighted by count, severity, and in-progress complaints
                risk_score = min(100, (stats['count'] * 5) + (avg_severity * 10) + (stats['in_progress'] * 15))
                stats['risk_score'] = round(risk_score, 1)
                stats['risk_level'] = self._get_risk_level(risk_score)
        
        # Rank boroughs by risk
        ranked = sorted(borough_stats.items(), key=lambda x: x[1].get('risk_score', 0), reverse=True)
        
        return {
            'borough_stats': borough_stats,
            'ranked_boroughs': ranked,
            'total_complaints': sum(s['count'] for s in borough_stats.values()),
            'highest_risk': ranked[0][0] if ranked else None,
            'lowest_risk': ranked[-1][0] if ranked else None
        }
    
    def _get_risk_level(self, score: float) -> str:
        """Convert numeric risk score to risk level"""
        if score >= 70:
            return 'High'
        elif score >= 40:
            return 'Medium'
        else:
            return 'Low'
    
    def _generate_analysis(self, query: str, context: str, risk_analysis: Dict) -> str:
        """
        Generate natural language analysis using LLM
        """
        borough_summary = ""
        for borough, stats in risk_analysis.get('borough_stats', {}).items():
            borough_summary += f"\n- {borough}: {stats['count']} complaints, Risk Score: {stats.get('risk_score', 'N/A')}, Level: {stats.get('risk_level', 'N/A')}"
        
        prompt = f"""You are a NYC restaurant location advisor analyzing 311 complaint data.

**User Question:** {query}

**Location Risk Analysis:**
- Total Complaints Analyzed: {risk_analysis.get('total_complaints', 0)}
- Highest Risk Borough: {risk_analysis.get('highest_risk', 'N/A')}
- Lowest Risk Borough: {risk_analysis.get('lowest_risk', 'N/A')}

**Borough Breakdown:**
{borough_summary}

**Raw Context:**
{context[:1500]}

**Provide a response with:**

## Location Risk Assessment

### Borough Risk Rankings
(Rank from safest to riskiest with specific data)

### Recommended Areas
(Top 2-3 areas for opening a restaurant and why)

### Areas to Avoid or Approach with Caution
(Which areas have concerning patterns)

### Key Risk Factors by Location
(What specific issues affect each area)

### Actionable Recommendations
1. (First recommendation)
2. (Second recommendation)
3. (Third recommendation)

Be specific with numbers and borough names. Keep response under 400 words."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are Kitchen Compass AI's Location Risk Specialist. You help restaurant owners choose the best locations based on 311 complaint data analysis. Be specific, data-driven, and actionable."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating analysis: {str(e)}"
    
    def _prepare_heatmap_data(self, location_data: List[Dict]) -> Dict[str, Any]:
        """
        Prepare data for heatmap visualization
        Returns GeoJSON-compatible format for ArcGIS/Leaflet
        """
        features = []
        
        for loc in location_data:
            if loc.get('lat') and loc.get('lng'):
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [loc['lng'], loc['lat']]
                    },
                    "properties": {
                        "borough": loc.get('borough', 'Unknown'),
                        "complaint_type": loc.get('complaint_type', 'General'),
                        "severity": loc.get('severity', 1),
                        "status": loc.get('status', 'Unknown'),
                        "address": loc.get('address', '')
                    }
                }
                features.append(feature)
        
        # Add borough center points with aggregated data
        borough_aggregates = {}
        for loc in location_data:
            borough = loc.get('borough')
            if borough:
                if borough not in borough_aggregates:
                    borough_aggregates[borough] = {
                        'count': 0,
                        'total_severity': 0
                    }
                borough_aggregates[borough]['count'] += 1
                borough_aggregates[borough]['total_severity'] += loc.get('severity', 1)
        
        for borough, data in borough_aggregates.items():
            if borough in self.borough_coords:
                coords = self.borough_coords[borough]
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [coords['lng'], coords['lat']]
                    },
                    "properties": {
                        "borough": borough,
                        "complaint_count": data['count'],
                        "avg_severity": round(data['total_severity'] / data['count'], 2) if data['count'] > 0 else 0,
                        "is_aggregate": True
                    }
                }
                features.append(feature)
        
        return {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "total_points": len(location_data),
                "boroughs_covered": list(borough_aggregates.keys()),
                "center": {"lat": 40.7128, "lng": -74.0060},  # NYC center
                "zoom": 11
            }
        }
    
    def get_heatmap_data(self, query: str = "", filters: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Public method to get heatmap data for API endpoint
        Can be called directly from FastAPI route
        """
        # Search vector store for location data
        search_query = query if query else "complaints locations borough NYC restaurant"
        docs = self.vector_store.search(search_query, k=50)
        
        # Combine document content
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Parse and prepare data
        location_data = self._parse_location_data(context)
        
        # Apply filters if provided
        if filters:
            if filters.get('borough'):
                location_data = [l for l in location_data if l.get('borough') == filters['borough'].upper()]
            if filters.get('complaint_type'):
                location_data = [l for l in location_data if l.get('complaint_type') == filters['complaint_type']]
        
        # Prepare heatmap data
        heatmap_data = self._prepare_heatmap_data(location_data)
        
        # Add risk analysis
        risk_analysis = self._calculate_risk_scores(location_data)
        heatmap_data['risk_analysis'] = risk_analysis
        
        return heatmap_data
    
    def get_prompt(self, query: str, context: str = "") -> str:
        """Generate location risk prompt"""
        return f"Analyze location risks for NYC restaurant: {query}\n\nContext: {context}"