"""
311 Complaint Analyzer Agent - No Plotly Version
Analyzes NYC 311 complaints to help restaurant entrepreneurs understand neighborhood risks
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

class ComplaintAnalyzerAgent:
    """
    Analyzes 311 complaint data to provide insights about neighborhood risks,
    noise complaints, and other factors that could affect restaurant success.
    """
    
    def __init__(self, data_path: str):
        """Initialize the agent with 311 complaint data"""
        self.data_path = data_path
        self.df = None
        self.load_and_prepare_data()
        
    def load_and_prepare_data(self):
        """Load and clean the 311 complaint data"""
        print("Loading 311 complaint data...")
        self.df = pd.read_csv(self.data_path)
        
        # Clean and prepare data
        self.df['created_date'] = pd.to_datetime(self.df['created_date'], errors='coerce')
        self.df['borough'] = self.df['borough'].fillna('Unknown')
        self.df['complaint_type'] = self.df['complaint_type'].fillna('Other')
        self.df['descriptor'] = self.df['descriptor'].fillna('Not Specified')
        
        # Extract useful features
        if 'created_date' in self.df.columns:
            self.df['year'] = self.df['created_date'].dt.year
            self.df['month'] = self.df['created_date'].dt.month
            self.df['day_of_week'] = self.df['created_date'].dt.day_name()
            self.df['hour'] = self.df['created_date'].dt.hour
        
        # Categorize restaurant-relevant complaints
        self.df['restaurant_relevant'] = self.df['complaint_type'].apply(
            self._is_restaurant_relevant
        )
        
        print(f"Loaded {len(self.df)} complaint records")
        print(f"Date range: {self.df['created_date'].min()} to {self.df['created_date'].max()}")
        
    def _is_restaurant_relevant(self, complaint_type: str) -> bool:
        """Determine if a complaint type is relevant to restaurant businesses"""
        relevant_keywords = [
            'noise', 'sanitation', 'food', 'health', 'restaurant',
            'rodent', 'garbage', 'cleanliness', 'smell', 'odor'
        ]
        if pd.isna(complaint_type):
            return False
        complaint_lower = complaint_type.lower()
        return any(keyword in complaint_lower for keyword in relevant_keywords)
    
    def get_borough_summary(self, borough: Optional[str] = None) -> Dict[str, Any]:
        """Get complaint summary for a specific borough or all boroughs"""
        if borough:
            data = self.df[self.df['borough'] == borough.upper()]
        else:
            data = self.df
            
        summary = {
            'total_complaints': int(len(data)),
            'restaurant_relevant_complaints': int(data['restaurant_relevant'].sum()),
            'most_common_complaint': data['complaint_type'].value_counts().head(1).to_dict(),
            'complaints_by_type': data['complaint_type'].value_counts().head(10).to_dict(),
            'borough_breakdown': self.df['borough'].value_counts().to_dict()
        }
        
        return summary
    
    def analyze_neighborhood_risk(self, zip_code: str) -> Dict[str, Any]:
        """
        Analyze complaint risk for a specific ZIP code
        Returns risk assessment and recommendations
        """
        # Convert to string and clean
        zip_code = str(zip_code).strip()
        
        # Filter data - handle both string and float comparisons
        zip_data = self.df[self.df['incident_zip'].astype(str).str.strip() == zip_code]
        
        if len(zip_data) == 0:
            return {
                'zip_code': zip_code,
                'risk_level': 'Unknown',
                'message': 'No complaint data available for this ZIP code',
                'total_complaints': 0
            }
        
        # Calculate risk metrics
        total_complaints = len(zip_data)
        noise_complaints = len(zip_data[zip_data['complaint_type'].str.contains('Noise', na=False)])
        restaurant_relevant = int(zip_data['restaurant_relevant'].sum())
        
        # Calculate risk score (simple heuristic)
        complaints_per_month = total_complaints / max(1, zip_data['created_date'].nunique() / 30)
        
        if complaints_per_month > 50:
            risk_level = 'High'
        elif complaints_per_month > 20:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        return {
            'zip_code': zip_code,
            'risk_level': risk_level,
            'total_complaints': int(total_complaints),
            'noise_complaints': int(noise_complaints),
            'restaurant_relevant_complaints': int(restaurant_relevant),
            'complaints_per_month': round(complaints_per_month, 2),
            'top_complaint_types': zip_data['complaint_type'].value_counts().head(5).to_dict(),
            'recommendation': self._get_risk_recommendation(risk_level, noise_complaints, total_complaints)
        }
    
    def _get_risk_recommendation(self, risk_level: str, noise_complaints: int, total: int) -> str:
        """Generate recommendation based on risk level"""
        noise_pct = (noise_complaints / max(1, total)) * 100
        
        if risk_level == 'High':
            return f"⚠️ High complaint area ({noise_pct:.1f}% noise-related). Consider soundproofing and strict closing hours."
        elif risk_level == 'Medium':
            return f"⚡ Moderate complaints ({noise_pct:.1f}% noise-related). Plan for good neighbor relations and compliance."
        else:
            return f"✅ Low complaint area ({noise_pct:.1f}% noise-related). Good location for a restaurant."
    
    def answer_question(self, question: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Answer user questions about complaints using the data
        This is the RAG interface - retrieves relevant data and structures response
        """
        question_lower = question.lower()
        
        # Detect question intent
        if any(word in question_lower for word in ['borough', 'which area', 'where']):
            return self._handle_borough_question(question, context)
        
        elif any(word in question_lower for word in ['zip', 'zipcode', 'zip code']):
            return self._handle_zipcode_question(question, context)
        
        elif any(word in question_lower for word in ['noise', 'loud', 'sound']):
            return self._handle_noise_question(question, context)
        
        elif any(word in question_lower for word in ['type', 'complaint', 'common', 'most']):
            return self._handle_complaint_type_question(question, context)
        
        elif any(word in question_lower for word in ['time', 'when', 'hour', 'day']):
            return self._handle_time_question(question, context)
        
        else:
            return self._handle_general_question(question, context)
    
    def _handle_borough_question(self, question: str, context: Optional[Dict]) -> Dict[str, Any]:
        """Handle borough-related questions"""
        borough_summary = self.get_borough_summary()
        
        # Find borough with least complaints
        borough_complaints = borough_summary['borough_breakdown']
        sorted_boroughs = sorted(borough_complaints.items(), key=lambda x: x[1])
        
        return {
            'answer': f"Based on 311 complaint data, {sorted_boroughs[0][0]} has the fewest complaints ({sorted_boroughs[0][1]}), "
                     f"while {sorted_boroughs[-1][0]} has the most ({sorted_boroughs[-1][1]}).",
            'data': borough_summary,
            'sources': ['NYC 311 Service Requests Dataset']
        }
    
    def _handle_zipcode_question(self, question: str, context: Optional[Dict]) -> Dict[str, Any]:
        """Handle ZIP code-related questions"""
        # Try to extract ZIP code from question
        import re
        zip_match = re.search(r'\b\d{5}\b', question)
        
        if zip_match:
            zip_code = zip_match.group()
            risk_analysis = self.analyze_neighborhood_risk(zip_code)
            
            recommendation = risk_analysis.get('recommendation', 'Unable to generate recommendation')
            
            return {
                'answer': f"ZIP code {zip_code} analysis: {recommendation}",
                'data': risk_analysis,
                'sources': ['NYC 311 Service Requests Dataset']
            }
        else:
            # Show top ZIP codes
            top_zips = self.df['incident_zip'].value_counts().head(10)
            return {
                'answer': f"The ZIP codes with most complaints are: {', '.join([str(z) for z in top_zips.head(5).index])}",
                'data': {str(k): v for k, v in top_zips.to_dict().items()},
                'sources': ['NYC 311 Service Requests Dataset']
            }
    
    def _handle_noise_question(self, question: str, context: Optional[Dict]) -> Dict[str, Any]:
        """Handle noise-related questions"""
        noise_df = self.df[self.df['complaint_type'].str.contains('Noise', na=False)]
        
        noise_by_type = noise_df['complaint_type'].value_counts()
        noise_by_borough = noise_df['borough'].value_counts()
        
        total_noise = len(noise_df)
        total_complaints = len(self.df)
        noise_percentage = (total_noise / total_complaints) * 100
        
        return {
            'answer': f"{noise_percentage:.1f}% of all complaints are noise-related ({total_noise:,} out of {total_complaints:,}). "
                     f"The most common noise complaint is '{noise_by_type.index[0]}' with {noise_by_type.values[0]} reports.",
            'data': {
                'total_noise_complaints': int(total_noise),
                'noise_percentage': round(noise_percentage, 2),
                'by_type': noise_by_type.head(5).to_dict(),
                'by_borough': noise_by_borough.to_dict()
            },
            'sources': ['NYC 311 Service Requests Dataset']
        }
    
    def _handle_complaint_type_question(self, question: str, context: Optional[Dict]) -> Dict[str, Any]:
        """Handle questions about complaint types"""
        top_complaints = self.df['complaint_type'].value_counts().head(10)
        
        return {
            'answer': f"The most common complaint type is '{top_complaints.index[0]}' with {top_complaints.values[0]:,} reports. "
                     f"This accounts for {(top_complaints.values[0] / len(self.df) * 100):.1f}% of all complaints.",
            'data': top_complaints.to_dict(),
            'sources': ['NYC 311 Service Requests Dataset']
        }
    
    def _handle_time_question(self, question: str, context: Optional[Dict]) -> Dict[str, Any]:
        """Handle time-related questions"""
        hour_counts = self.df['hour'].value_counts().sort_index()
        day_counts = self.df['day_of_week'].value_counts()
        
        peak_hour = int(hour_counts.idxmax())
        peak_day = day_counts.idxmax()
        
        return {
            'answer': f"Most complaints occur on {peak_day} and peak at {peak_hour}:00. "
                     f"This pattern can help you plan staffing and operations.",
            'data': {
                'peak_hour': peak_hour,
                'peak_day': peak_day,
                'hourly_distribution': hour_counts.to_dict(),
                'daily_distribution': day_counts.to_dict()
            },
            'sources': ['NYC 311 Service Requests Dataset']
        }
    
    def _handle_general_question(self, question: str, context: Optional[Dict]) -> Dict[str, Any]:
        """Handle general questions"""
        summary = self.get_borough_summary()
        
        return {
            'answer': f"This dataset contains {summary['total_complaints']:,} 311 complaints, "
                     f"with {summary['restaurant_relevant_complaints']:,} relevant to restaurant businesses. "
                     f"The most common complaint type is related to noise and quality of life issues.",
            'data': summary,
            'sources': ['NYC 311 Service Requests Dataset']
        }