"""
Kitchen AI Compass - Multi-Agent System
"""

# Only import what works
try:
    from .base_agent import BaseAgent
except:
    BaseAgent = None

try:
    from .intent_router_agent import IntentRouterAgent
except:
    IntentRouterAgent = None

try:
    from .location_risk_agent import LocationRiskAgent
except:
    LocationRiskAgent = None

try:
    from .violation_prevention_agent import ViolationPreventionAgent
except:
    ViolationPreventionAgent = None

try:
    from .compliance_guide_agent import ComplianceGuideAgent
except:
    ComplianceGuideAgent = None

try:
    from .strategic_advisor_agent import StrategicAdvisorAgent
except:
    StrategicAdvisorAgent = None

# Don't import ComplaintAnalyzerAgent here - it will be imported directly

__all__ = [
    'BaseAgent',
    'IntentRouterAgent',
    'LocationRiskAgent',
    'ViolationPreventionAgent',
    'ComplianceGuideAgent',
    'StrategicAdvisorAgent'
]

__version__ = '1.0.0'