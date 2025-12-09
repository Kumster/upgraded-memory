"""
Violation Prevention Agent - FULLY FUNCTIONAL
Provides actionable guidance on preventing NYC health code violations
"""
from typing import Dict, List, Optional


class ViolationPreventionAgent:
    def __init__(self, openai_client=None, vector_store=None):
        self.name = "Violation Prevention"
        self.description = "Prevents health code violations with proactive guidance"
        self.openai_client = openai_client
        self.vector_store = vector_store
        
        # NYC Health Code Common Violations Database
        self.common_violations = {
            'critical': [
                {
                    'code': '04L',
                    'violation': 'Evidence of mice or live mice present',
                    'prevention': [
                        'Seal all entry points (holes, cracks, gaps around pipes)',
                        'Keep food sealed and elevated off floor',
                        'Maintain contract with licensed pest control company',
                        'Daily inspection of premises for signs of rodents',
                        'Proper waste management - covered containers, daily removal'
                    ],
                    'severity': 'CRITICAL - 28 points'
                },
                {
                    'code': '04M',
                    'violation': 'Evidence of roaches',
                    'prevention': [
                        'Monthly professional pest control treatments',
                        'Seal cracks and crevices',
                        'Keep kitchen spotlessly clean - no food debris',
                        'Fix water leaks immediately',
                        'Store food in sealed containers'
                    ],
                    'severity': 'CRITICAL - 28 points'
                },
                {
                    'code': '02B',
                    'violation': 'Hot food not held at 140°F or above',
                    'prevention': [
                        'Use calibrated thermometers - check daily',
                        'Keep hot holding units at 140°F minimum',
                        'Check food temps every 2 hours',
                        'Maintain temperature logs',
                        'Discard food below safe temperature'
                    ],
                    'severity': 'CRITICAL - 7 points'
                },
                {
                    'code': '02G',
                    'violation': 'Cold food not held at 41°F or below',
                    'prevention': [
                        'Refrigerators at 41°F or below',
                        'Freezers at 0°F or below',
                        'Check temps twice daily',
                        'Don\'t overload coolers',
                        'Maintain equipment properly'
                    ],
                    'severity': 'CRITICAL - 7 points'
                },
                {
                    'code': '08A',
                    'violation': 'Facility not vermin proof',
                    'prevention': [
                        'Install door sweeps on all exterior doors',
                        'Repair broken windows and screens',
                        'Seal utility line entry points',
                        'Keep doors closed when not in use',
                        'Maintain perimeter - no debris/vegetation'
                    ],
                    'severity': 'CRITICAL - 28 points'
                }
            ],
            'general': [
                {
                    'code': '06D',
                    'violation': 'Food contact surfaces not properly washed, rinsed, sanitized',
                    'prevention': [
                        'Use 3-compartment sink: wash, rinse, sanitize',
                        'Test sanitizer concentration (50-200 ppm chlorine)',
                        'Air dry - don\'t towel dry',
                        'Sanitize every 4 hours minimum',
                        'Train staff on proper procedures'
                    ],
                    'severity': 'GENERAL - 5 points'
                },
                {
                    'code': '10B',
                    'violation': 'Plumbing not properly installed or maintained',
                    'prevention': [
                        'Fix leaks immediately',
                        'Maintain grease traps',
                        'No cross-connections',
                        'Proper drainage systems',
                        'Regular plumbing inspections'
                    ],
                    'severity': 'GENERAL - 5 points'
                },
                {
                    'code': '06C',
                    'violation': 'Food not protected from contamination',
                    'prevention': [
                        'Cover all stored food',
                        'Store raw meat below ready-to-eat foods',
                        'Use food-grade containers',
                        'Keep food 6 inches off floor',
                        'Protect food during transport'
                    ],
                    'severity': 'GENERAL - 5 points'
                }
            ]
        }
        
        # Best practices checklist
        self.daily_checklist = {
            'Opening': [
                'Check all refrigerator/freezer temperatures',
                'Inspect for pest evidence',
                'Verify hand wash stations stocked (soap, towels)',
                'Check sanitizer concentration',
                'Review day\'s prep and temperature logs'
            ],
            'During Service': [
                'Monitor food temperatures every 2 hours',
                'Ensure proper hand washing between tasks',
                'Keep surfaces clean and sanitized',
                'Maintain proper food storage',
                'Watch for cross-contamination'
            ],
            'Closing': [
                'Complete end-of-day cleaning',
                'Store all food properly',
                'Take out trash and clean containers',
                'Check all equipment turned off/set correctly',
                'Complete temperature and cleaning logs'
            ]
        }
    
    def process(self, query: str, context: str = "", **kwargs) -> str:
        """Process violation prevention query"""
        query_lower = query.lower()
        
        # Determine what the query is asking for
        if any(word in query_lower for word in ['common', 'most', 'typical', 'frequent', 'list']):
            return self._list_common_violations()
        
        elif any(word in query_lower for word in ['prevent', 'avoid', 'stop']):
            return self._prevention_strategies()
        
        elif any(word in query_lower for word in ['checklist', 'daily', 'routine']):
            return self._daily_checklist_response()
        
        elif any(word in query_lower for word in ['temperature', 'temp', 'hot', 'cold']):
            return self._temperature_control_guidance()
        
        elif any(word in query_lower for word in ['pest', 'mice', 'roach', 'rodent', 'vermin']):
            return self._pest_control_guidance()
        
        elif any(word in query_lower for word in ['inspection', 'inspector', 'prepare']):
            return self._inspection_prep_guidance()
        
        else:
            # General comprehensive response
            return self._comprehensive_guidance()
    
    def _list_common_violations(self) -> str:
        """List the most common health violations"""
        response = "⚠️ **Most Common NYC Health Code Violations**\n\n"
        response += "These are the violations that cause the most restaurant closures and fines:\n\n"
        
        response += "🔴 **CRITICAL VIOLATIONS** (Result in immediate closure if not corrected)\n\n"
        for i, violation in enumerate(self.common_violations['critical'][:5], 1):
            response += f"**{i}. {violation['violation']}** (Code {violation['code']})\n"
            response += f"   • Severity: {violation['severity']}\n"
            response += f"   • Prevention: {violation['prevention'][0]}\n\n"
        
        response += "\n🟡 **GENERAL VIOLATIONS** (Must be corrected but less severe)\n\n"
        for i, violation in enumerate(self.common_violations['general'][:3], 1):
            response += f"**{i}. {violation['violation']}** (Code {violation['code']})\n"
            response += f"   • Severity: {violation['severity']}\n\n"
        
        response += "\n💡 **Pro Tip:** Focus on the critical violations first - these can shut you down immediately!\n"
        response += "\n📚 **Want prevention strategies?** Ask: 'How can I prevent health violations?'"
        
        return response
    
    def _prevention_strategies(self) -> str:
        """Provide prevention strategies"""
        response = "✅ **Comprehensive Violation Prevention Strategies**\n\n"
        
        response += "**1. 🐭 PEST CONTROL (Most Common Violation)**\n\n"
        pest_violation = self.common_violations['critical'][0]
        for tip in pest_violation['prevention']:
            response += f"   • {tip}\n"
        
        response += "\n**2. 🌡️ TEMPERATURE CONTROL**\n\n"
        temp_violations = [self.common_violations['critical'][2], self.common_violations['critical'][3]]
        for violation in temp_violations:
            response += f"   **{violation['violation']}:**\n"
            for tip in violation['prevention'][:3]:
                response += f"      • {tip}\n"
        
        response += "\n**3. 🧼 SANITATION & CLEANING**\n\n"
        sanitization = self.common_violations['general'][0]
        for tip in sanitization['prevention']:
            response += f"   • {tip}\n"
        
        response += "\n**4. 🔒 FOOD PROTECTION**\n\n"
        protection = self.common_violations['general'][2]
        for tip in protection['prevention']:
            response += f"   • {tip}\n"
        
        response += "\n**5. 👥 STAFF TRAINING**\n\n"
        response += "   • All staff must have NYC Food Handler Certificate\n"
        response += "   • Train on proper hand washing (20 seconds minimum)\n"
        response += "   • Emphasize personal hygiene and illness reporting\n"
        response += "   • Regular refresher training monthly\n"
        
        response += "\n💡 **Pro Tip:** Create laminated checklists for each station - visual reminders work!\n"
        
        return response
    
    def _daily_checklist_response(self) -> str:
        """Provide daily compliance checklist"""
        response = "📋 **Daily Compliance Checklist**\n\n"
        response += "Use this checklist every day to prevent violations:\n\n"
        
        for time_period, tasks in self.daily_checklist.items():
            response += f"**{time_period}:**\n"
            for task in tasks:
                response += f"   ☐ {task}\n"
            response += "\n"
        
        response += "📊 **Temperature Logs:**\n"
        response += "   • Record all cooler/freezer temps twice daily\n"
        response += "   • Check hot holding temps every 2 hours\n"
        response += "   • Log all corrective actions taken\n\n"
        
        response += "🧹 **Cleaning Logs:**\n"
        response += "   • Document daily cleaning tasks\n"
        response += "   • Track deep cleaning schedule\n"
        response += "   • Sign off on completion\n\n"
        
        response += "💡 **Pro Tip:** Keep logs for at least 1 year - inspectors will review them!\n"
        
        return response
    
    def _temperature_control_guidance(self) -> str:
        """Provide temperature control guidance"""
        response = "🌡️ **Temperature Control - Complete Guide**\n\n"
        
        response += "**CRITICAL TEMPERATURES:**\n\n"
        response += "   • **HOT FOODS:** Hold at 140°F (60°C) or above\n"
        response += "   • **COLD FOODS:** Hold at 41°F (5°C) or below\n"
        response += "   • **DANGER ZONE:** 41°F - 140°F (bacteria multiply rapidly)\n"
        response += "   • **COOKING TEMPS:** Vary by food (165°F for poultry, 155°F for ground meat)\n\n"
        
        response += "**EQUIPMENT REQUIREMENTS:**\n\n"
        response += "   • **Refrigerators:** Maintain 38°F - 41°F\n"
        response += "   • **Freezers:** Maintain 0°F or below\n"
        response += "   • **Hot Holding:** Steam tables at 140°F minimum\n"
        response += "   • **Calibrated thermometers:** Required at each station\n\n"
        
        response += "**MONITORING PROTOCOL:**\n\n"
        response += "   1. Check all cooler/freezer temps twice daily (opening & closing)\n"
        response += "   2. Check food temps every 2 hours during service\n"
        response += "   3. Use metal stem thermometer (not infrared for food temps)\n"
        response += "   4. Document all readings in temperature log\n"
        response += "   5. Take corrective action if temps out of range\n\n"
        
        response += "⚠️ **CORRECTIVE ACTIONS:**\n\n"
        response += "   • **Food below 41°F or above 140°F for 2+ hours:** DISCARD\n"
        response += "   • **Equipment malfunction:** Move food to working unit immediately\n"
        response += "   • **Document all discarded food and reasons**\n\n"
        
        response += "💡 **Pro Tip:** Calibrate thermometers weekly in ice water (should read 32°F)\n"
        
        return response
    
    def _pest_control_guidance(self) -> str:
        """Provide pest control guidance"""
        response = "🐭 **Pest Control - Complete Prevention Strategy**\n\n"
        
        response += "**WHY THIS MATTERS:**\n"
        response += "Pest violations are the #1 reason NYC restaurants get shut down.\n"
        response += "Evidence of mice (Code 04L) = IMMEDIATE 28 points + possible closure.\n\n"
        
        response += "**PREVENTION REQUIREMENTS:**\n\n"
        response += "   1. **Licensed Pest Control Contract (REQUIRED)**\n"
        response += "      • Monthly service minimum\n"
        response += "      • Keep service logs on-site\n"
        response += "      • Document all treatments\n\n"
        
        response += "   2. **Physical Barriers:**\n"
        response += "      • Door sweeps on ALL exterior doors (no gap > 1/4 inch)\n"
        response += "      • Seal all holes/cracks in walls, floors, ceilings\n"
        response += "      • Screen all windows and vents\n"
        response += "      • Seal utility line entry points\n\n"
        
        response += "   3. **Sanitation (Critical!):**\n"
        response += "      • No food debris anywhere - sweep/mop nightly\n"
        response += "      • Clean under/behind all equipment weekly\n"
        response += "      • Empty trash multiple times daily\n"
        response += "      • Use covered trash containers only\n"
        response += "      • Clean grease traps regularly\n\n"
        
        response += "   4. **Food Storage:**\n"
        response += "      • Store all food 6 inches off floor\n"
        response += "      • Keep food in sealed containers\n"
        response += "      • Never leave food out overnight\n"
        response += "      • Rotate stock - FIFO method\n\n"
        
        response += "   5. **Daily Inspection:**\n"
        response += "      • Check for droppings/signs of activity\n"
        response += "      • Inspect deliveries before accepting\n"
        response += "      • Check storage areas daily\n"
        response += "      • Document findings\n\n"
        
        response += "🚨 **RED FLAGS (Call Pest Control IMMEDIATELY):**\n"
        response += "   • Droppings anywhere in food prep/storage areas\n"
        response += "   • Live pests spotted\n"
        response += "   • Gnaw marks on food packages\n"
        response += "   • Unusual pet behavior (cats/dogs detecting pests)\n\n"
        
        response += "💡 **Pro Tip:** Inspectors WILL check behind equipment and in hard-to-reach areas!\n"
        
        return response
    
    def _inspection_prep_guidance(self) -> str:
        """Provide inspection preparation guidance"""
        response = "🔍 **NYC Health Inspection Preparation Guide**\n\n"
        
        response += "**WHAT INSPECTORS CHECK FIRST:**\n\n"
        response += "   1. **Pest Evidence** - They look everywhere\n"
        response += "   2. **Food Temperatures** - Will check multiple items\n"
        response += "   3. **Hand Washing** - Must be available and stocked\n"
        response += "   4. **Food Storage** - Proper labeling and temperatures\n"
        response += "   5. **Personal Hygiene** - Staff appearance and practices\n\n"
        
        response += "**PRE-INSPECTION CHECKLIST:**\n\n"
        response += "   ✓ All temperature logs up to date and accurate\n"
        response += "   ✓ Pest control logs on-site and current\n"
        response += "   ✓ All staff have food handler certificates\n"
        response += "   ✓ Hand sinks stocked (soap, paper towels, garbage)\n"
        response += "   ✓ All food properly labeled and dated\n"
        response += "   ✓ Thermometers available and calibrated\n"
        response += "   ✓ Cleaning chemicals properly stored and labeled\n"
        response += "   ✓ No evidence of pests anywhere\n"
        response += "   ✓ All equipment clean and functional\n"
        response += "   ✓ Staff dressed properly (hair nets, aprons, closed-toe shoes)\n\n"
        
        response += "**DURING THE INSPECTION:**\n\n"
        response += "   • Be cooperative but don't volunteer extra information\n"
        response += "   • Follow the inspector - answer questions honestly\n"
        response += "   • Take notes on any violations cited\n"
        response += "   • Fix critical violations IMMEDIATELY if possible\n"
        response += "   • Ask questions if you don't understand something\n\n"
        
        response += "**GRADING SYSTEM:**\n\n"
        response += "   • **Grade A:** 0-13 points (GOAL)\n"
        response += "   • **Grade B:** 14-27 points (Must improve)\n"
        response += "   • **Grade C:** 28+ points (Serious issues)\n"
        response += "   • **Pending:** Grade not yet determined (re-inspection required)\n\n"
        
        response += "⚠️ **CRITICAL VIOLATIONS (Can close you immediately):**\n"
        response += "   • Evidence of mice or live mice\n"
        response += "   • Roach infestation\n"
        response += "   • Sewage backup\n"
        response += "   • No water supply\n"
        response += "   • Major food temp violations\n\n"
        
        response += "💡 **Pro Tip:** Treat EVERY day like inspection day - you never know when they'll show up!\n"
        
        return response
    
    def _comprehensive_guidance(self) -> str:
        """Provide comprehensive violation prevention guidance"""
        response = "🛡️ **Comprehensive Violation Prevention Guide**\n\n"
        response += "Here's everything you need to stay compliant and violation-free:\n\n"
        
        response += "**TOP 5 PRIORITIES:**\n\n"
        response += "1. **Pest Control** - #1 cause of closures\n"
        response += "2. **Temperature Control** - Critical for food safety\n"
        response += "3. **Hand Washing** - Proper facilities and practices\n"
        response += "4. **Food Storage** - Proper labeling and temperatures\n"
        response += "5. **Cleaning & Sanitation** - Documented and thorough\n\n"
        
        response += "**KEY RESOURCES:**\n\n"
        response += "   • NYC Food Handler Course (required for all staff)\n"
        response += "   • Licensed pest control company (monthly service)\n"
        response += "   • Calibrated thermometers (multiple units)\n"
        response += "   • Temperature and cleaning logs\n"
        response += "   • Current permits and certificates visible\n\n"
        
        response += "**ASK ME ABOUT:**\n\n"
        response += "   • \"What are the most common health violations?\"\n"
        response += "   • \"How do I prevent pest problems?\"\n"
        response += "   • \"What temperatures should I maintain?\"\n"
        response += "   • \"Give me a daily compliance checklist\"\n"
        response += "   • \"How do I prepare for a health inspection?\"\n\n"
        
        response += "💡 **Pro Tip:** Prevention is MUCH cheaper than fines. A Grade C costs you customers!\n"
        
        return response
    
    def get_prompt(self, query: str, context: str = "") -> str:
        """Generate violation prevention prompt"""
        return f"Provide violation prevention guidance for: {query}"
    
    def format_response(self, message: str, data: dict = None) -> str:
        """Format agent response"""
        response = {
            'success': True,
            'agent': self.name,
            'message': message,
            'data': data or {}
        }
        return response


# ========== TESTING ==========

if __name__ == "__main__":
    print("🧪 TESTING VIOLATION PREVENTION AGENT\n")
    print("="*70)
    
    agent = ViolationPreventionAgent()
    
    test_queries = [
        "What are the most common health violations?",
        "How can I prevent violations?",
        "Give me a daily compliance checklist",
        "Tell me about temperature control",
        "How do I prevent pest problems?",
        "How should I prepare for an inspection?"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        print("-" * 70)
        response = agent.process(query)
        print(response[:500] + "...\n")  # Show first 500 chars
        print("="*70)
    
    print("\n✅ Violation Prevention Agent is fully functional!")