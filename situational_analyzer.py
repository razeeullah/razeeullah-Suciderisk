"""
Situational Crisis & Peace of Mind (PoM) Analyzer
Version: 2.1.0 (Viva Edition)

Identifies acute life stressors (financial, social, academic) and 
provides immediate grounding and clinical validation.
"""

import re
from datetime import datetime

class SituationalAnalyzer:
    """
    Holistic analyzer for situational stressors.
    Features:
    - Emergency detect (e.g., immediate "fucked up" states).
    - Stressor categorization (Financial, Social, Academic).
    - Peace of Mind (PoM) scoring.
    - Grounding techniques for acute distress.
    """

    # Stressor categories and their patterns
    STRESSORS = {
        "Financial Crisis": {
            "markers": [
                r"\btrading loss\b", r"\blost money\b", r"\bdebt\b", r"\bpoverty\b", 
                r"\bbankrupt\b", r"\bbusiness loss\b", r"\blost everything\b",
                r"\bfinancial trauma\b", r"\bstock market\b", r"\bcrypto\b"
            ],
            "keywords": ["Trading Loss", "Financial Debt", "Economic Stress"],
            "search_query": "How to recover mentally from a major business loss"
        },
        "Social & Trust Issues": {
            "markers": [
                r"\bbetray\w*\b", r"\bfake promises\b", r"\blet me down\b", r"\bcheated\b",
                r"\btoxic relationship\b", r"\balone\b", r"\bisolated\b", r"\bno one to trust\b",
                r"\bbroken trust\b", r"\bhate me\b"
            ],
            "keywords": ["Broken Trust", "Social Isolation", "Betrayal"],
            "search_query": "How to rebuild trust after being let down"
        },
        "Performance Stress": {
            "markers": [
                r"\bfailing\b", r"\bgrade\w*\b", r"\bcareer failure\b", r"\bcan't keep up\b",
                r"\bdropped out\b", r"\bfucked up\b", r"\bf\*cked up\b", r"\bmessed up\b",
                r"\bburnout\b", r"\bnot good enough\b"
            ],
            "keywords": ["Academic Burnout", "Career Pressure", "Self-Worth"],
            "search_query": "Managing academic burnout and self-worth"
        },
        "Existential/Severe": {
            "markers": [
                r"\bdone with life\b", r"\btotal hopelessness\b", r"\bend it\b", 
                r"\balive for what\b", r"\bno point\b", r"\beverything is heavy\b"
            ],
            "keywords": ["Existential Crisis", "Hopelessness", "Life Fatigue"],
            "search_query": "Finding a path forward when everything feels heavy"
        }
    }

    # Curated article links for the "Piece of Mind" toolbox
    ARTICLE_LIBRARY = {
        "Financial Crisis": {
            "title": "Why your net worth isn't your self-worth",
            "url": "https://www.psychologytoday.com/us/blog/the-financial-mind/202104/separating-net-worth-self-worth",
            "description": "Coping strategies for significant financial losses."
        },
        "Social & Trust Issues": {
            "title": "Rebuilding Trust & Setting Boundaries",
            "url": "https://www.helpguide.org/articles/relationships-communication/setting-healthy-boundaries.htm",
            "description": "How to heal after social betrayal or abandonment."
        },
        "Performance Stress": {
            "title": "Academic Pressure & Your Mental Health",
            "url": "https://www.verywellmind.com/how-to-manage-academic-burnout-5211516",
            "description": "Dealing with the weight of student or career expectations."
        },
        "Existential/Severe": {
            "title": "When Life Feels Too Heavy",
            "url": "https://www.mentalhealth.org.uk/explore-mental-health/a-z-topics/hopelessness",
            "description": "Resources for finding light in dark existential moments."
        }
    }

    # Standard clinical grounding for panic or severe distress
    GROUNDING_TECHNIQUES = [
        "**Identify 5-4-3-2-1:**",
        "1. Look at 5 objects near you.",
        "2. Feel 4 textures (e.g., your desk, your shirt).",
        "3. Listen for 3 distinct sounds.",
        "4. Notice 2 smells.",
        "5. Taste 1 thing (or sip water).",
        "**Breathing:** Inhale for 4s, Hold for 4s, Exhale for 6s."
    ]

    def analyze(self, text):
        """
        Main situational analysis logic.
        """
        if not text:
            return None

        # 1. Emergency Detection (Keywords for acute crisis)
        is_emergency = bool(re.search(r"\bf[u\*]{2}ked up (from|with|by) life\b", text, re.IGNORECASE)) or \
                       bool(re.search(r"\bdone with everything\b", text, re.IGNORECASE))

        # 2. Stressor Extraction
        detected = []
        cumulative_stress = 0
        
        for name, cfg in self.STRESSORS.items():
            matches = [m for m in cfg["markers"] if re.search(m, text, re.IGNORECASE)]
            if matches:
                weight = min(len(matches) * 25, 100)
                cumulative_stress += weight
                detected.append({
                    "theme": name,
                    "score": weight,
                    "keywords": cfg["keywords"],
                    "search_query": cfg["search_query"],
                    "article": self.ARTICLE_LIBRARY.get(name)
                })

        # 3. PoM Scoring (Inverse of stress intensity)
        avg_stress = (cumulative_stress / len(self.STRESSORS)) if detected else 0
        pom_score = max(100 - avg_stress, 0)
        
        # Override for emergency states
        if is_emergency:
            pom_score = min(pom_score, 15)

        return {
            "pom_score": round(pom_score),
            "is_emergency": is_emergency,
            "stressors": detected,
            "grounding": self.GROUNDING_TECHNIQUES if is_emergency else None,
            "validation": self._generate_validation(detected, is_emergency)
        }

    def _generate_validation(self, stressors, emergency):
        """Generates a clinically empathetic response."""
        if emergency:
            return "I hear how heavy this feels. It's completely understandable to feel overwhelmed by such intense weight."
        
        if not stressors:
            return "Thank you for sharing your thoughts. I'm here to support you through any situational stressors."
            
        primary = stressors[0]["theme"]
        if primary == "Financial Crisis":
            return "Losing money or facing debt can feel like losing your footing. It's a valid and heavy stressor."
        elif primary == "Social & Trust Issues":
            return "Being let down or feeling isolated hurts deeply. Your feelings around this trust are valid."
        elif primary == "Performance Stress":
            return "Performance pressure is exhausting. Remember that your worth is not defined by a single failure or grade."
        else:
            return "I recognize the complexity of what you're facing. You are handling a lot right now."

if __name__ == "__main__":
    # Internal Unit Test
    situ = SituationalAnalyzer()
    print(situ.analyze("I'm f**ked up from life. Everything is heavy."))
