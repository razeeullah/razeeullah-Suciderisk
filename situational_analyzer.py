import re
from datetime import datetime

class SituationalAnalyzer:
    """
    Holistic Crisis Analyst for Life Stressor Events.
    Identifies situational crises and provides grounding, scoring, and resources.
    """

    STRESSORS = {
        "Financial Crisis": {
            "markers": [
                r"\btrading loss\b", r"\blost money\b", r"\bdebt\b", r"\bpoverty\b", 
                r"\bbankrupt\b", r"\bbusiness loss\b", r"\blost everything\b",
                r"\bfinancial trauma\b", r"\bstock market\b", r"\bcrypto\b"
            ],
            "keywords": ["Trading Loss", "Financial Debt", "Business Failure", "Economic Stress"],
            "search_query": "How to recover mentally from a major business loss"
        },
        "Social & Trust Issues": {
            "markers": [
                r"\bbetray\w*\b", r"\bfake promises\b", r"\blet me down\b", r"\bcheated\b",
                r"\btoxic relationship\b", r"\balone\b", r"\bisolated\b", r"\bno one to trust\b",
                r"\bbroken trust\b", r"\bhate me\b"
            ],
            "keywords": ["Broken Trust", "Social Isolation", "Relationship Betrayal", "Toxic Environment"],
            "search_query": "How to rebuild trust after being let down"
        },
        "Performance Stress": {
            "markers": [
                r"\bfailing\b", r"\bgrade\w*\b", r"\bcareer failure\b", r"\bcan't keep up\b",
                r"\bdropped out\b", r"\bfucked up\b", r"\bf*cked up\b", r"\bmessed up\b",
                r"\bburnout\b", r"\bnot good enough\b"
            ],
            "keywords": ["Academic Burnout", "Career Failure", "Performance Anxiety", "Self-Worth Issues"],
            "search_query": "Managing academic burnout and self-worth"
        },
        "Existential/Severe": {
            "markers": [
                r"\bdone with life\b", r"\btotal hopelessness\b", r"\bend it\b", 
                r"\balive for what\b", r"\bno point\b", r"\beverything is heavy\b"
            ],
            "keywords": ["Existential Crisis", "Total Hopelessness", "Life Fatigue", "Severe Distress"],
            "search_query": "Finding a path forward when everything feels heavy"
        }
    }

    ARTICLE_LIBRARY = {
        "Financial Crisis": {
            "title": "Why your bank account isn't your identity",
            "url": "https://www.psychologytoday.com/us/blog/the-financial-mind/202104/separating-net-worth-self-worth",
            "description": "Learn to cope with the psychology of trading losses and separate net worth from self-worth."
        },
        "Social & Trust Issues": {
            "title": "Setting Boundaries & Rebuilding Trust",
            "url": "https://www.helpguide.org/articles/relationships-communication/setting-healthy-boundaries.htm",
            "description": "How to rebuild trust after being let down and identifying narcissistic patterns."
        },
        "Performance Stress": {
            "title": "Managing Burnout and Academic Pressure",
            "url": "https://www.verywellmind.com/how-to-manage-academic-burnout-5211516",
            "description": "Tools for managing academic burnout and maintaining self-worth during failures."
        },
        "Existential/Severe": {
            "title": "Finding a Path Forward",
            "url": "https://www.mentalhealth.org.uk/explore-mental-health/a-z-topics/hopelessness",
            "description": "Resources for finding a path forward when everything feels heavy."
        }
    }

    GROUNDING_TECHNIQUES = [
        "**5-4-3-2-1 Technique:**",
        "1. Identify 5 things you can see around you.",
        "2. Identify 4 things you can touch.",
        "3. Identify 3 things you can hear.",
        "4. Identify 2 things you can smell.",
        "5. Identify 1 thing you can taste.",
        "Focus on your breath: Inhale for 4, Hold for 4, Exhale for 4."
    ]

    def analyze(self, text):
        if not text:
            return None

        detected_stressors = []
        is_fucked_up = False
        
        # Check for immediate grounding trigger
        if re.search(r"\bf[u\*]{2}ked up (from|with|by) life\b", text, re.IGNORECASE) or \
           re.search(r"\bdone with everything\b", text, re.IGNORECASE):
            is_fucked_up = True

        total_stress_score = 0
        
        for name, data in self.STRESSORS.items():
            matches = []
            for pattern in data["markers"]:
                if re.search(pattern, text, re.IGNORECASE):
                    matches.append(pattern)
            
            if matches:
                # Calculate stress contribution
                score = min(len(matches) * 25, 100)
                total_stress_score += score
                
                detected_stressors.append({
                    "theme": name,
                    "score": score,
                    "keywords": data["keywords"],
                    "search_query": data["search_query"],
                    "article": self.ARTICLE_LIBRARY.get(name)
                })

        # Calculate Peace of Mind (POM) score
        # 100 is calm, 0 is severe crisis
        avg_stress = (total_stress_score / len(self.STRESSORS)) if detected_stressors else 0
        pom_score = max(100 - avg_stress, 0)
        
        # If is_fucked_up is true, POM score is likely very low or needs immediate attention
        if is_fucked_up and pom_score > 20:
            pom_score = 20

        return {
            "pom_score": round(pom_score),
            "is_emergency": is_fucked_up,
            "stressors": detected_stressors,
            "grounding": self.GROUNDING_TECHNIQUES if is_fucked_up else None,
            "validation": self._generate_validation(detected_stressors, is_fucked_up)
        }

    def _generate_validation(self, stressors, is_emergency):
        if is_emergency:
            return "I hear how much weight you're carrying right now. It sounds like things feel completely overwhelming."
        
        if not stressors:
            return "Thank you for sharing. I'm here to listen to whatever life stressors you might be facing."
            
        primary = stressors[0]["theme"]
        if primary == "Financial Crisis":
            return "I hear how much weight you're carrying with those financial challenges. Losing money or facing business stress is incredibly draining."
        elif primary == "Social & Trust Issues":
            return "It sounds like you've been let down by someone you trusted. Dealing with betrayal or isolation is deeply painful."
        elif primary == "Performance Stress":
            return "Dealing with failure or pressure in your career/studies can make you feel stuck. It's a lot to handle."
        else:
            return "I hear how heavy things feel right now. It's understandable to feel this way when everything seems to be going wrong."
# Dummy change to trigger redeploy
