import re
import json
from datetime import datetime, timedelta

class DiagnosticAssistant:
    """
    Clinical Diagnostic Assistant based on DSM-5 Framework.
    Analyzes text across 7 diagnostic categories, includes a safety net for context,
    and supports longitudinal persistence analysis.
    """

    CATEGORIES = {
        "Anxiety Disorders": {
            "markers": [
                r"\bworr(y|ied|ying)\b", r"\banxious\b", r"\banxiety\b", r"\bpanic\b", r"\bfraid\b", r"\bfear\b",
                r"\bcan't stop thinking\b", r"\bwhat if\b", r"\bavoid\w*\b",
                r"\bheart is? racing\b", r"\bbreathless\b", r"\btight chest\b", r"\btrembling\b"
            ],
            "dsm_link": "GAD, Panic Disorder, Social Anxiety, OCD"
        },
        "Mood Disorders": {
            "markers": [
                r"\bdepress\w*\b", r"\bsad\w*\b", r"\bhopeless\w*\b", r"\bworthless\w*\b",
                r"\bno interest\b", r"\banhedonia\b", r"\btired all the time\b",
                r"\bcan't get out of bed\b", r"\bups and downs\b", r"\bracing (thoughts|mind)\b", r"\bmanic\b"
            ],
            "dsm_link": "Major Depression, Bipolar Disorder"
        },
        "Psychotic Disorders": {
            "markers": [
                r"\bhearing voices\b", r"\bwhispers\b", r"\bwatched\b", r"\bwatching me\b",
                r"\bnot real\b", r"\bhallucination\b", r"\bconspiracy\b", r"\bsecret messages\b",
                r"\bin my head\b", r"\bdelsuion\w*\b"
            ],
            "dsm_link": "Schizophrenia, Hallucinations, Delusions"
        },
        "Eating Disorders": {
            "markers": [
                r"\bfat\b", r"\bweight\b", r"\bcalor\w*\b", r"\bdiet\w*\b", r"\bstarv\w*\b",
                r"\bbinge\b", r"\bpurge\b", r"\bmirror\b", r"\tbody shape\b", r"\too much food\b"
            ],
            "dsm_link": "Anorexia, Bulimia, Body Dysmorphia"
        },
        "Trauma-Related": {
            "markers": [
                r"\bflashback\b", r"\bnightmare\b", r"\btrigger\b", r"\bjumpy\b",
                r"\bhappen\w* again\b", r"\bcan't forget\b", r"\bintrusive\b",
                r"\bstartl\w*\b", r"\btrauma\b"
            ],
            "dsm_link": "PTSD, C-PTSD"
        },
        "Neurodevelopmental": {
            "markers": [
                r"\bcan't focus\b", r"\bdistract\w*\b", r"\bfidget\w*\b", r"\bimpulsive\b",
                r"\bsocial\w* awkward\b", r"\bsensory overload\b", r"\bhyperfixat\w*\b",
                r"\borganiz\w*\b", r"\bprocrastinat\w*\b"
            ],
            "dsm_link": "ADHD, Autism Spectrum markers"
        },
        "Personality Disorders": {
            "markers": [
                r"\babandon\w*\b", r"\bempty\b", r"\bno middle ground\b", r"\ball or nothing\b",
                r"\bhate me\b", r"\bsplitting\b", r"\bspecial treatment\b",
                r"\bavoiding people\b", r"\brejection\b"
            ],
            "dsm_link": "Borderline, Narcissistic, Avoidant patterns"
        }
    }

    CONTEXT_FILTERS = [
        r"\bmovie\b", r"\bfilm\b", r"\bcharacter\b", r"\bstory\b", r"\bbook\b",
        r"\bshow\b", r"\bepisode\b", r"\bfictional\b", r"\bdreamt\b", r"\bin a dream\b"
    ]

    def __init__(self):
        pass

    def _check_context_safety(self, text):
        """Detects if text is likely fictional or media-related."""
        matches = []
        for pattern in self.CONTEXT_FILTERS:
            if re.search(pattern, text, re.IGNORECASE):
                matches.append(pattern)
        
        # If multiple context markers are found, lower the confidence/score
        return len(matches) > 1

    def _get_evidence(self, text, patterns):
        """Extracts matched markers from the text."""
        evidence = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                evidence.extend(list(set(matches)))
        return list(set(evidence))

    def _calculate_score(self, text, category_name, static_factors=None):
        """Calculates risk score for a category."""
        category = self.CATEGORIES[category_name]
        evidence = self._get_evidence(text, category["markers"])
        
        # Base score calculation based on evidence count
        # This is a simplified logic; in production, this would use a proper ML model or weighted patterns
        base_score = min(len(evidence) * 15, 80) 
        
        # Adjust for static factors (family history, previous diagnosis)
        if static_factors and category_name in static_factors:
            base_score += 20
        
        # Cap at 100
        return min(base_score, 100), evidence

    def _get_recommendation(self, score):
        """Maps score to triage action."""
        if score <= 30:
            return "Suggest wellness exercises, meditation, or positive journaling."
        elif score <= 60:
            return "Suggest a 'Mental Health Check-in' quiz or self-help CBT modules."
        elif score <= 85:
            return "Recommend speaking with a licensed therapist or counselor."
        else:
            return "Immediate Intervention: Display emergency hotlines and local clinic info."

    def analyze(self, text_entries, static_factors=None):
        """
        Analyze longitudinal text entries.
        text_entries: List of dicts [{'text': str, 'date': str}]
        """
        if not text_entries:
            return {"error": "No text provided"}

        # Sort entries by date
        sorted_entries = sorted(text_entries, key=lambda x: x['date'])
        all_text = " ".join([e['text'] for e in sorted_entries])
        latest_text = sorted_entries[-1]['text']
        
        is_fictional = self._check_context_safety(all_text)
        
        results = []
        for category_name in self.CATEGORIES:
            score, evidence = self._calculate_score(all_text, category_name, static_factors)
            
            # Context override
            confidence = 100
            if is_fictional:
                score *= 0.3
                confidence = 30
            
            # Persistence check (Simplified: if markers appear in multiple entries over time)
            # Mood disorder persistence (DSM requires 14 days)
            persistence_score = 1.0
            if category_name == "Mood Disorders":
                dates_with_markers = set()
                for entry in sorted_entries:
                    if self._get_evidence(entry['text'], self.CATEGORIES[category_name]['markers']):
                        dates_with_markers.add(entry['date'])
                
                # Check date range
                if len(dates_with_markers) > 1:
                    d_objs = [datetime.strptime(d, "%Y-%m-%d") for d in dates_with_markers]
                    date_range = (max(d_objs) - min(d_objs)).days
                    if date_range >= 14:
                        persistence_score = 1.2 # Boost score for persistent symptoms
            
            final_score = min(score * persistence_score, 100)
            
            results.append({
                "Condition": category_name,
                "Score": round(final_score),
                "Confidence_Score": confidence,
                "Evidence_Detected": evidence,
                "DSM_Reference": self.CATEGORIES[category_name]["dsm_link"],
                "Recommended_Action": self._get_recommendation(final_score)
            })
            
        return {
            "timestamp": datetime.now().isoformat(),
            "longitudinal_entries": len(text_entries),
            "is_context_safe": not is_fictional,
            "diagnostics": results
        }

if __name__ == "__main__":
    # Test block
    assistant = DiagnosticAssistant()
    sample_text = [
        {"text": "I feel so hopeless and can't get out of bed for two weeks.", "date": "2023-10-01"},
        {"text": "Still worthless. My heart is racing and I feel anxious too.", "date": "2023-10-15"}
    ]
    report = assistant.analyze(sample_text)
    print(json.dumps(report, indent=4))
