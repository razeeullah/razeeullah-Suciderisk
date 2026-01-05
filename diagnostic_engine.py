"""
Clinical Diagnostic Engine (DSM-5 Framework)
Version: 2.1.0 (Viva Edition)

A pattern-matching engine designed to detect clinical markers across 7 major 
DSM-5 diagnostic categories. It supports longitudinal analysis (persistence 
over time) and includes fictional context filtering.
"""

import re
import json
from datetime import datetime, timedelta

class DiagnosticAssistant:
    """
    Analyzes patient journal entries for clinical indicators.
    Features:
    - Pattern matching for 7 DSM-5 categories.
    - Context safety filtering (detects fictional/media references).
    - Longitudinal persistence tracking (e.g., the 14-day rule for MDD).
    """

    # Diagnostic categories and their respective linguistic markers
    CATEGORIES = {
        "Anxiety Disorders": {
            "markers": [
                r"\bworr(y|ied|ying)\b", r"\banxious\b", r"\banxiety\b", r"\bpanic\b", r"\bfraid\b", r"\bfear\b",
                r"\bcan't stop thinking\b", r"\bwhat if\b", r"\bavoid\w*\b",
                r"\bheart is? racing\b", r"\bbreathless\b", r"\btight chest\b", r"\btrembling\b"
            ],
            "dsm_link": "GAD, Panic Disorder, OCD"
        },
        "Mood Disorders": {
            "markers": [
                r"\bdepress\w*\b", r"\bsad\w*\b", r"\bhopeless\w*\b", r"\bworthless\w*\b",
                r"\bno interest\b", r"\banhedonia\b", r"\btired all the time\b",
                r"\bcan't get out of bed\b", r"\bups and downs\b", r"\bracing (thoughts|mind)\b", r"\bmanic\b"
            ],
            "dsm_link": "Major Depression (MDD), Bipolar Disorder"
        },
        "Psychotic Disorders": {
            "markers": [
                r"\bhearing voices\b", r"\bwhispers\b", r"\bwatched\b", r"\bwatching me\b",
                r"\bnot real\b", r"\bhallucination\b", r"\bconspiracy\b", r"\bsecret messages\b",
                r"\bin my head\b", r"\bdelsuion\w*\b"
            ],
            "dsm_link": "Schizophrenia, Psychosis"
        },
        "Eating Disorders": {
            "markers": [
                r"\bfat\b", r"\bweight\b", r"\bcalor\w*\b", r"\bdiet\w*\b", r"\bstarv\w*\b",
                r"\bbinge\b", r"\bpurge\b", r"\bmirror\b", r"\tbody shape\b", r"\too much food\b"
            ],
            "dsm_link": "Anorexia, Bulimia, Binge Eating"
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
            "dsm_link": "ADHD, ASD"
        },
        "Personality Disorders": {
            "markers": [
                r"\babandon\w*\b", r"\bempty\b", r"\bno middle ground\b", r"\ball or nothing\b",
                r"\bhate me\b", r"\bsplitting\b", r"\bspecial treatment\b",
                r"\bavoiding people\b", r"\brejection\b"
            ],
            "dsm_link": "BPD, NPD, Avoidant PD"
        }
    }

    # Markers that suggest text might be fictional (e.g., from a movie)
    CONTEXT_FILTERS = [
        r"\bmovie\b", r"\bfilm\b", r"\bcharacter\b", r"\bstory\b", r"\bbook\b",
        r"\bshow\b", r"\bepisode\b", r"\bfictional\b", r"\bdreamt\b", r"\bin a dream\b"
    ]

    def _check_context_safety(self, text):
        """Returns True if multiple fictional context markers are present."""
        matches = [p for p in self.CONTEXT_FILTERS if re.search(p, text, re.IGNORECASE)]
        return len(matches) > 1

    def _get_evidence(self, text, patterns):
        """Finds unique matches for diagnostic patterns in the text."""
        evidence = []
        for p in patterns:
            found = re.findall(p, text, re.IGNORECASE)
            if found:
                evidence.extend(list(set(found)))
        return list(set(evidence))

    def _get_recommendation(self, score):
        """Converts numerical score to clinical guidance."""
        if score <= 30:
            return "General Wellness: Suggest mindfulness and routine tracking."
        elif score <= 60:
            return "Moderate Risk: Recommend structured self-help and psychoeducation."
        elif score <= 85:
            return "Significant Risk: Professional evaluation by a therapist requested."
        else:
            return "CRITICAL: Immediate clinical intervention required."

    def analyze(self, entries, static_factors=None):
        """
        Processes a sequence of journal entries to generate a diagnostic report.
        
        Args:
            entries: List of dicts [{'text': str, 'date': str}]
            static_factors: Optional dict of pre-existing risk factors.
        """
        if not entries:
            return {"error": "Dataset empty"}

        # Combine text for holistic view, latest for current state
        full_text = " ".join([e['text'] for e in entries])
        is_fictional = self._check_context_safety(full_text)
        
        results = []
        for cat_name, config in self.CATEGORIES.items():
            # 1. Base Score (weighted by evidence count)
            evidence = self._get_evidence(full_text, config["markers"])
            score = min(len(evidence) * 15, 80) 
            
            # 2. Static Factor Adjustment
            if static_factors and cat_name in static_factors:
                score += 20
            
            # 3. Longitudinal Scaling (The 'Persistence' Multiplier)
            multiplier = 1.0
            if cat_name == "Mood Disorders" and len(entries) > 1:
                # Check for the 14-day persistence rule found in DSM-5 for MDD
                dates = sorted([datetime.strptime(e['date'], "%Y-%m-%d") for e in entries 
                                if self._get_evidence(e['text'], config['markers'])])
                if len(dates) >= 2 and (dates[-1] - dates[0]).days >= 14:
                    multiplier = 1.25 # Escalate score for chronic symptoms
            
            final_score = min(score * multiplier, 100)
            confidence = 90 if not is_fictional else 30
            
            # 4. Contextual Downgrade
            if is_fictional:
                final_score *= 0.4

            results.append({
                "Condition": cat_name,
                "Score": round(final_score),
                "Confidence": confidence,
                "Evidence_Detected": evidence,
                "DSM_Link": config["dsm_link"],
                "Recommended_Action": self._get_recommendation(final_score)
            })
            
        return {
            "analysis_id": f"CLN-{datetime.now().strftime('%Y%m%d%H%M')}",
            "is_context_safe": not is_fictional,
            "diagnostics": results
        }

if __name__ == "__main__":
    # Internal Unit Test
    cln = DiagnosticAssistant()
    test_data = [
        {"text": "I feel so sad and lonely.", "date": "2024-01-01"},
        {"text": "Truly hopeless. I haven't slept in 15 days.", "date": "2024-01-16"}
    ]
    print(json.dumps(cln.analyze(test_data), indent=2))
