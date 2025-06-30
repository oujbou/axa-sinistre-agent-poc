"""
Classification utilities for AXA insurance claims processing
Implements hybrid classification: Rules + LLM + Business Logic
"""
import re
from typing import Dict, List, Tuple, Optional
from loguru import logger
from dataclasses import dataclass

try:
    from models.claim_models import ClaimType, Severity, MOTS_CLES_ASSURANCE_FRANCAISE
except ImportError:
    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.models.claim_models import ClaimType, Severity, MOTS_CLES_ASSURANCE_FRANCAISE


@dataclass
class ClassificationResult:
    """Result from rule-based classification"""
    scores: Dict[ClaimType, float]
    keywords_found: Dict[ClaimType, List[str]]
    best_match: Optional[ClaimType]
    confidence: float
    method: str = "rules"


@dataclass
class LLMClassificationResult:
    """Result from LLM semantic classification"""
    type_sinistre: ClaimType
    severite: Severity
    score_confiance: float
    justification: str
    montant_estime: Optional[float]
    flags: Dict[str, bool]
    score_urgence: int
    actions_immediates: List[str]
    method: str = "llm"


class RuleBasedClassifier:
    """
    Rule-based classifier using French insurance keywords and patterns
    Fast and explainable classification for obvious cases
    """

    def __init__(self):
        self.keywords_dict = MOTS_CLES_ASSURANCE_FRANCAISE
        logger.info("Rule-based classifier initialized with French insurance keywords")

    def classify(self, texte: str) -> ClassificationResult:
        """
        Classify text using keyword matching and pattern recognition

        Args:
            texte: Extracted text from OCR

        Returns:
            ClassificationResult: Scores and best match
        """
        texte_lower = texte.lower()
        scores = {}
        found_keywords = {}

        for claim_type, keywords_list in self.keywords_dict.items():
            score = 0
            keywords = []

            # Primary keywords (high weight)
            for keyword in keywords_list:
                if keyword.lower() in texte_lower:
                    # Weight based on keyword specificity
                    weight = 3 if len(keyword.split()) > 1 else 2
                    score += weight
                    keywords.append(keyword)

            # Special patterns for vehicle accidents
            if claim_type == ClaimType.ACCIDENT_AUTO:
                # License plate patterns
                if re.search(r'[A-Z]{2}-\d{3}-[A-Z]{2}', texte):
                    score += 4
                    keywords.append("pattern:plaque_immatriculation")

                # Vehicle references
                if re.search(r'véhicule [AB]', texte, re.IGNORECASE):
                    score += 3
                    keywords.append("pattern:vehicule_constat")

            # Special patterns for water damage
            elif claim_type == ClaimType.DEGAT_DES_EAUX:
                # Plumbing terms
                plumbing_terms = ["canalisation", "robinet", "tuyau", "plomberie"]
                for term in plumbing_terms:
                    if term in texte_lower:
                        score += 2
                        keywords.append(f"plomberie:{term}")

            # Special patterns for theft
            elif claim_type == ClaimType.VOL:
                # Crime-related terms
                crime_terms = ["plainte", "commissariat", "gendarmerie", "police"]
                for term in crime_terms:
                    if term in texte_lower:
                        score += 2
                        keywords.append(f"criminalite:{term}")

            if score > 0:
                scores[claim_type] = score
                found_keywords[claim_type] = keywords

        # Determine best match and confidence
        best_match = None
        confidence = 0.0

        if scores:
            best_match = max(scores, key=scores.get)
            max_score = scores[best_match]

            # Normalize confidence (max realistic score ~15)
            confidence = min(max_score / 15.0, 1.0)

            # Boost confidence if multiple indicators
            if len(found_keywords[best_match]) >= 3:
                confidence = min(confidence * 1.2, 1.0)

        logger.debug(f"Rule-based classification: {best_match} (confidence: {confidence:.2f})")

        return ClassificationResult(
            scores=scores,
            keywords_found=found_keywords,
            best_match=best_match,
            confidence=confidence
        )


class SeverityAnalyzer:
    """
    Analyze claim severity based on multiple indicators
    """

    @staticmethod
    def analyze_severity(texte: str, claim_type: ClaimType, montant: Optional[float] = None) -> Tuple[Severity, float]:
        """
        Determine severity based on text indicators and amount

        Args:
            texte: Description text
            claim_type: Type of claim
            montant: Estimated amount if available

        Returns:
            Tuple[Severity, confidence]: Severity level and confidence
        """
        texte_lower = texte.lower()
        severity_indicators = {
            Severity.CRITIQUE: [
                "urgence", "critique", "grave", "hospitalisation", "blessé grave",
                "explosion", "incendie majeur", "effondrement", "danger"
            ],
            Severity.ELEVEE: [
                "important", "significatif", "dégâts importants", "blessé léger",
                "fuite importante", "dégâts étendus", "investigation"
            ],
            Severity.MODEREE: [
                "moyen", "modéré", "visible", "réparable", "standard"
            ],
            Severity.FAIBLE: [
                "léger", "mineur", "petit", "superficiel", "rayure", "égratignure"
            ]
        }

        # Score based on text indicators
        severity_scores = {}
        for severity, indicators in severity_indicators.items():
            score = sum(1 for indicator in indicators if indicator in texte_lower)
            if score > 0:
                severity_scores[severity] = score

        # Adjust based on amount
        if montant:
            if montant > 10000:
                severity_scores[Severity.CRITIQUE] = severity_scores.get(Severity.CRITIQUE, 0) + 3
            elif montant > 5000:
                severity_scores[Severity.ELEVEE] = severity_scores.get(Severity.ELEVEE, 0) + 2
            elif montant > 1000:
                severity_scores[Severity.MODEREE] = severity_scores.get(Severity.MODEREE, 0) + 1
            else:
                severity_scores[Severity.FAIBLE] = severity_scores.get(Severity.FAIBLE, 0) + 1

        # Claim type specific adjustments
        if claim_type == ClaimType.ACCIDENT_AUTO:
            if any(term in texte_lower for term in ["blessé", "ambulance", "hôpital"]):
                severity_scores[Severity.CRITIQUE] = severity_scores.get(Severity.CRITIQUE, 0) + 2

        elif claim_type == ClaimType.INCENDIE:
            # Fire is usually serious
            severity_scores[Severity.ELEVEE] = severity_scores.get(Severity.ELEVEE, 0) + 2

        # Determine final severity
        if severity_scores:
            best_severity = max(severity_scores, key=severity_scores.get)
            confidence = min(severity_scores[best_severity] / 5.0, 1.0)
        else:
            # Default to moderate if no indicators
            best_severity = Severity.MODEREE
            confidence = 0.3

        return best_severity, confidence


class UrgencyCalculator:
    """
    Calculate urgency score based on multiple factors
    """

    @staticmethod
    def calculate_urgency(
            claim_type: ClaimType,
            severity: Severity,
            flags: Dict[str, bool],
            texte: str
    ) -> int:
        """
        Calculate urgency score from 1 (routine) to 10 (critical)

        Args:
            claim_type: Type of claim
            severity: Severity level
            flags: Special flags (medical, fraud, etc.)
            texte: Original text for additional context

        Returns:
            int: Urgency score 1-10
        """
        base_scores = {
            Severity.FAIBLE: 2,
            Severity.MODEREE: 4,
            Severity.ELEVEE: 6,
            Severity.CRITIQUE: 8
        }

        score = base_scores.get(severity, 4)

        # Urgency boosters
        if flags.get("urgence_medicale", False):
            score += 3

        if flags.get("potentiel_fraude", False):
            score += 1  # Fraud needs investigation but not immediate

        if claim_type in [ClaimType.INCENDIE, ClaimType.CATASTROPHE_NATURELLE]:
            score += 2  # These are typically urgent

        # Text-based urgency indicators
        texte_lower = texte.lower()
        urgent_terms = ["urgence", "urgent", "immédiat", "critique", "danger"]
        if any(term in texte_lower for term in urgent_terms):
            score += 2

        # Time-sensitive indicators
        time_sensitive = ["aujourd'hui", "maintenant", "ce soir", "demain"]
        if any(term in texte_lower for term in time_sensitive):
            score += 1

        # Cap between 1 and 10
        return max(1, min(score, 10))


class BusinessRulesEngine:
    """
    Apply AXA-specific business rules and validation
    """

    @staticmethod
    def apply_business_rules(classification_data: Dict, ocr_metadata: Dict) -> Dict:
        """
        Apply business rules and validate classification

        Args:
            classification_data: Initial classification result
            ocr_metadata: OCR processing metadata

        Returns:
            Dict: Enhanced classification with business rules applied
        """
        result = classification_data.copy()
        warnings = []

        montant = result.get("montant_estime", 0) or 0
        severite = result.get("severite")
        type_sinistre = result.get("type_sinistre")

        # Rule 1: Amount/Severity consistency
        if montant > 15000 and severite != Severity.CRITIQUE.value:
            result["severite"] = Severity.CRITIQUE.value
            warnings.append("Sévérité ajustée à CRITIQUE pour montant > 15k€")

        elif montant > 5000 and severite == Severity.FAIBLE.value:
            result["severite"] = Severity.ELEVEE.value
            warnings.append("Sévérité ajustée à ÉLEVÉE pour montant > 5k€")

        # Rule 2: Mandatory investigation triggers
        investigation_triggers = [
            montant > 5000,
            result.get("flags", {}).get("potentiel_fraude", False),
            type_sinistre == ClaimType.ACCIDENT_AUTO.value and "blessé" in ocr_metadata.get("texte_extrait", ""),
            severite == Severity.CRITIQUE.value
        ]

        if any(investigation_triggers):
            result["necessite_enquete"] = True
            result["score_urgence"] = max(result.get("score_urgence", 1), 6)
            warnings.append("Investigation obligatoire déclenchée")

        # Rule 3: Authorization levels
        if montant < 1000:
            result["niveau_autorisation"] = "agent"
        elif montant < 5000:
            result["niveau_autorisation"] = "superviseur"
        elif montant < 15000:
            result["niveau_autorisation"] = "chef_service"
        else:
            result["niveau_autorisation"] = "direction"

        # Rule 4: Processing timeline
        urgence = result.get("score_urgence", 5)
        if urgence >= 8 or result.get("flags", {}).get("urgence_medicale", False):
            result["delai_traitement_estime"] = "24h"
        elif urgence >= 6:
            result["delai_traitement_estime"] = "3-5j"
        elif urgence >= 4:
            result["delai_traitement_estime"] = "1-2sem"
        else:
            result["delai_traitement_estime"] = "3-4sem"

        # Rule 5: Required expertise
        expertise_required = [
            type_sinistre == ClaimType.INCENDIE.value,
            type_sinistre == ClaimType.CATASTROPHE_NATURELLE.value,
            montant > 10000,
            "technique" in ocr_metadata.get("texte_extrait", "").lower()
        ]

        if any(expertise_required):
            result["necessite_expertise"] = True
            warnings.append("Expertise technique requise")

        # Rule 6: Medical urgency detection
        texte = ocr_metadata.get("texte_extrait", "")
        medical_terms = ["blessé", "hospitalisation", "ambulance", "urgence médicale", "soins"]
        if any(term in texte.lower() for term in medical_terms):
            result["flags"]["urgence_medicale"] = True
            result["score_urgence"] = max(result.get("score_urgence", 1), 8)

        # Rule 7: Immediate actions based on type and severity
        actions = []
        if result.get("flags", {}).get("urgence_medicale", False):
            actions.append("Contacter service médical AXA")
            actions.append("Vérifier prise en charge soins d'urgence")

        if result.get("necessite_enquete", False):
            actions.append("Désigner enquêteur expert")
            actions.append("Collecter pièces justificatives")

        if type_sinistre == ClaimType.VOL.value:
            actions.append("Vérifier dépôt de plainte")
            actions.append("Demander récépissé commissariat")

        if montant > 10000:
            actions.append("Validation direction obligatoire")
            actions.append("Expertise contradictoire recommandée")

        result["actions_immediates"] = actions
        result["business_warnings"] = warnings

        logger.info(f"Business rules applied: {len(warnings)} adjustments made")
        return result