"""
Minimal Report Agent for POC demonstration
Simple implementation - JSON and Text export only
"""
from typing import Dict, Any, Optional
import time
import json
from datetime import datetime
from loguru import logger

# Fix imports to work with project structure
try:
    from agents.base_agent import BaseAgent
    from models.claim_models import ClaimType, Severity, ClaimReport
except ImportError:
    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.agents.base_agent import BaseAgent
    from src.models.claim_models import ClaimType, Severity, ClaimReport


class ReportAgent(BaseAgent):
    """
    Report Agent

    Capabilities:
    - Basic report structure
    - JSON and Text export
    - Essential information compilation
    """

    def __init__(self, llm_client, config: Dict[str, Any] = None):
        super().__init__("Report_Agent", llm_client, config)
        logger.info("Report Agent initialized - minimal version with JSON and Text export")

    def reason(self, input_data: Dict[str, Any]) -> str:
        """
        REASONING: Simple strategy determination
        """
        classification = input_data.get("resultat_classification", {})
        type_sinistre = classification.get("type_sinistre", "inconnu")

        return f"Génération rapport JSON et texte pour sinistre type {type_sinistre}."

    def execute(self, input_data: Dict[str, Any], reasoning: str) -> Dict[str, Any]:
        """
        EXECUTION: Generate basic report data
        """
        start_time = time.time()

        try:
            classification = input_data.get("resultat_classification", {})
            ocr = input_data.get("resultat_ocr", {})

            if not classification:
                raise ValueError("Pas de données de classification")

            # Create simple report structure
            report_data = {
                "id_rapport": f"AXA-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                "date_generation": datetime.now().isoformat(),
                "type_sinistre": classification.get("type_sinistre", "inconnu"),
                "severite": classification.get("severite", "moderee"),
                "resume": self._create_summary(classification),
                "details": {
                    "confiance_classification": classification.get("score_confiance", 0),
                    "montant_estime": classification.get("montant_estime"),
                    "urgence": classification.get("score_urgence", 5),
                    "mots_cles": classification.get("mots_cles_trouves", [])
                },
                "actions_recommandees": classification.get("actions_immediates", []),
                "informations_ocr": {
                    "confiance_ocr": ocr.get("score_confiance", 0),
                    "longueur_texte": len(ocr.get("texte_extrait", ""))
                },
                "delai_traitement": classification.get("delai_traitement_estime", "5-10 jours"),
                "temps_generation": time.time() - start_time
            }

            return {"rapport_data": report_data}

        except Exception as e:
            logger.error(f"Erreur génération rapport : {str(e)}")
            return {"erreur": str(e)}

    def _create_summary(self, classification: Dict) -> str:
        """Create basic summary text"""
        type_sinistre = classification.get("type_sinistre", "inconnu").replace("_", " ").title()
        severite = classification.get("severite", "moderee").upper()
        montant = classification.get("montant_estime")

        summary = f"Sinistre de type {type_sinistre} avec sévérité {severite}."

        if montant:
            summary += f" Montant estimé: {montant:,.0f}€."

        urgence = classification.get("score_urgence", 5)
        if urgence >= 7:
            summary += " Traitement prioritaire requis."

        return summary

    def act(self, execution_output: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        ACTION: Generate JSON and Text exports
        """
        if "erreur" in execution_output:
            return None

        rapport_data = execution_output.get("rapport_data", {})

        # Generate JSON export
        json_content = json.dumps(rapport_data, ensure_ascii=False, indent=2)
        json_export = {
            "format": "application/json",
            "contenu": json_content,
            "taille": len(json_content)
        }

        # Generate Text export
        text_content = self._generate_text_report(rapport_data)
        text_export = {
            "format": "text/plain",
            "contenu": text_content,
            "taille": len(text_content)
        }

        return {
            "export_json": json_export,
            "export_text": text_export
        }

    def _generate_text_report(self, data: Dict) -> str:
        """Generate simple text report"""
        text = f"""
RAPPORT DE SINISTRE AXA
=======================

Référence: {data.get('id_rapport', 'N/A')}
Date: {datetime.now().strftime('%d/%m/%Y à %H:%M')}

INFORMATIONS PRINCIPALES
------------------------
Type de sinistre: {data.get('type_sinistre', 'N/A').replace('_', ' ').title()}
Sévérité: {data.get('severite', 'N/A').upper()}
Urgence: {data.get('details', {}).get('urgence', 'N/A')}/10

RÉSUMÉ
------
{data.get('resume', 'Aucun résumé disponible')}

DÉTAILS TECHNIQUES
------------------
Confiance classification: {data.get('details', {}).get('confiance_classification', 0):.1%}
Confiance OCR: {data.get('informations_ocr', {}).get('confiance_ocr', 0):.1%}
"""

        # Add amount if available
        montant = data.get('details', {}).get('montant_estime')
        if montant:
            text += f"Montant estimé: {montant:,.0f}€\n"

        # Add keywords
        mots_cles = data.get('details', {}).get('mots_cles', [])
        if mots_cles:
            text += f"Mots-clés identifiés: {', '.join(mots_cles)}\n"

        # Add actions
        actions = data.get('actions_recommandees', [])
        if actions:
            text += f"\nACTIONS RECOMMANDÉES\n"
            text += f"--------------------\n"
            for i, action in enumerate(actions, 1):
                text += f"{i}. {action}\n"

        # Add timeline
        text += f"\nDÉLAI DE TRAITEMENT: {data.get('delai_traitement', 'Non défini')}\n"

        # Footer
        text += f"\n" + "=" * 50 + f"\n"
        text += f"Rapport généré automatiquement en {data.get('temps_generation', 0):.2f} secondes\n"
        text += f"Système AXA de traitement des sinistres\n"

        return text.strip()

    def criticize(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        CRITIC: Basic validation
        """
        if "erreur" in output_data:
            return {
                "retour": f"Échec génération : {output_data['erreur']}",
                "confiance": 0.0,
                "necessite_relance": True
            }

        rapport_data = output_data.get("rapport_data", {})
        json_export = output_data.get("export_json", {})
        text_export = output_data.get("export_text", {})

        # Simple quality checks
        has_data = bool(rapport_data.get("resume")) and bool(rapport_data.get("type_sinistre"))
        has_exports = bool(json_export.get("contenu")) and bool(text_export.get("contenu"))

        quality_score = 1.0 if (has_data and has_exports) else 0.5

        feedback = "Rapport généré avec succès - JSON et Text disponibles" if quality_score > 0.5 else "Rapport incomplet"

        return {
            "retour": feedback,
            "confiance": quality_score,
            "necessite_relance": quality_score < 0.5
        }

    def terminate(self, output_data: Dict[str, Any], critic_feedback: str) -> Dict[str, Any]:
        """
        TERMINATION: Prepare final output
        """
        if "erreur" in output_data:
            return {
                "succes": False,
                "erreur": output_data["erreur"],
                "rapport_final": None
            }

        try:
            rapport_data = output_data.get("rapport_data", {})
            classification_input = self.state.input_data.get("resultat_classification", {})

            # Create minimal ClaimReport
            claim_report = ClaimReport(
                id_sinistre=rapport_data.get("id_rapport", f"AXA-{datetime.now().strftime('%Y%m%d')}"),
                type_sinistre=ClaimType(classification_input.get("type_sinistre", "inconnu")),
                severite=Severity(classification_input.get("severite", "moderee")),
                resume=rapport_data.get("resume", "Rapport automatique"),
                informations_extraites=classification_input,
                actions_recommandees=classification_input.get("actions_immediates", []),
                temps_traitement_estime=rapport_data.get("delai_traitement", "5-10 jours"),
                necessite_revision_humaine=classification_input.get("score_confiance", 1) < 0.7,
                score_confiance=classification_input.get("score_confiance", 0.0)
            )

            logger.info(f"Report Agent terminé - Type: {claim_report.type_sinistre.value}")

            return {
                "succes": True,
                "rapport_final": claim_report.dict(),
                "rapport_data": rapport_data,
                "exports": {
                    "json": output_data.get("export_json", {}),
                    "text": output_data.get("export_text", {})
                },
                "resume_generation": {
                    "formats_export": 2,  # JSON + Text
                    "temps_generation": rapport_data.get("temps_generation", 0),
                    "taille_total": (
                            output_data.get("export_json", {}).get("taille", 0) +
                            output_data.get("export_text", {}).get("taille", 0)
                    )
                }
            }

        except Exception as e:
            logger.error(f"Erreur finalisation rapport : {str(e)}")
            return {
                "succes": False,
                "erreur": f"Erreur finalisation: {str(e)}",
                "rapport_final": None
            }