"""
Minimal Report Agent for POC demonstration
Simple implementation - JSON and Text export only
"""
from typing import Dict, Any, Optional
import time
import json
from datetime import datetime
from enum import Enum
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
try:
    from utils.enum_utils import safe_enum_value, safe_json_dumps
except ImportError:
    from src.utils.enum_utils import safe_enum_value, safe_json_dumps



class EnumJSONEncoder(json.JSONEncoder):
    """Encodeur JSON qui gère les enums automatiquement"""
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value  # Convertit ClaimType.ACCIDENT_AUTO → "accident_auto"
        if isinstance(obj, datetime):
            return obj.isoformat()  # Bonus : gère aussi les dates
        return super().default(obj)


def normalize_enum_value(value):
    """Convertit un enum en string de façon sécurisée"""
    if hasattr(value, 'value'):
        return value.value
    return str(value)

def safe_json_dumps(data, **kwargs):
    """JSON dumps qui gère les enums automatiquement"""
    return json.dumps(data, cls=EnumJSONEncoder, ensure_ascii=False, **kwargs)


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

    # def _create_summary(self, classification: Dict) -> str:
    #     """Create basic summary text"""
    #     type_sinistre = classification.get("type_sinistre", "inconnu").replace("_", " ").title()
    #     severite = classification.get("severite", "moderee").upper()
    #     montant = classification.get("montant_estime")
    #
    #     summary = f"Sinistre de type {type_sinistre} avec sévérité {severite}."
    #
    #     if montant:
    #         summary += f" Montant estimé: {montant:,.0f}€."
    #
    #     urgence = classification.get("score_urgence", 5)
    #     if urgence >= 7:
    #         summary += " Traitement prioritaire requis."
    #
    #     return summary
    #
    # def _create_summary(self, classification: Dict) -> str:
    #     """Create basic summary text"""
    #     # Gérer ClaimType enum correctement
    #     type_sinistre_raw = classification.get("type_sinistre", "inconnu")
    #
    #     # Si c'est un enum ClaimType, extraire la valeur
    #     if hasattr(type_sinistre_raw, 'value'):
    #         type_sinistre = type_sinistre_raw.value
    #     else:
    #         type_sinistre = str(type_sinistre_raw)
    #
    #     try:
    #         type_sinistre_display = type_sinistre.replace("_", " ").title()
    #     except AttributeError:
    #         # Au cas où type_sinistre ne serait pas un string
    #         type_sinistre_display = str(type_sinistre).replace("_", " ").title()
    #
    #     severite_raw = classification.get("severite", "moderee")
    #     if hasattr(severite_raw, 'value'):
    #         severite = severite_raw.value
    #     else:
    #         severite = str(severite_raw)
    #
    #     try:
    #         severite_display = severite.upper()
    #     except AttributeError:
    #         severite_display = str(severite).upper()
    #
    #     montant = classification.get("montant_estime")
    #
    #     summary = f"Sinistre de type {type_sinistre_display} avec sévérité {severite_display}."
    #
    #     if montant:
    #         summary += f" Montant estimé: {montant:,.0f}€."
    #
    #     urgence = classification.get("score_urgence", 5)
    #     if urgence >= 7:
    #         summary += " Traitement prioritaire requis."
    #
    #     return summary

    def _create_summary(self, classification: Dict) -> str:
        """Version ultra-simple pour éviter tout problème d'enum"""

        # Conversion forcée en string
        type_raw = classification.get("type_sinistre", "inconnu")
        type_str = type_raw.value if hasattr(type_raw, 'value') else str(type_raw)

        sev_raw = classification.get("severite", "moderee")
        sev_str = sev_raw.value if hasattr(sev_raw, 'value') else str(sev_raw)

        # Formatage simple
        type_display = type_str.replace("_", " ").title()
        sev_display = sev_str.upper()

        return f"Sinistre de type {type_display} avec sévérité {sev_display}."

    def act(self, execution_output: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        ACTION: Generate JSON and Text exports
        """
        if "erreur" in execution_output:
            return None

        rapport_data = execution_output.get("rapport_data", {})

        # Generate JSON export
        # json_content = json.dumps(rapport_data, ensure_ascii=False, indent=2)
        try:
            json_content = safe_json_dumps(rapport_data, indent=2)
            json_export = {
                "format": "application/json",
                "contenu": json_content,
                "taille": len(json_content)
            }
        except Exception as e:
            logger.error(f"JSON export failed: {e}")
            json_export = {
                "format": "application/json",
                "contenu": '{"erreur": "Serialization failed"}',
                "taille": 0
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

    # def terminate(self, output_data: Dict[str, Any], critic_feedback: str) -> Dict[str, Any]:
    #     """
    #     TERMINATION: Prepare final output
    #     """
    #     if "erreur" in output_data:
    #         return {
    #             "succes": False,
    #             "erreur": output_data["erreur"],
    #             "rapport_final": None
    #         }
    #
    #     try:
    #         rapport_data = output_data.get("rapport_data", {})
    #         classification_input = self.state.input_data.get("resultat_classification", {})
    #
    #         # Create minimal ClaimReport
    #         claim_report = ClaimReport(
    #             id_sinistre=rapport_data.get("id_rapport", f"AXA-{datetime.now().strftime('%Y%m%d')}"),
    #             type_sinistre=ClaimType(classification_input.get("type_sinistre", "inconnu")),
    #             severite=Severity(classification_input.get("severite", "moderee")),
    #             resume=rapport_data.get("resume", "Rapport automatique"),
    #             informations_extraites=classification_input,
    #             actions_recommandees=classification_input.get("actions_immediates", []),
    #             temps_traitement_estime=rapport_data.get("delai_traitement", "5-10 jours"),
    #             necessite_revision_humaine=classification_input.get("score_confiance", 1) < 0.7,
    #             score_confiance=classification_input.get("score_confiance", 0.0)
    #         )
    #
    #         logger.info(f"Report Agent terminé - Type: {claim_report.type_sinistre.value}")
    #
    #         return {
    #             "succes": True,
    #             "rapport_final": claim_report.dict(),
    #             "rapport_data": rapport_data,
    #             "exports": {
    #                 "json": output_data.get("export_json", {}),
    #                 "text": output_data.get("export_text", {})
    #             },
    #             "resume_generation": {
    #                 "formats_export": 2,  # JSON + Text
    #                 "temps_generation": rapport_data.get("temps_generation", 0),
    #                 "taille_total": (
    #                         output_data.get("export_json", {}).get("taille", 0) +
    #                         output_data.get("export_text", {}).get("taille", 0)
    #                 )
    #             }
    #         }
    #
    #     except Exception as e:
    #         logger.error(f"Erreur finalisation rapport : {str(e)}")
    #         return {
    #             "succes": False,
    #             "erreur": f"Erreur finalisation: {str(e)}",
    #             "rapport_final": None
    #         }

    # def terminate(self, output_data: Dict[str, Any], critic_feedback: str) -> Dict[str, Any]:
    #     """
    #     TERMINATION: Version simple pour POC
    #     """
    #     if "erreur" in output_data:
    #         return {
    #             "succes": False,
    #             "erreur": output_data["erreur"],
    #             "rapport_final": None
    #         }
    #
    #     try:
    #         rapport_data = output_data.get("rapport_data", {})
    #         classification_input = self.state.input_data.get("resultat_classification", {})
    #
    #         # Version simple : créer directement un dictionnaire
    #         claim_report_dict = {
    #             "id_sinistre": rapport_data.get("id_rapport", f"AXA-{datetime.now().strftime('%Y%m%d')}"),
    #             "type_sinistre": normalize_enum_value(classification_input.get("type_sinistre", "inconnu")),
    #             "severite": normalize_enum_value(classification_input.get("severite", "moderee")),
    #             "resume": rapport_data.get("resume", "Rapport automatique"),
    #             "score_confiance": classification_input.get("score_confiance", 0.0),
    #             "genere_le": datetime.now().isoformat()
    #         }
    #
    #         logger.info(f"Report Agent terminé - Type: {claim_report_dict['type_sinistre']}")
    #
    #         return {
    #             "succes": True,
    #             "rapport_final": claim_report_dict,
    #             "exports": {
    #                 "json": output_data.get("export_json", {}),
    #                 "text": output_data.get("export_text", {})
    #             }
    #         }
    #
    #     except Exception as e:
    #         logger.error(f"Erreur finalisation rapport : {str(e)}")
    #         return {
    #             "succes": False,
    #             "erreur": f"Erreur finalisation: {str(e)}",
    #             "rapport_final": None
    #         }
    def terminate(self, output_data: Dict[str, Any], critic_feedback: str) -> Dict[str, Any]:
        """
        TERMINATION: Version ultra-simple pour éviter les problèmes d'enum
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

            # SOLUTION: Convertir TOUS les enums en strings immédiatement
            type_sinistre_raw = classification_input.get("type_sinistre", "inconnu")
            if hasattr(type_sinistre_raw, 'value'):
                type_sinistre = type_sinistre_raw.value
            else:
                type_sinistre = str(type_sinistre_raw)

            severite_raw = classification_input.get("severite", "moderee")
            if hasattr(severite_raw, 'value'):
                severite = severite_raw.value
            else:
                severite = str(severite_raw)

            # Rapport simple sans objets enum
            claim_report_dict = {
                "id_sinistre": rapport_data.get("id_rapport", f"AXA-{datetime.now().strftime('%Y%m%d')}"),
                "type_sinistre": type_sinistre,  # String garantie
                "severite": severite,  # String garantie
                "resume": rapport_data.get("resume", "Rapport automatique"),
                "score_confiance": classification_input.get("score_confiance", 0.0),
                "genere_le": datetime.now().isoformat()
            }

            logger.info(f"Report Agent terminé - Type: {type_sinistre}")

            return {
                "succes": True,
                "rapport_final": claim_report_dict,
                "exports": {
                    "json": output_data.get("export_json", {}),
                    "text": output_data.get("export_text", {})
                }
            }

        except Exception as e:
            logger.error(f"Erreur finalisation rapport : {str(e)}")
            return {
                "succes": False,
                "erreur": f"Erreur finalisation: {str(e)}",
                "rapport_final": None
            }