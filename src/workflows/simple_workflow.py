"""
LangGraph workflow for conditional routing and multi-agent orchestration
"""
from typing import Dict, Any, Literal
import time
from datetime import datetime
from loguru import logger

try:
    from langgraph.graph import StateGraph, END
    from models.claim_models import ProcessingInput, WorkflowState
    from agents.ocr_agent import OCRAgent
    from agents.classification_agent import ClassificationAgent
    from agents.report_agent import ReportAgent
except ImportError:
    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from langgraph.graph import StateGraph, END
    from src.models.claim_models import ProcessingInput, WorkflowState
    from src.agents.ocr_agent import OCRAgent
    from src.agents.classification_agent import ClassificationAgent
    from src.agents.report_agent import ReportAgent


class SimpleAXAWorkflow:
    """Version simplifiée worflow AXA pour démonstration"""

    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.ocr_agent = OCRAgent(llm_client)
        self.classification_agent = ClassificationAgent(llm_client)
        self.report_agent = ReportAgent(llm_client)

        self.quality_threshold = 0.6
        self.high_confidence_threshold = 0.8

        self.workflow = self._build_workflow()
        logger.info("Simple workflow initialized")

    def _build_workflow(self) -> StateGraph:
        """Workflow simple"""
        workflow = StateGraph(dict)

        # Nœuds
        workflow.add_node("ocr", self._ocr_node)
        workflow.add_node("classification", self._classification_node)
        workflow.add_node("report", self._report_node)
        workflow.add_node("fast_track", self._fast_track_node)
        workflow.add_node("detailed_analysis", self._detailed_analysis_node)
        workflow.add_node("human_review", self._human_review_node)

        # Entry point
        workflow.set_entry_point("ocr")

        # Routage simplifié
        workflow.add_conditional_edges(
            "ocr",
            self._route_after_ocr,
            {
                "continue": "classification",
                "human_review": "human_review"
            }
        )

        workflow.add_conditional_edges(
            "classification",
            self._route_after_classification,
            {
                "report": "report",
                "fast_track": "fast_track",
                "detailed_analysis": "detailed_analysis",
                "human_review": "human_review"
            }
        )

        # Terminaisons
        workflow.add_edge("report", END)
        workflow.add_edge("fast_track", END)
        workflow.add_edge("detailed_analysis", END)
        workflow.add_edge("human_review", END)

        return workflow.compile()

    def _ocr_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """OCR simplifié"""
        logger.info("🔍 OCR Node")
        start_time = time.time()

        try:
            ocr_result = self.ocr_agent.execute_react_cycle(state["input_data"])

            if ocr_result.status.value == "completed" and ocr_result.output_data.get("succes"):
                state["ocr_result"] = ocr_result.output_data
                state["ocr_confidence"] = ocr_result.output_data.get("resultat_ocr", {}).get("score_confiance", 0.5)
                state["success"] = True
            else:
                state["success"] = False
                state["errors"] = [f"OCR failed: {ocr_result.output_data.get('erreur', 'Unknown')}"]

        except Exception as e:
            logger.error(f"OCR error: {e}")
            state["success"] = False
            state["errors"] = [f"OCR error: {str(e)}"]

        state["step_times"] = state.get("step_times", {})
        state["step_times"]["ocr"] = time.time() - start_time
        state["current_step"] = "ocr"

        return state

    def _route_after_ocr(self, state: Dict[str, Any]) -> Literal["continue", "human_review"]:
        """Routage simple après OCR"""
        if not state.get("success", False):
            logger.info("🔀 OCR failed -> human_review")
            return "human_review"

        ocr_confidence = state.get("ocr_confidence", 0)

        if ocr_confidence >= self.quality_threshold:
            logger.info(f"🔀 OCR confidence {ocr_confidence:.2f} -> continue")
            return "continue"
        else:
            logger.info(f"🔀 OCR confidence {ocr_confidence:.2f} too low -> human_review")
            return "human_review"

    def _classification_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Classification simplifiée"""
        logger.info("🏷️ Classification Node")
        start_time = time.time()

        try:
            classification_result = self.classification_agent.execute_react_cycle(state["ocr_result"])

            if classification_result.status.value == "completed" and classification_result.output_data.get("succes"):
                state["classification_result"] = classification_result.output_data

                # Extraire données simples pour routage
                class_data = classification_result.output_data.get("resultat_classification", {})
                state["classification_confidence"] = class_data.get("score_confiance", 0.5)
                state["type_sinistre"] = class_data.get("type_sinistre", "inconnu")
                state["montant_estime"] = class_data.get("montant_estime", 0) or 0
                state["urgence_score"] = class_data.get("score_urgence", 5)

                # Flags simples
                state["medical_emergency"] = state.get("urgence_score", 5) >= 8
                state["high_amount"] = state.get("montant_estime", 0) > 10000
                state["fraud_risk"] = False  # Simplifié pour le POC

                state["success"] = True
            else:
                state["success"] = False
                state["errors"] = state.get("errors", []) + ["Classification failed"]

        except Exception as e:
            logger.error(f"Classification error: {e}")
            state["success"] = False
            state["errors"] = state.get("errors", []) + [f"Classification error: {str(e)}"]

        state["step_times"]["classification"] = time.time() - start_time
        state["current_step"] = "classification"

        return state

    def _route_after_classification(self, state: Dict[str, Any]) -> Literal["report", "fast_track", "detailed_analysis", "human_review"]:
        """Routage simple après classification"""

        if not state.get("success", False):
            logger.info("🔀 Classification failed -> human_review")
            return "human_review"

        confidence = state.get("classification_confidence", 0)
        montant = state.get("montant_estime", 0)
        urgence = state.get("urgence_score", 5)
        medical = state.get("medical_emergency", False)

        # Logique simple
        if medical or urgence >= 8:
            logger.info(f"🔀 Medical/urgent case -> fast_track")
            return "fast_track"
        elif montant > 15000:
            logger.info(f"🔀 High amount {montant}€ -> detailed_analysis")
            return "detailed_analysis"
        elif confidence < 0.5:
            logger.info(f"🔀 Low confidence {confidence:.2f} -> human_review")
            return "human_review"
        else:
            logger.info(f"🔀 Standard case -> report")
            return "report"

    # def _report_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
    #     """Rapport standard"""
    #     logger.info("📋 Report Node")
    #     start_time = time.time()
    #
    #     try:
    #         # Données combinées
    #         report_input = {
    #             **state["ocr_result"],
    #             **state["classification_result"]
    #         }
    #
    #         report_result = self.report_agent.execute_react_cycle(report_input)
    #
    #         if report_result.status.value == "completed" and report_result.output_data.get("succes"):
    #             state["report_result"] = report_result.output_data
    #             state["final_step"] = "completed"
    #         else:
    #             state["errors"] = state.get("errors", []) + ["Report failed"]
    #             state["final_step"] = "error"
    #
    #     except Exception as e:
    #         logger.error(f"Report error: {e}")
    #         state["errors"] = state.get("errors", []) + [f"Report error: {str(e)}"]
    #         state["final_step"] = "error"
    #
    #     state["step_times"]["report"] = time.time() - start_time
    #     state["current_step"] = "completed"
    #
    #     return state

    def _report_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """BYPASS DU REPORT AGENT - Version simple sans erreur"""
        logger.info("📋 Report Node - BYPASS VERSION")
        start_time = time.time()

        try:
            # Au lieu d'utiliser le report_agent, créons un rapport simple directement
            class_data = state.get("classification_result", {}).get("resultat_classification", {})

            # Extraction sécurisée des données
            type_sinistre = class_data.get("type_sinistre", "inconnu")
            if hasattr(type_sinistre, 'value'):
                type_sinistre = type_sinistre.value

            severite = class_data.get("severite", "moderee")
            if hasattr(severite, 'value'):
                severite = severite.value

            montant = class_data.get("montant_estime", 0) or 0
            confiance = class_data.get("score_confiance", 0.5)

            # Rapport simple créé directement
            simple_report = {
                "id_rapport": f"AXA-{state.get('session_id', 'default')}",
                "date_generation": datetime.now().isoformat(),
                "type_sinistre": type_sinistre,
                "severite": severite,
                "montant_estime": montant,
                "score_confiance": confiance,
                "resume": f"Sinistre {type_sinistre.replace('_', ' ')} - Sévérité {severite}",
                "statut": "traite_automatiquement"
            }

            # JSON export simple
            import json
            json_export = {
                "format": "application/json",
                "contenu": json.dumps(simple_report, ensure_ascii=False, indent=2),
                "taille": len(json.dumps(simple_report))
            }

            # Text export simple
            text_content = f"""
    RAPPORT DE SINISTRE AXA
    =======================
    ID: {simple_report['id_rapport']}
    Type: {type_sinistre.replace('_', ' ').title()}
    Sévérité: {severite.upper()}
    Montant: {montant}€
    Confiance: {confiance:.1%}
    Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}
            """.strip()

            text_export = {
                "format": "text/plain",
                "contenu": text_content,
                "taille": len(text_content)
            }

            # Résultat final
            state["report_result"] = {
                "succes": True,
                "rapport_data": simple_report,
                "export_json": json_export,
                "export_text": text_export
            }

            state["final_step"] = "completed"
            logger.info("✅ Rapport simple généré avec succès")

        except Exception as e:
            logger.error(f"Erreur rapport simple: {e}")
            state["report_result"] = {"succes": False, "erreur": str(e)}
            state["final_step"] = "error"

        state["step_times"]["report"] = time.time() - start_time
        state["current_step"] = "completed"

        return state

    def _fast_track_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Fast track simple"""
        logger.info("🚀 Fast Track Node")

        state["fast_report"] = {
            "id_rapport": f"FAST-{state.get('session_id', 'default')}",
            "type_sinistre": state.get("type_sinistre", "inconnu"),
            "traitement": "automatique_rapide",
            "confiance": state.get("classification_confidence", 0),
            "actions": ["Traitement automatique", "Notification client"]
        }

        state["current_step"] = "fast_track"
        state["final_step"] = "fast_track"

        return state

    def _detailed_analysis_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse détaillée simple"""
        logger.info("🔬 Detailed Analysis Node")

        state["detailed_report"] = {
            "id_rapport": f"DETAIL-{state.get('session_id', 'default')}",
            "analyse_approfondie": True,
            "montant_impact": state.get("montant_estime", 0),
            "actions_specialisees": ["Expert spécialisé", "Investigation", "Validation direction"],
            "delai_traitement": "5-10 jours"
        }

        state["current_step"] = "detailed_analysis"
        state["final_step"] = "detailed_analysis"

        return state

    def _human_review_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Révision humaine simple"""
        logger.info("👤 Human Review Node")

        state["human_review"] = {
            "id_rapport": f"HUMAN-{state.get('session_id', 'default')}",
            "intervention_humaine": True,
            "raison_escalade": "Qualité insuffisante ou cas complexe",
            "statut": "en_attente_revision"
        }

        state["current_step"] = "human_review"
        state["final_step"] = "human_review"

        return state

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Exécution simplifiée"""
        logger.info("🚀 Starting Simple AXA Workflow")

        # État initial simple
        initial_state = {
            "session_id": f"AXA-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "input_data": input_data,
            "current_step": "start",
            "success": True,
            "errors": [],
            "step_times": {}
        }

        try:
            final_state = self.workflow.invoke(initial_state)

            success = final_state.get("success", False) and len(final_state.get("errors", [])) == 0
            total_time = sum(final_state.get("step_times", {}).values())

            return {
                "session_id": final_state.get("session_id"),
                "success": success,
                "workflow_path": list(final_state.get("step_times", {}).keys()),
                "routing_decisions": [],  # Simplifié
                "results": {
                    "ocr": final_state.get("ocr_result", {}),
                    "classification": final_state.get("classification_result", {}),
                    "report": final_state.get("report_result", {}),
                    "fast_report": final_state.get("fast_report", {}),
                    "detailed_report": final_state.get("detailed_report", {}),
                    "human_review": final_state.get("human_review", {})
                },
                "errors": final_state.get("errors", []),
                "execution_time": total_time,
                "final_step": final_state.get("final_step", final_state.get("current_step", "unknown"))
            }

        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            return {
                "session_id": initial_state["session_id"],
                "success": False,
                "workflow_path": ["error"],
                "routing_decisions": [],
                "results": {},
                "errors": [f"Workflow error: {str(e)}"],
                "execution_time": 0,
                "final_step": "error"
            }


def create_simple_workflow(llm_client) -> SimpleAXAWorkflow:
    """Factory simple"""
    return SimpleAXAWorkflow(llm_client)