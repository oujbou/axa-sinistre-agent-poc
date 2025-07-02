"""
LangGraph workflow demonstrating conditional routing and multi-agent orchestration
"""
from typing import Dict, Any, List, Literal
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


class SmartWorkflowState:
    """Smart state with routing decisions"""

    def __init__(self, input_data: Dict[str, Any]):
        self.session_id = f"AXA-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.input_data = input_data
        self.current_step = "start"
        self.results = {}
        self.errors = []
        self.start_time = datetime.now()
        self.step_times = {}
        self.routing_decisions = []  # Track routing choices
        self.confidence_scores = {}  # Track confidence at each step

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "input_data": self.input_data,
            "current_step": self.current_step,
            "results": self.results,
            "errors": self.errors,
            "start_time": self.start_time.isoformat(),
            "step_times": self.step_times,
            "routing_decisions": self.routing_decisions,
            "confidence_scores": self.confidence_scores
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SmartWorkflowState':
        instance = cls.__new__(cls)
        instance.session_id = data["session_id"]
        instance.input_data = data["input_data"]
        instance.current_step = data["current_step"]
        instance.results = data["results"]
        instance.errors = data["errors"]
        instance.start_time = datetime.fromisoformat(data["start_time"])
        instance.step_times = data["step_times"]
        instance.routing_decisions = data.get("routing_decisions", [])
        instance.confidence_scores = data.get("confidence_scores", {})
        return instance


class SmartAXAWorkflow:
    """
    Smart workflow with conditional routing
    """

    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.ocr_agent = OCRAgent(llm_client)
        self.classification_agent = ClassificationAgent(llm_client)
        self.report_agent = ReportAgent(llm_client)

        # Thresholds for routing decisions
        self.quality_threshold = 0.7
        self.high_confidence_threshold = 0.85
        self.fraud_risk_threshold = 0.6

        # Build workflow
        self.workflow = self._build_smart_workflow()

        logger.info("Workflow initialized with conditional routing")

    def _build_smart_workflow(self) -> StateGraph:
        """Build workflow with conditional routing"""

        workflow = StateGraph(dict)

        # Add agent nodes
        workflow.add_node("ocr", self._ocr_node)
        workflow.add_node("classification", self._classification_node)
        workflow.add_node("report", self._report_node)

        # Add control nodes
        workflow.add_node("quality_check", self._quality_check_node)
        workflow.add_node("human_review", self._human_review_node)
        workflow.add_node("fast_track", self._fast_track_node)
        workflow.add_node("detailed_analysis", self._detailed_analysis_node)

        # Entry point
        workflow.set_entry_point("ocr")

        # CONDITIONAL ROUTING
        workflow.add_conditional_edges(
            "ocr",
            self._route_after_ocr,
            {
                "good_quality": "classification",
                "poor_quality": "human_review",
                "needs_retry": "ocr"  # Could loop back
            }
        )

        workflow.add_conditional_edges(
            "classification",
            self._route_after_classification,
            {
                "standard_case": "report",
                "high_confidence": "fast_track",
                "complex_case": "detailed_analysis",
                "fraud_risk": "human_review",
                "needs_review": "quality_check"
            }
        )

        workflow.add_conditional_edges(
            "quality_check",
            self._route_after_quality,
            {
                "approved": "report",
                "needs_human": "human_review",
                "retry_classification": "classification"
            }
        )

        # Terminal edges
        workflow.add_edge("fast_track", END)
        workflow.add_edge("detailed_analysis", END)
        workflow.add_edge("report", END)
        workflow.add_edge("human_review", END)

        return workflow.compile()

    def _ocr_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """OCR processing with quality assessment"""
        logger.info("🔍 OCR Node: Processing document")
        step_start = time.time()

        workflow_state = SmartWorkflowState.from_dict(state)
        workflow_state.current_step = "ocr"

        try:
            # Execute OCR
            ocr_result = self.ocr_agent.execute_react_cycle(workflow_state.input_data)

            if ocr_result.status.value == "completed" and ocr_result.output_data.get("succes"):
                workflow_state.results["ocr"] = ocr_result.output_data

                # Extract confidence for routing
                ocr_confidence = ocr_result.output_data.get("resultat_ocr", {}).get("score_confiance", 0)
                workflow_state.confidence_scores["ocr"] = ocr_confidence

                logger.info(f"✅ OCR completed - confidence: {ocr_confidence:.2f}")
            else:
                workflow_state.errors.append(f"OCR failed: {ocr_result.output_data.get('erreur', 'Unknown')}")
                workflow_state.confidence_scores["ocr"] = 0.0

            workflow_state.step_times["ocr"] = time.time() - step_start
            return workflow_state.to_dict()

        except Exception as e:
            logger.error(f"OCR node error: {e}")
            workflow_state.errors.append(f"OCR error: {str(e)}")
            return workflow_state.to_dict()

    def _route_after_ocr(self, state: Dict[str, Any]) -> Literal["good_quality", "poor_quality", "needs_retry"]:
        """ROUTING LOGIC"""

        workflow_state = SmartWorkflowState.from_dict(state)
        ocr_confidence = workflow_state.confidence_scores.get("ocr", 0)

        # Decision logic based on OCR quality
        if ocr_confidence >= self.high_confidence_threshold:
            decision = "good_quality"
            reason = f"High OCR confidence ({ocr_confidence:.2f})"
        elif ocr_confidence >= self.quality_threshold:
            decision = "good_quality"
            reason = f"Acceptable OCR confidence ({ocr_confidence:.2f})"
        else:
            decision = "poor_quality"
            reason = f"Low OCR confidence ({ocr_confidence:.2f}) - human review needed"

        # Log routing decision
        workflow_state.routing_decisions.append({
            "step": "after_ocr",
            "decision": decision,
            "reason": reason,
            "confidence": ocr_confidence
        })

        logger.info(f"🔀 OCR Routing: {decision} - {reason}")
        return decision

    def _classification_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Classification with business intelligence"""
        logger.info("🏷️ Classification Node: Analyzing claim type")
        step_start = time.time()

        workflow_state = SmartWorkflowState.from_dict(state)
        workflow_state.current_step = "classification"

        try:
            # Execute classification
            classification_input = workflow_state.results["ocr"]
            classification_result = self.classification_agent.execute_react_cycle(classification_input)

            if classification_result.status.value == "completed" and classification_result.output_data.get("succes"):
                workflow_state.results["classification"] = classification_result.output_data

                # Extract business intelligence for routing
                class_data = classification_result.output_data.get("resultat_classification", {})
                class_confidence = class_data.get("score_confiance", 0)
                montant = class_data.get("montant_estime", 0) or 0
                urgence = class_data.get("score_urgence", 5)
                flags = class_data.get("flags", {})

                workflow_state.confidence_scores["classification"] = class_confidence

                # Store business context for routing
                workflow_state.results["business_context"] = {
                    "montant": montant,
                    "urgence": urgence,
                    "fraude_suspectee": flags.get("potentiel_fraude", False),
                    "enquete_requise": flags.get("necessite_enquete", False),
                    "urgence_medicale": flags.get("urgence_medicale", False)
                }

                logger.info(f"✅ Classification: {class_data.get('type_sinistre')} - confidence: {class_confidence:.2f}")
            else:
                workflow_state.errors.append("Classification failed")
                workflow_state.confidence_scores["classification"] = 0.0

            workflow_state.step_times["classification"] = time.time() - step_start
            return workflow_state.to_dict()

        except Exception as e:
            logger.error(f"Classification error: {e}")
            workflow_state.errors.append(f"Classification error: {str(e)}")
            return workflow_state.to_dict()

    def _route_after_classification(self, state: Dict[str, Any]) -> Literal[
        "standard_case", "high_confidence", "complex_case", "fraud_risk", "needs_review"]:
        """SMART ROUTING - Business logic driven"""

        workflow_state = SmartWorkflowState.from_dict(state)
        class_confidence = workflow_state.confidence_scores.get("classification", 0)

        # Récupérer les données métier depuis le résultat de classification
        classification_result = workflow_state.results.get("classification", {})
        resultat_classification = classification_result.get("resultat_classification", {})

        # business_context = workflow_state.results.get("business_context", {})

        montant = resultat_classification.get("montant", 0)
        urgence = resultat_classification.get("urgence", 5)
        type_sinistre = resultat_classification.get("type_sinistre", "inconnu")

        # Récupérer les flags depuis les données améliorées
        donnees_ameliorees = classification_result.get("donnees_ameliorees", {})
        indicateurs = donnees_ameliorees.get("indicateurs_speciaux", {})

        fraude_suspectee = indicateurs.get("fraude_suspectee", False)
        urgence_medicale = indicateurs.get("urgence_medicale", False)

        # Mise à jour du business context pour le routage
        business_context = {
            "montant": montant,
            "urgence": urgence,
            "fraude_suspectee": fraude_suspectee,
            "urgence_medicale": urgence_medicale,
            "type_sinistre": type_sinistre
        }
        workflow_state.results["business_context"] = business_context

        # Routing logic
        if fraude_suspectee:
            decision = "fraud_risk"
            reason = "Fraud indicators detected"
        elif urgence_medicale or urgence >= 9:
            decision = "high_confidence"  # Fast track for medical emergencies
            reason = f"Medical emergency (urgency: {urgence})"
        elif montant > 15000:
            decision = "complex_case"
            reason = f"High amount case (€{montant:,.0f})"
        elif class_confidence >= 0.9 and montant < 2000:
            decision = "high_confidence"
            reason = f"High confidence simple case"
        elif class_confidence < 0.6:
            decision = "needs_review"
            reason = f"Low classification confidence ({class_confidence:.2f})"
        elif type_sinistre in ["incendie", "catastrophe_naturelle"]:
            decision = "complex_case"
            reason = f"Complex claim type: {type_sinistre}"
        else:
            decision = "standard_case"
            reason = "Standard processing path"

        # Log routing decision
        workflow_state.routing_decisions.append({
            "step": "after_classification",
            "decision": decision,
            "reason": reason,
            "factors": {
                "confidence": class_confidence,
                "montant": montant,
                "urgence": urgence,
                "fraude": fraude_suspectee,
                "medical": urgence_medicale,
                "type": type_sinistre
            }
        })

        logger.info(f"🔀 Classification Routing: {decision} - {reason}")
        return decision

    def _quality_check_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Quality validation node"""
        logger.info("🔍 Quality Check: Validating processing quality")

        workflow_state = SmartWorkflowState.from_dict(state)
        workflow_state.current_step = "quality_check"

        # Calculate overall quality score
        ocr_conf = workflow_state.confidence_scores.get("ocr", 0)
        class_conf = workflow_state.confidence_scores.get("classification", 0)
        overall_quality = (ocr_conf + class_conf) / 2

        workflow_state.results["quality_assessment"] = {
            "overall_score": overall_quality,
            "ocr_confidence": ocr_conf,
            "classification_confidence": class_conf,
            "recommendation": "approve" if overall_quality >= 0.7 else "review"
        }

        logger.info(f"✅ Quality Check: Overall score {overall_quality:.2f}")
        return workflow_state.to_dict()

    def _route_after_quality(self, state: Dict[str, Any]) -> Literal["approved", "needs_human", "retry_classification"]:
        """Routing after quality check"""

        workflow_state = SmartWorkflowState.from_dict(state)
        quality_data = workflow_state.results.get("quality_assessment", {})
        overall_score = quality_data.get("overall_score", 0)

        if overall_score >= 0.8:
            decision = "approved"
            reason = f"High quality score ({overall_score:.2f})"
        elif overall_score >= 0.6:
            decision = "approved"  # Borderline but acceptable
            reason = f"Acceptable quality ({overall_score:.2f})"
        else:
            decision = "needs_human"
            reason = f"Quality too low ({overall_score:.2f})"

        workflow_state.routing_decisions.append({
            "step": "after_quality",
            "decision": decision,
            "reason": reason,
            "quality_score": overall_score
        })

        logger.info(f"🔀 Quality Routing: {decision} - {reason}")
        return decision

    def _report_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Report generation"""
        logger.info("📋 Report Node: Generating standard report")
        step_start = time.time()

        workflow_state = SmartWorkflowState.from_dict(state)
        workflow_state.current_step = "report"

        try:
            # Combine data for report
            report_input = {
                **workflow_state.results["ocr"],
                **workflow_state.results["classification"]
            }

            report_result = self.report_agent.execute_react_cycle(report_input)

            if report_result.status.value == "completed" and report_result.output_data.get("succes"):
                workflow_state.results["report"] = report_result.output_data
                logger.info("✅ Standard report generated")
            else:
                workflow_state.errors.append("Report generation failed")

            workflow_state.step_times["report"] = time.time() - step_start
            workflow_state.current_step = "completed"

            return workflow_state.to_dict()

        except Exception as e:
            logger.error(f"Report error: {e}")
            workflow_state.errors.append(f"Report error: {str(e)}")
            return workflow_state.to_dict()

    def _fast_track_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Fast track for high-confidence cases"""
        logger.info("🚀 Fast Track: High-confidence rapid processing")

        workflow_state = SmartWorkflowState.from_dict(state)
        workflow_state.current_step = "fast_track"

        # Generate simplified report for fast cases
        class_data = workflow_state.results["classification"]["resultat_classification"]

        fast_report = {
            "id_rapport": f"FAST-{workflow_state.session_id}",
            "type_sinistre": class_data.get("type_sinistre"),
            "severite": class_data.get("severite"),
            "traitement": "automatique_rapide",
            "confiance": workflow_state.confidence_scores.get("classification", 0),
            "actions": ["Traitement automatique approuvé", "Notification client immédiate"]
        }

        workflow_state.results["fast_report"] = fast_report
        workflow_state.current_step = "completed"

        logger.info("🚀 Fast track completed")
        return workflow_state.to_dict()

    def _detailed_analysis_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Detailed analysis for complex cases"""
        logger.info("🔬 Detailed Analysis: Complex case processing")

        workflow_state = SmartWorkflowState.from_dict(state)
        workflow_state.current_step = "detailed_analysis"

        # Enhanced analysis for complex cases
        business_context = workflow_state.results.get("business_context", {})

        detailed_report = {
            "id_rapport": f"DETAIL-{workflow_state.session_id}",
            "analyse_approfondie": True,
            "montant_impact": business_context.get("montant", 0),
            "facteurs_complexite": ["Montant élevé", "Enquête requise"],
            "actions_specialisees": [
                "Désignation expert spécialisé",
                "Investigation approfondie",
                "Validation direction"
            ],
            "delai_traitement": "5-10 jours ouvrés"
        }

        workflow_state.results["detailed_report"] = detailed_report
        workflow_state.current_step = "completed"

        logger.info("🔬 Detailed analysis completed")
        return workflow_state.to_dict()

    def _human_review_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Human review for edge cases"""
        logger.info("👤 Human Review: Manual intervention required")

        workflow_state = SmartWorkflowState.from_dict(state)
        workflow_state.current_step = "human_review"

        # Simulate human review decision
        review_report = {
            "id_rapport": f"HUMAN-{workflow_state.session_id}",
            "intervention_humaine": True,
            "raison_escalade": workflow_state.routing_decisions[-1][
                "reason"] if workflow_state.routing_decisions else "Quality concerns",
            "actions_requises": [
                "Révision manuelle du dossier",
                "Validation par superviseur",
                "Contact client si nécessaire"
            ],
            "statut": "en_attente_revision"
        }

        workflow_state.results["human_review"] = review_report
        workflow_state.current_step = "completed"

        logger.info("👤 Human review initiated")
        return workflow_state.to_dict()

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute smart workflow with routing"""
        logger.info("🚀 Starting Smart AXA Workflow with conditional routing")

        initial_state = SmartWorkflowState(input_data)

        try:
            final_state = self.workflow.invoke(initial_state.to_dict())

            total_time = sum(final_state.get("step_times", {}).values())

            return {
                "session_id": final_state.get("session_id"),
                "success": len(final_state.get("errors", [])) == 0,
                "workflow_path": [d["step"] for d in final_state.get("routing_decisions", [])],
                "routing_decisions": final_state.get("routing_decisions", []),
                "results": final_state.get("results", {}),
                "errors": final_state.get("errors", []),
                "execution_time": total_time,
                "final_step": final_state.get("current_step"),
                "intelligence_summary": self._create_intelligence_summary(final_state)
            }

        except Exception as e:
            logger.error(f"Smart workflow failed: {e}")
            return {
                "session_id": initial_state.session_id,
                "success": False,
                "workflow_path": ["error"],
                "routing_decisions": [],
                "results": {},
                "errors": [f"Workflow error: {str(e)}"],
                "execution_time": 0,
                "final_step": "error"
            }

    def _create_intelligence_summary(self, final_state: Dict[str, Any]) -> Dict[str, Any]:
        """Create intelligent summary"""

        routing_decisions = final_state.get("routing_decisions", [])
        results = final_state.get("results", {})

        return {
            "workflow_intelligence": {
                "routing_decisions_made": len(routing_decisions),
                "adaptive_path": [d["decision"] for d in routing_decisions],
                "business_factors_considered": len([d for d in routing_decisions if "factors" in d]),
                "automatic_optimization": "enabled"
            },
            "final_outcome": {
                "processing_type": final_state.get("current_step", "unknown"),
                "confidence_levels": final_state.get("confidence_scores", {}),
                "business_context_used": bool(results.get("business_context")),
                "quality_validated": bool(results.get("quality_assessment"))
            },
            "langraph_advantages_demonstrated": [
                "Conditional routing based on confidence",
                "Business rule-driven path selection",
                "Quality gates with fallbacks",
                "Dynamic workflow adaptation",
                "State persistence across nodes"
            ]
        }


def create_smart_workflow(llm_client) -> SmartAXAWorkflow:
    """Factory for smart workflow"""
    return SmartAXAWorkflow(llm_client)