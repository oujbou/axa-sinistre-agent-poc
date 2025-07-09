"""
Classification Agent implementing REACT pattern for insurance claim classification
Specialized for AXA France business rules and French language processing
"""
from typing import Dict, Any, Optional
import time
import json
from loguru import logger

# Fix imports to work with project structure
try:
    from agents.base_agent import BaseAgent
    from models.claim_models import ClaimType, Severity, ClaimClassification, OCRResult
    from utils.classification_utils import (
        RuleBasedClassifier, SeverityAnalyzer, UrgencyCalculator, BusinessRulesEngine
    )
except ImportError:
    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.agents.base_agent import BaseAgent
    from src.models.claim_models import ClaimType, Severity, ClaimClassification, OCRResult
    from src.utils.classification_utils import (
        RuleBasedClassifier, SeverityAnalyzer, UrgencyCalculator, BusinessRulesEngine
    )


class ClassificationAgent(BaseAgent):
    """
    Classification Agent with REACT pattern for intelligent claim categorization

    Capabilities:
    - Hybrid classification (rules + LLM + business logic)
    - French insurance terminology
    - Severity and urgency assessment
    - Business rules validation
    - Fraud detection indicators
    """

    def __init__(self, llm_client, config: Dict[str, Any] = None):
        super().__init__("Classification_Agent", llm_client, config)

        # Initialize classification components
        self.rule_classifier = RuleBasedClassifier()
        self.severity_analyzer = SeverityAnalyzer()
        self.urgency_calculator = UrgencyCalculator()
        self.business_rules = BusinessRulesEngine()

        # Classification configuration
        self.confidence_threshold = config.get("confidence_threshold", 0.7) if config else 0.7
        self.use_llm_fallback = config.get("use_llm_fallback", True) if config else True
        self.enable_business_rules = config.get("enable_business_rules", True) if config else True

        logger.info("Classification Agent initialized with hybrid approach (rules + LLM + business)")

    def reason(self, input_data: Dict[str, Any]) -> str:
        """
        REASONING: Analyze OCR results and determine classification strategy

        Args:
            input_data: Contains OCR results and metadata

        Returns:
            str: Reasoning explanation and classification strategy
        """
        # Extract OCR data
        ocr_result = input_data.get("resultat_ocr", {})
        texte_extrait = ocr_result.get("texte_extrait", "")
        type_document = ocr_result.get("type_document", "inconnu")
        score_confiance_ocr = ocr_result.get("score_confiance", 0.0)

        reasoning_prompt = f"""
En tant qu'expert en classification de sinistres pour AXA Assurances France, analysez ces données OCR et déterminez la stratégie de classification optimale :

DONNÉES OCR:
- Texte extrait: "{texte_extrait[:500]}{'...' if len(texte_extrait) > 500 else ''}"
- Type de document détecté: {type_document}
- Confiance OCR: {score_confiance_ocr:.2f}
- Longueur du texte: {len(texte_extrait)} caractères

CONTEXTE MÉTIER:
Vous devez classifier ce sinistre selon les types AXA France:
- ACCIDENT_AUTO (collision, carambolage, rayure)
- DEGATS_HABITATION (maison, appartement, copropriété)  
- VOL (cambriolage, vol simple, vandalisme)
- INCENDIE (feu, explosion, fumée)
- DEGAT_DES_EAUX (fuite, inondation, plomberie)
- SANTE (frais médicaux, hospitalisation)
- VOYAGE (annulation, bagages perdus)
- RESPONSABILITE_CIVILE (dommages à tiers)
- Autres types spécialisés

STRATÉGIE À DÉTERMINER:
1. Méthode de classification prioritaire (règles métier vs analyse sémantique)
2. Indicateurs de sévérité attendus (FAIBLE, MODEREE, ELEVEE, CRITIQUE)
3. Facteurs d'urgence à considérer (1-10)
4. Signaux d'alerte (fraude potentielle, urgence médicale)
5. Règles métier AXA à appliquer (seuils, autorisations)

Fournissez une stratégie de classification détaillée et justifiée.
"""

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": "Vous êtes un expert en classification de sinistres AXA France. Fournissez une stratégie claire et méthodique."},
                    {"role": "user", "content": reasoning_prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )

            reasoning = response.choices[0].message.content
            logger.debug(f"Classification reasoning completed: {reasoning[:150]}...")
            return reasoning

        except Exception as e:
            logger.error(f"Erreur dans le raisonnement de classification : {str(e)}")
            return f"Erreur dans la phase de raisonnement : {str(e)}. Utilisation de la stratégie de classification hybride par défaut."

    def execute(self, input_data: Dict[str, Any], reasoning: str) -> Dict[str, Any]:
        """
        EXECUTION: Perform hybrid classification based on reasoning

        Args:
            input_data: OCR results and metadata
            reasoning: Strategy from reasoning phase

        Returns:
            Dict[str, Any]: Classification results
        """
        start_time = time.time()

        try:
            # Extract OCR data
            ocr_result = input_data.get("resultat_ocr", {})
            if not ocr_result:
                raise ValueError("Aucun résultat OCR fourni pour la classification")

            texte_extrait = ocr_result.get("texte_extrait", "")
            if not texte_extrait.strip():
                raise ValueError("Texte extrait vide, impossible de classifier")

            logger.info(f"Starting classification for text: {len(texte_extrait)} characters")

            # Step 1: Rule-based classification
            rule_result = self.rule_classifier.classify(texte_extrait)
            logger.debug(f"Rule-based result: {rule_result.best_match} (conf: {rule_result.confidence:.2f})")

            # Step 2: Determine if LLM analysis is needed
            needs_llm = (
                    rule_result.confidence < self.confidence_threshold or
                    rule_result.best_match is None or
                    self.use_llm_fallback
            )

            if needs_llm:
                llm_result = self._llm_semantic_classification(texte_extrait, rule_result, reasoning)
                logger.debug(
                    f"LLM classification: {llm_result.get('type_sinistre')} (conf: {llm_result.get('score_confiance', 0):.2f})")
            else:
                llm_result = self._create_fallback_llm_result(rule_result, texte_extrait)

            # Step 3: Merge and validate results
            final_classification = self._merge_classification_results(rule_result, llm_result, texte_extrait)

            # Add processing metadata
            final_classification["temps_traitement"] = time.time() - start_time
            final_classification["methode_utilisee"] = "hybride" if needs_llm else "regles"
            final_classification["reasoning_source"] = reasoning[:200] + "..." if len(reasoning) > 200 else reasoning

            return final_classification

        except Exception as e:
            logger.error(f"Erreur dans l'exécution de la classification : {str(e)}")
            return {
                "erreur": str(e),
                "type_sinistre": ClaimType.INCONNU.value,
                "severite": Severity.MODEREE.value,
                "score_confiance": 0.0,
                "temps_traitement": time.time() - start_time
            }

    def _llm_semantic_classification(self, texte: str, rule_result, reasoning: str) -> Dict[str, Any]:
        """Perform LLM-based semantic classification"""

#         prompt = f"""
# En tant qu'expert classification sinistres AXA France, analysez ce texte et fournissez une classification précise :
#
# TEXTE À CLASSIFIER:
# "{texte}"
#
# ANALYSE PRÉLIMINAIRE PAR RÈGLES:
# - Classification suggérée: {rule_result.best_match.value if rule_result.best_match else 'Aucune'}
# - Confiance règles: {rule_result.confidence:.2f}
# - Mots-clés détectés: {rule_result.keywords_found}
#
# STRATÉGIE DÉTERMINÉE:
# {reasoning[:300]}
#
# MISSION - Répondez uniquement en JSON valide:
# {{
#     "type_sinistre": "un des: accident_auto, degats_habitation, vol, incendie, degat_des_eaux, sante, voyage, responsabilite_civile, inconnu",
#     "severite": "un des: faible, moderee, elevee, critique",
#     "score_confiance": "0.0 à 1.0",
#     "justification": "explication courte de votre classification",
#     "montant_estime": "montant en euros ou null si impossible à estimer",
#     "flags": {{
#         "necessite_enquete": "true/false - investigation approfondie requise",
#         "potentiel_fraude": "true/false - éléments suspects détectés",
#         "urgence_medicale": "true/false - soins médicaux urgents",
#         "necessite_expertise": "true/false - expert technique requis"
#     }},
#     "score_urgence": "entier de 1 (routine) à 10 (critique)",
#     "actions_immediates": ["liste", "des", "actions", "à", "prendre"],
#     "mots_cles_decisifs": ["mots", "qui", "ont", "guidé", "la", "décision"]
# }}
# """

        prompt = f"""
        En tant qu'expert classification sinistres AXA France, analysez ce texte et fournissez une classification précise.

        TEXTE À CLASSIFIER:
        "{texte}"

        TYPES VALIDES uniquement:
        - accident_auto
        - degats_habitation  
        - vol
        - incendie
        - degat_des_eaux
        - sante
        - voyage
        - responsabilite_civile
        - catastrophe_naturelle
        - inconnu

        SÉVÉRITÉS VALIDES uniquement:
        - faible
        - moderee
        - elevee
        - critique

        EXEMPLE DE RÉPONSE JSON (respectez exactement ce format):
        {{
            "type_sinistre": "accident_auto",
            "severite": "moderee", 
            "score_confiance": 0.85,
            "justification": "Rayure de véhicule dans un parking",
            "montant_estime": 800,
            "flags": {{
                "necessite_enquete": false,
                "potentiel_fraude": false,
                "urgence_medicale": false,
                "necessite_expertise": false
            }},
            "score_urgence": 4,
            "actions_immediates": ["Contacter garage", "Devis réparation"],
            "mots_cles_decisifs": ["rayure", "parking", "véhicule"]
        }}

        IMPORTANT: Répondez UNIQUEMENT avec du JSON valide, rien d'autre.
        """

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": "Expert classification AXA France. Réponse JSON uniquement, précise et justifiée."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1500
            )

            try:
                result = json.loads(response.choices[0].message.content)
                return result
            except json.JSONDecodeError:
                # Fallback parsing if JSON is malformed
                logger.warning("Échec parsing JSON LLM, utilisation fallback")
                raw_text = response.choices[0].message.content
                return {
                    "type_sinistre": rule_result.best_match.value if rule_result.best_match else "inconnu",
                    "severite": "moderee",
                    "score_confiance": 0.5,
                    "justification": f"Parsing JSON échoué. Réponse brute: {raw_text[:200]}",
                    "montant_estime": None,
                    "flags": {
                        "necessite_enquete": False,
                        "potentiel_fraude": False,
                        "urgence_medicale": False,
                        "necessite_expertise": False
                    },
                    "score_urgence": 5,
                    "actions_immediates": ["Révision manuelle requise"],
                    "mots_cles_decisifs": []
                }

        except Exception as e:
            logger.error(f"Erreur classification LLM : {str(e)}")
            return self._create_fallback_llm_result(rule_result, texte)

    def _create_fallback_llm_result(self, rule_result, texte: str) -> Dict[str, Any]:
        """Create fallback result when LLM fails"""
        claim_type = rule_result.best_match if rule_result.best_match else ClaimType.INCONNU
        severity, _ = self.severity_analyzer.analyze_severity(texte, claim_type)

        return {
            "type_sinistre": claim_type.value,
            "severite": severity.value,
            "score_confiance": rule_result.confidence,
            "justification": "Classification basée sur règles métier uniquement",
            "montant_estime": None,
            "flags": {
                "necessite_enquete": rule_result.confidence < 0.5,
                "potentiel_fraude": False,
                "urgence_medicale": False,
                "necessite_expertise": False
            },
            "score_urgence": 5,
            "actions_immediates": ["Classification automatique"],
            "mots_cles_decisifs": rule_result.keywords_found.get(claim_type,
                                                                 []) if claim_type != ClaimType.INCONNU else []
        }

    def _merge_classification_results(self, rule_result, llm_result: Dict, texte: str) -> Dict[str, Any]:
        """Merge rule-based and LLM results intelligently"""

        # Start with LLM result as base
        merged = llm_result.copy()

        # Cross-validate type classification
        llm_type = ClaimType(llm_result["type_sinistre"])
        rule_type = rule_result.best_match

        # If rule confidence is very high and contradicts LLM, prefer rules
        if (rule_result.confidence > 0.8 and
                rule_type and
                rule_type != llm_type):
            logger.warning(
                f"Type conflict: Rules={rule_type.value}, LLM={llm_type.value}. Preferring rules (conf={rule_result.confidence:.2f})")
            merged["type_sinistre"] = rule_type.value
            merged["justification"] += f" [Correction: règles métier prioritaires sur LLM]"

        # Enhanced keyword analysis
        if rule_type and rule_type in rule_result.keywords_found:
            rule_keywords = rule_result.keywords_found[rule_type]
            llm_keywords = merged.get("mots_cles_decisifs", [])
            merged["mots_cles_decisifs"] = list(set(rule_keywords + llm_keywords))

        # Confidence adjustment based on agreement
        rule_conf = rule_result.confidence
        llm_conf = merged["score_confiance"]

    def _merge_classification_results(self, rule_result, llm_result: Dict, texte: str) -> Dict[str, Any]:
        """Merge rule-based and LLM results intelligently"""

        # Start with LLM result as base
        merged = llm_result.copy()

        # Cross-validate type classification
        llm_type = ClaimType(llm_result["type_sinistre"])
        rule_type = rule_result.best_match

        # If rule confidence is very high and contradicts LLM, prefer rules
        if (rule_result.confidence > 0.8 and
                rule_type and
                rule_type != llm_type):
            logger.warning(
                f"Type conflict: Rules={rule_type.value}, LLM={llm_type.value}. Preferring rules (conf={rule_result.confidence:.2f})")
            merged["type_sinistre"] = rule_type.value
            merged["justification"] += f" [Correction: règles métier prioritaires sur LLM]"

        # Enhanced keyword analysis
        if rule_type and rule_type in rule_result.keywords_found:
            rule_keywords = rule_result.keywords_found[rule_type]
            llm_keywords = merged.get("mots_cles_decisifs", [])
            merged["mots_cles_decisifs"] = list(set(rule_keywords + llm_keywords))

        # Confidence adjustment based on agreement
        rule_conf = rule_result.confidence
        llm_conf = merged["score_confiance"]

        if rule_type == llm_type:
            # Agreement boosts confidence
            merged["score_confiance"] = min((rule_conf + llm_conf) / 2 * 1.2, 1.0)
        else:
            # Disagreement requires caution
            merged["score_confiance"] = (rule_conf + llm_conf) / 2 * 0.8
            merged["flags"]["necessite_enquete"] = True
            merged["justification"] += f" [Attention: désaccord règles/LLM]"

        return merged

    def act(self, execution_output: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        ACTION: Apply business rules and enrich classification

        Args:
            execution_output: Results from execution phase

        Returns:
            Optional[Dict[str, Any]]: Enhanced results with business logic
        """
        if "erreur" in execution_output:
            return None

        try:
            # Apply business rules if enabled
            if self.enable_business_rules:
                ocr_metadata = self.state.input_data.get("resultat_ocr", {})
                enhanced_result = self.business_rules.apply_business_rules(
                    execution_output,
                    ocr_metadata
                )

                logger.info(f"Business rules applied: {len(enhanced_result.get('business_warnings', []))} adjustments")
            else:
                enhanced_result = execution_output.copy()

            # Calculate final urgency score
            claim_type = ClaimType(enhanced_result["type_sinistre"])
            severity = Severity(enhanced_result["severite"])
            flags = enhanced_result.get("flags", {})
            texte = self.state.input_data.get("resultat_ocr", {}).get("texte_extrait", "")

            final_urgency = self.urgency_calculator.calculate_urgency(
                claim_type, severity, flags, texte
            )
            enhanced_result["score_urgence"] = final_urgency

            # Add processing statistics
            enhanced_result["statistiques_classification"] = {
                "mots_cles_total": len(enhanced_result.get("mots_cles_decisifs", [])),
                "flags_actifs": sum(1 for v in flags.values() if v),
                "niveau_automatisation": "elevé" if enhanced_result["score_confiance"] > 0.8 else "moyen" if
                enhanced_result["score_confiance"] > 0.6 else "faible"
            }

            return enhanced_result

        except Exception as e:
            logger.error(f"Erreur dans la phase d'action de classification : {str(e)}")
            return {"erreur_action": str(e)}

    def criticize(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        CRITIC: Evaluate classification quality and provide feedback

        Args:
            output_data: All output data to evaluate

        Returns:
            Dict[str, Any]: Critic feedback with confidence and validation
        """
        if "erreur" in output_data:
            return {
                "retour": f"Classification échouée : {output_data['erreur']}",
                "confiance": 0.0,
                "necessite_relance": True,
                "problemes": ["classification_echouee"]
            }

        type_sinistre = output_data.get("type_sinistre", "inconnu")
        severite = output_data.get("severite", "moderee")
        score_confiance = output_data.get("score_confiance", 0.0)
        flags = output_data.get("flags", {})

        issues = []
        feedback_parts = []
        validation_score = score_confiance

        # Evaluate classification confidence
        if score_confiance < 0.4:
            issues.append("confiance_tres_faible")
            feedback_parts.append(f"Confiance très faible ({score_confiance:.2f}) - révision humaine recommandée")
        elif score_confiance < 0.6:
            issues.append("confiance_faible")
            feedback_parts.append(f"Confiance modérée ({score_confiance:.2f}) - validation superviseur conseillée")

        # Evaluate type classification
        if type_sinistre == "inconnu":
            issues.append("type_non_determine")
            feedback_parts.append("Type de sinistre non déterminé - classification manuelle requise")
            validation_score *= 0.5

        # Check for logical inconsistencies
        montant = output_data.get("montant_estime", 0) or 0
        if montant > 10000 and severite == "faible":
            issues.append("incoherence_montant_severite")
            feedback_parts.append("Incohérence: montant élevé avec sévérité faible")
            validation_score *= 0.8

        # Evaluate flags consistency
        if flags.get("urgence_medicale", False) and output_data.get("score_urgence", 5) < 7:
            issues.append("incoherence_urgence_medicale")
            feedback_parts.append("Incohérence: urgence médicale mais score d'urgence faible")

        # Check business rules application
        if output_data.get("business_warnings"):
            feedback_parts.append(f"{len(output_data['business_warnings'])} ajustements par règles métier")

        # Overall quality assessment
        if validation_score >= 0.85:
            feedback_parts.append("Classification de haute qualité - traitement automatique recommandé")
        elif validation_score >= 0.7:
            feedback_parts.append("Classification acceptable - validation rapide suffisante")
        elif validation_score >= 0.5:
            feedback_parts.append("Classification incertaine - révision détaillée requise")
        else:
            feedback_parts.append("Classification peu fiable - traitement manuel obligatoire")

        # Check for escalation needs
        escalation_needed = any([
            validation_score < 0.6,
            type_sinistre == "inconnu",
            flags.get("potentiel_fraude", False),
            montant > 15000
        ])

        final_feedback = ". ".join(feedback_parts) if feedback_parts else "Classification terminée avec succès"

        return {
            "retour": final_feedback,
            "confiance": validation_score,
            "necessite_relance": validation_score < 0.4,
            "necessite_escalade": escalation_needed,
            "problemes": issues,
            "score_qualite": validation_score,
            "recommandation_workflow": self._determine_workflow_recommendation(validation_score, flags, type_sinistre)
        }

    def _determine_workflow_recommendation(self, confidence: float, flags: Dict, type_sinistre: str) -> str:
        """Determine recommended workflow path"""

        if flags.get("urgence_medicale", False):
            return "workflow_urgence_medicale"
        elif flags.get("potentiel_fraude", False):
            return "workflow_enquete_fraude"
        elif confidence < 0.5:
            return "workflow_revision_manuelle"
        elif type_sinistre in ["incendie", "catastrophe_naturelle"]:
            return "workflow_expertise_specialisee"
        elif confidence > 0.8:
            return "workflow_automatique"
        else:
            return "workflow_validation_superviseur"

    def terminate(self, output_data: Dict[str, Any], critic_feedback: str) -> Dict[str, Any]:
        """
        TERMINATION: Prepare final classification results

        Args:
            output_data: All output data
            critic_feedback: Feedback from critic phase

        Returns:
            Dict[str, Any]: Final validated classification results
        """
        if "erreur" in output_data:
            return {
                "succes": False,
                "erreur": output_data["erreur"],
                "resultat_classification": None
            }

        try:
            # Create standardized ClaimClassification
            classification_result = ClaimClassification(
                type_sinistre=ClaimType(output_data.get("type_sinistre", "inconnu")),
                severite=Severity(output_data.get("severite", "moderee")),
                score_confiance=output_data.get("score_confiance", 0.0),
                mots_cles_trouves=output_data.get("mots_cles_decisifs", []),
                montant_estime=output_data.get("montant_estime"),
                necessite_enquete=output_data.get("flags", {}).get("necessite_enquete", False),
                score_urgence=output_data.get("score_urgence", 5),
                methode_classification=output_data.get("methode_utilisee", "hybride"),
                temps_traitement=output_data.get("temps_traitement", 0.0),
                raisonnement_llm=output_data.get("justification", "")
            )

            # Prepare enhanced data
            enhanced_data = {}

            # Business rules information
            if "business_warnings" in output_data:
                enhanced_data["avertissements_metier"] = output_data["business_warnings"]

            # Processing statistics
            if "statistiques_classification" in output_data:
                enhanced_data["statistiques"] = output_data["statistiques_classification"]

            # Workflow recommendations
            if "recommandation_workflow" in output_data:
                enhanced_data["workflow_recommande"] = output_data["recommandation_workflow"]

            # Actions and next steps
            enhanced_data["actions_immediates"] = output_data.get("actions_immediates", [])
            enhanced_data["niveau_autorisation"] = output_data.get("niveau_autorisation", "agent")
            enhanced_data["delai_traitement_estime"] = output_data.get("delai_traitement_estime", "1-2sem")

            # Flags summary
            flags = output_data.get("flags", {})
            enhanced_data["indicateurs_speciaux"] = {
                "enquete_requise": flags.get("necessite_enquete", False),
                "expertise_requise": flags.get("necessite_expertise", False),
                "urgence_medicale": flags.get("urgence_medicale", False),
                "suspicion_fraude": flags.get("potentiel_fraude", False)
            }

            logger.info(f"Classification Agent terminé - Type: {classification_result.type_sinistre.value}, "
                        f"Sévérité: {classification_result.severite.value}, "
                        f"Confiance: {classification_result.score_confiance:.2f}")

            return {
                "succes": True,
                "resultat_classification": classification_result.dict(),
                "donnees_ameliorees": enhanced_data,
                "resume_classification": {
                    "type": classification_result.type_sinistre.value,
                    "severite": classification_result.severite.value,
                    "confiance": classification_result.score_confiance,
                    "urgence": classification_result.score_urgence,
                    "temps_traitement": classification_result.temps_traitement,
                    "methode": classification_result.methode_classification
                }
            }

        except Exception as e:
            logger.error(f"Erreur dans la finalisation de classification : {str(e)}")
            return {
                "succes": False,
                "erreur": f"Erreur finalisation: {str(e)}",
                "resultat_classification": None
            }