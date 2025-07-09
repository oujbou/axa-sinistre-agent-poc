"""
OCR Agent implementing REACT pattern for document text extraction
Specialized for AXA insurance documents (French/English)
"""
from typing import Dict, Any, Optional
import time
from pathlib import Path
from loguru import logger

try:
    from agents.base_agent import BaseAgent
    from models.claim_models import DocumentType, OCRResult, ProcessingInput
    from utils.ocr_utils import OCRProcessor
except ImportError:
    # Alternative import strategy if running from different location
    import sys

    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.agents.base_agent import BaseAgent
    from src.models.claim_models import DocumentType, OCRResult, ProcessingInput
    from src.utils.ocr_utils import OCRProcessor

class OCRAgent(BaseAgent):
    """
    OCR Agent with REACT pattern for intelligent text extraction

    Capabilities:
    - PDF text extraction (native + OCR)
    - Image OCR with preprocessing
    - Document type detection
    - Quality validation
    - French/English language support
    """

    def __init__(self, llm_client, config: Dict[str, Any] = None):
        super().__init__("OCR_Agent", llm_client, config)
        self.ocr_processor = OCRProcessor()

        # OCR-specific configuration
        self.min_confidence_threshold = config.get("min_confidence", 0.3) if config else 0.3
        self.max_file_size_mb = config.get("max_file_size_mb", 50) if config else 50

        logger.info("OCR Agent initialized with advanced preprocessing")

    def reason(self, input_data: Dict[str, Any]) -> str:
        """
        REASONING: Analyze input and determine OCR strategy

        Args:
            input_data: Contains saisie_texte, chemin_fichier, contenu_fichier, type_fichier

        Returns:
            str: Reasoning explanation and OCR strategy
        """
        processing_input = ProcessingInput(**input_data)

        reasoning_prompt = f"""
En tant que spécialiste OCR pour les sinistres AXA Assurances, analysez cette entrée et déterminez la stratégie d'extraction optimale :

Analyse de l'entrée :
- Texte fourni : {'Oui' if processing_input.saisie_texte else 'Non'}
- Fichier fourni : {'Oui' if processing_input.chemin_fichier or processing_input.contenu_fichier else 'Non'}
- Type de fichier : {processing_input.type_fichier or 'N/A'}
- Contexte utilisateur : {processing_input.contexte_utilisateur}

Déterminez :
1. Méthode de traitement nécessaire (texte direct, extraction PDF, OCR image)
2. Type de document attendu (constat amiable, formulaire assurance, etc.)
3. Priorité de détection linguistique (français/anglais)
4. Attentes de qualité et seuils de confiance
5. Défis potentiels et stratégies d'atténuation

Fournissez une stratégie claire pour une extraction de texte optimale.
"""

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": "Vous êtes un spécialiste OCR pour les documents d'assurance. Fournissez un raisonnement clair et technique pour les stratégies d'extraction de texte."},
                    {"role": "user", "content": reasoning_prompt}
                ],
                max_tokens=800,
                temperature=0.3
            )

            reasoning = response.choices[0].message.content
            logger.debug(f"OCR reasoning completed: {reasoning[:100]}...")
            return reasoning

        except Exception as e:
            logger.error(f"Error in OCR reasoning: {str(e)}")
            return f"Erreur dans la phase de raisonnement : {str(e)}. Utilisation de l'approche OCR standard par défaut."

    def execute(self, input_data: Dict[str, Any], reasoning: str) -> Dict[str, Any]:
        """
        EXECUTION: Perform text extraction based on reasoning

        Args:
            input_data: Original input data
            reasoning: Strategy from reasoning phase

        Returns:
            Dict[str, Any]: Extraction results
        """
        start_time = time.time()
        processing_input = ProcessingInput(**input_data)

        try:
            # Case 1: Direct text input
            if processing_input.saisie_texte:
                logger.info("Processing direct text input")
                return self._process_text_input(processing_input.saisie_texte, start_time)

            # Case 2: File processing
            elif processing_input.chemin_fichier or processing_input.contenu_fichier:
                if processing_input.chemin_fichier:
                    return self._process_file_path(processing_input.chemin_fichier, start_time)
                else:
                    return self._process_file_content(
                        processing_input.contenu_fichier,
                        processing_input.type_fichier,
                        start_time
                    )

            else:
                raise ValueError("Aucune entrée valide fournie (ni texte ni fichier)")

        except Exception as e:
            logger.error(f"Error in OCR execution: {str(e)}")
            return {
                "erreur": str(e),
                "texte_extrait": "",
                "type_document": DocumentType.INCONNU.value,
                "score_confiance": 0.0,
                "temps_traitement": time.time() - start_time
            }

    def _process_text_input(self, text: str, start_time: float) -> Dict[str, Any]:
        """Process direct text input"""
        document_type = self.ocr_processor.detect_document_type(text)

        return {
            "texte_extrait": text,
            "type_document": document_type,
            "score_confiance": 1.0,
            "langue_detectee": "mixte",
            "temps_traitement": time.time() - start_time,
            "methode": "texte_direct",
            "metadonnees": {
                "nombre_caracteres": len(text),
                "nombre_mots": len(text.split())
            }
        }

    def _process_file_path(self, file_path: str, start_time: float) -> Dict[str, Any]:
        """Process file from file path"""
        file_path_obj = Path(file_path)

        if not file_path_obj.exists():
            raise FileNotFoundError(f"Fichier non trouvé : {file_path}")

        # Check file size
        file_size_mb = file_path_obj.stat().st_size / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            raise ValueError(f"Fichier trop volumineux : {file_size_mb:.1f}MB > {self.max_file_size_mb}MB")

        # Determine file type and process
        suffix = file_path_obj.suffix.lower()

        if suffix == '.pdf':
            extracted_text, metadata = self.ocr_processor.extract_text_from_pdf(file_path)
        elif suffix in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
            extracted_text, metadata = self.ocr_processor.extract_text_from_image(file_path)
        else:
            raise ValueError(f"Type de fichier non supporté : {suffix}")

        # Detect document type and validate
        document_type = self.ocr_processor.detect_document_type(extracted_text)
        confidence_score = self.ocr_processor.validate_extraction(extracted_text, metadata)

        return {
            "texte_extrait": extracted_text,
            "type_document": document_type,
            "score_confiance": confidence_score,
            "langue_detectee": "mixte",
            "temps_traitement": time.time() - start_time,
            "methode": metadata.get("method", "inconnu"),
            "metadonnees": metadata
        }

    def _process_file_content(self, file_content: bytes, file_type: str, start_time: float) -> Dict[str, Any]:
        """Process file from raw bytes"""
        extracted_text, metadata = self.ocr_processor.extract_from_bytes(file_content, file_type)

        # Detect document type and validate
        document_type = self.ocr_processor.detect_document_type(extracted_text)
        confidence_score = self.ocr_processor.validate_extraction(extracted_text, metadata)

        return {
            "texte_extrait": extracted_text,
            "type_document": document_type,
            "score_confiance": confidence_score,
            "langue_detectee": "mixte",
            "temps_traitement": time.time() - start_time,
            "methode": metadata.get("method", "inconnu"),
            "metadonnees": metadata
        }

    def act(self, execution_output: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        ACTION: Perform post-processing actions on extracted text

        Args:
            execution_output: Results from execution phase

        Returns:
            Optional[Dict[str, Any]]: Enhanced results with post-processing
        """
        if "erreur" in execution_output:
            return None

        extracted_text = execution_output.get("texte_extrait", "")

        # Text cleaning and enhancement
        cleaned_text = self._clean_extracted_text(extracted_text)

        # Extract key information using LLM
        key_info = self._extract_key_information(cleaned_text)

        return {
            "texte_nettoye": cleaned_text,
            "informations_cles": key_info,
            "statistiques_texte": self._calculate_text_stats(cleaned_text)
        }

    def _clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        import re

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Fix common OCR errors for French text
        replacements = {
            'œ': 'oe',
            'Œ': 'OE',
            '«': '"',
            '»': '"',
            ''': "'",
            ''': "'",
            '"': '"',
            '"': '"'
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        return text.strip()

#     def _extract_key_information(self, text: str) -> Dict[str, Any]:
#         """Extract key information using LLM"""
#         if not text.strip():
#             return {}
#
# #         extraction_prompt = f"""
# # Extrayez les informations clés de ce texte de document d'assurance :
# #
# # Texte : {text[:2000]}...
# #
# # Extrayez et structurez :
# # 1. Dates (toutes les dates mentionnées)
# # 2. Noms (personnes, entreprises)
# # 3. Adresses
# # 4. Informations véhicule (si applicable)
# # 5. Montants/Numéros
# # 6. Numéros de police/contrat
# # 7. Numéros de téléphone
# # 8. Adresses email
# #
# # Retournez sous forme de données structurées en format JSON.
# # """
#             extraction_prompt = f"""
# Extrayez les informations clés de ce texte de document d'assurance.
#
# Texte : {text[:2000]}...
#
# Répondez UNIQUEMENT en JSON valide avec cette structure:
# {{
#     "dates": ["liste des dates trouvées"],
#     "noms": ["noms de personnes/entreprises"],
#     "adresses": ["adresses complètes"],
#     "vehicules": ["informations véhicules"],
#     "montants": ["montants en euros"],
#     "contrats": ["numéros police/contrat"],
#     "telephones": ["numéros de téléphone"],
#     "emails": ["adresses email"]
# }}
#
# Si aucune information n'est trouvée pour une catégorie, utilisez une liste vide [].
# IMPORTANT: Répondez UNIQUEMENT avec du JSON valide, rien d'autre.
# """
#
#         try:
#             response = self.llm_client.chat.completions.create(
#                 model="gpt-4o-mini",
#                 messages=[
#                     {"role": "system",
#                      "content": "Vous êtes un expert en extraction d'informations structurées à partir de documents d'assurance. Retournez uniquement du JSON valide."},
#                     {"role": "user", "content": extraction_prompt}
#                 ],
#                 max_tokens=1000,
#                 temperature=0.1
#             )
#
#             # Parse JSON response
#             import json
#         #     key_info = json.loads(response.choices[0].message.content)
#         #     return key_info
#         #
#         # except Exception as e:
#         #     logger.warning(f"Error extracting key information: {str(e)}")
#         #     return {"erreur": "Impossible d'extraire les informations structurées"}
#             raw_response = response.choices[0].message.content.strip()
#             if not raw_response:
#                 logger.warning("LLM returned empty response")
#                 return {"erreur": "Réponse LLM vide"}
#
#             # Supprimer les markdown si présents
#             if raw_response.startswith("```json"):
#                 raw_response = raw_response.replace("```json", "").replace("```", "").strip()
#             elif raw_response.startswith("```"):
#                 raw_response = raw_response.replace("```", "").strip()
#
#             # Tentative de parsing JSON
#             try:
#                 key_info = json.loads(raw_response)
#                 logger.debug(f"Successfully extracted {len(key_info)} info categories")
#                 return key_info
#
#             except json.JSONDecodeError as json_err:
#                 logger.warning(f"JSON parsing failed: {json_err}")
#                 logger.debug(f"Raw response was: {raw_response[:200]}...")
#
#                 # Fallback: extraction basique par regex
#                 return self._extract_info_fallback(text)
#
#         except Exception as e:
#             logger.warning(f"Error extracting key information: {str(e)}")
#             return self._extract_info_fallback(text)

    def _extract_key_information(self, text: str) -> Dict[str, Any]:
        """Extract key information using LLM"""
        if not text.strip():
            return {}

        extraction_prompt = f"""
    Extrayez les informations clés de ce texte de document d'assurance.

    Texte : {text[:2000]}...

    Répondez UNIQUEMENT en JSON valide avec cette structure:
    {{
        "dates": ["liste des dates trouvées"],
        "noms": ["noms de personnes/entreprises"],
        "adresses": ["adresses complètes"],
        "vehicules": ["informations véhicules"],
        "montants": ["montants en euros"],
        "contrats": ["numéros police/contrat"],
        "telephones": ["numéros de téléphone"],
        "emails": ["adresses email"]
    }}

    Si aucune information n'est trouvée pour une catégorie, utilisez une liste vide [].
    IMPORTANT: Répondez UNIQUEMENT avec du JSON valide, rien d'autre.
    """

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": "Vous êtes un expert en extraction d'informations. Répondez UNIQUEMENT en JSON valide, aucun autre texte."},
                    {"role": "user", "content": extraction_prompt}
                ],
                max_tokens=800,
                temperature=0.1
            )

            raw_response = response.choices[0].message.content.strip()

            # Nettoyage de la réponse
            if not raw_response:
                logger.warning("LLM returned empty response")
                return self._extract_info_fallback(text)

            # Supprimer les markdown si présents
            if raw_response.startswith("```json"):
                raw_response = raw_response.replace("```json", "").replace("```", "").strip()
            elif raw_response.startswith("```"):
                raw_response = raw_response.replace("```", "").strip()

            # Tentative de parsing JSON
            try:
                import json
                key_info = json.loads(raw_response)
                logger.debug(f"Successfully extracted {len(key_info)} info categories")
                return key_info

            except json.JSONDecodeError as json_err:
                logger.warning(f"JSON parsing failed: {json_err}")
                logger.debug(f"Raw response was: {raw_response[:200]}...")

                # Fallback: extraction basique par regex
                return self._extract_info_fallback(text)

        except Exception as e:
            logger.warning(f"Error extracting key information: {str(e)}")
            return self._extract_info_fallback(text)
    def _extract_info_fallback(self, text: str) -> Dict[str, Any]:
        """Fallback extraction using regex patterns"""
        import re

        info = {
            "dates": [],
            "noms": [],
            "adresses": [],
            "vehicules": [],
            "montants": [],
            "contrats": [],
            "telephones": [],
            "emails": []
        }

        try:
            # Extraction des montants (€, euros)
            montants = re.findall(r'(\d+(?:[\s,]\d{3})*(?:[.,]\d{2})?)\s*(?:€|euros?)', text, re.IGNORECASE)
            info["montants"] = montants

            # Extraction des plaques d'immatriculation
            plaques = re.findall(r'[A-Z]{2}-\d{3}-[A-Z]{2}', text)
            if plaques:
                info["vehicules"] = plaques

            # Extraction des dates (format français)
            dates = re.findall(r'\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}', text)
            info["dates"] = dates

            # Extraction des numéros de téléphone
            telephones = re.findall(r'(?:0[1-9])(?:[-.\s]?\d{2}){4}', text)
            info["telephones"] = telephones

            # Extraction des emails
            emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
            info["emails"] = emails

            logger.debug(f"Fallback extraction completed: {sum(len(v) for v in info.values())} items found")

        except Exception as e:
            logger.error(f"Even fallback extraction failed: {e}")
            info["erreur"] = f"Extraction impossible: {str(e)}"

        return info


    def _calculate_text_stats(self, text: str) -> Dict[str, Any]:
        """Calculate text statistics"""
        words = text.split()

        return {
            "nombre_caracteres": len(text),
            "nombre_mots": len(words),
            "nombre_lignes": len(text.split('\n')),
            "longueur_mot_moyenne": sum(len(word) for word in words) / len(words) if words else 0,
            "contient_chiffres": any(char.isdigit() for char in text),
            "contient_caracteres_speciaux": any(not char.isalnum() and not char.isspace() for char in text)
        }

    def criticize(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        CRITIC: Evaluate OCR results and provide feedback

        Args:
            output_data: All output data to evaluate
        Returns:
            Dict[str, Any]: Critic feedback with confidence and retry recommendations
        """
        if "erreur" in output_data:
            return {
                "retour": f"OCR échoué : {output_data['erreur']}",
                "confiance": 0.0,
                "necessite_relance": True,
                "problemes": ["extraction_echouee"]
            }

        extracted_text = output_data.get("texte_extrait", "")
        confidence_score = output_data.get("score_confiance", 0.0)
        document_type = output_data.get("type_document", "inconnu")

        issues = []
        feedback_parts = []

        # Evaluate text quality
        if len(extracted_text) < 50:
            issues.append("texte_trop_court")
            feedback_parts.append("Le texte extrait est très court - peut indiquer une qualité OCR médiocre")

        # Evaluate confidence
        if confidence_score < self.min_confidence_threshold:
            issues.append("confiance_faible")
            feedback_parts.append(
                f"Confiance OCR ({confidence_score:.2f}) sous le seuil ({self.min_confidence_threshold})")

        # Evaluate document type detection
        if document_type == "inconnu":
            issues.append("type_document_inconnu")
            feedback_parts.append("Impossible de déterminer le type de document - peut affecter le traitement en aval")

        # Overall assessment
        if confidence_score >= 0.8 and len(extracted_text) > 100:
            feedback_parts.append("Extraction de haute qualité adaptée au traitement")
        elif confidence_score >= 0.6:
            feedback_parts.append("Extraction de qualité modérée - procédez avec prudence")
        else:
            feedback_parts.append("Extraction de faible qualité - considérez une révision manuelle")

        final_feedback = ". ".join(feedback_parts) if feedback_parts else "OCR terminé avec succès"
        needs_retry = confidence_score < 0.4 or len(extracted_text) < 20

        return {
            "retour": final_feedback,
            "confiance": confidence_score,
            "necessite_relance": needs_retry,
            "problemes": issues,
            "score_qualite": min(confidence_score + (len(extracted_text) / 1000), 1.0)
        }

    def terminate(self, output_data: Dict[str, Any], critic_feedback: str) -> Dict[str, Any]:
        """
        TERMINATION: Prepare final OCR results

        Args:
            output_data: All output data
            critic_feedback: Feedback from critic phase

        Returns:
            Dict[str, Any]: Final validated OCR results
        """
        if "erreur" in output_data:
            return {
                "succes": False,
                "erreur": output_data["erreur"],
                "resultat_ocr": None
            }

        # Create standardized OCR result
        ocr_result = OCRResult(
            texte_extrait=output_data.get("texte_extrait", ""),
            type_document=DocumentType(output_data.get("type_document", "inconnu")),
            score_confiance=output_data.get("score_confiance", 0.0),
            langue_detectee=output_data.get("langue_detectee", "inconnu"),
            metadonnees=output_data.get("metadonnees", {}),
            temps_traitement=output_data.get("temps_traitement", 0.0)
        )

        # Add enhanced data if available
        enhanced_data = {}
        if "texte_nettoye" in output_data:
            enhanced_data["texte_nettoye"] = output_data["texte_nettoye"]
        if "informations_cles" in output_data:
            enhanced_data["informations_cles"] = output_data["informations_cles"]
        if "statistiques_texte" in output_data:
            enhanced_data["statistiques_texte"] = output_data["statistiques_texte"]

        logger.info(
            f"OCR Agent completed - confidence: {ocr_result.score_confiance:.2f}, {len(ocr_result.texte_extrait)} chars extracted")

        return {
            "succes": True,
            "resultat_ocr": ocr_result.dict(),
            "donnees_ameliorees": enhanced_data,
            "resume_traitement": {
                "type_document": ocr_result.type_document.value,
                "confiance": ocr_result.score_confiance,
                "temps_traitement": ocr_result.temps_traitement,
                "nombre_caracteres": len(ocr_result.texte_extrait)
            }
        }