"""
OCR utilities for document processing
"""
import fitz
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import io
import os
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
from loguru import logger
import time
import cv2
import numpy as np

from config.settings import settings


class OCRProcessor:
    """
    OCR processor with multiple backends and preprocessing
    Optimized for insurance documents (French/English)
    """

    def __init__(self):
        self.tesseract_path = settings.TESSERACT_PATH
        if self.tesseract_path and Path(self.tesseract_path).exists():
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_path

        self.languages = settings.OCR_LANGUAGES
        logger.info(f"OCR Processor initialized with languages: {self.languages}")

    def extract_text_from_pdf(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Extract text from PDF using PyMuPDF

        Args:
            file_path: Path to PDF file

        Returns:
            Tuple[str, Dict]: Extracted text and metadata
        """
        start_time = time.time()

        try:
            doc = fitz.open(file_path)
            full_text = ""
            metadata = {
                "page_count": len(doc),
                "method": "pymupdf",
                "file_size": os.path.getsize(file_path),
                "pages_processed": []
            }

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()

                if text.strip():
                    # Text-based PDF
                    full_text += f"\\n--- Page {page_num + 1} ---\\n{text}"
                    metadata["pages_processed"].append({
                        "page": page_num + 1,
                        "method": "text_extraction",
                        "char_count": len(text)
                    })
                else:
                    # Image-based PDF - use OCR
                    logger.debug(f"Page {page_num + 1} is image-based, using OCR")
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x resolution for better OCR
                    img_data = pix.tobytes("png")

                    # Convert to PIL Image for preprocessing
                    image = Image.open(io.BytesIO(img_data))
                    preprocessed_image = self._preprocess_image(image)

                    # OCR with Tesseract
                    ocr_text = pytesseract.image_to_string(
                        preprocessed_image,
                        lang=self.languages,
                        config='--oem 3 --psm 6'  # Best for documents
                    )

                    full_text += f"\\n--- Page {page_num + 1} (OCR) ---\\n{ocr_text}"
                    metadata["pages_processed"].append({
                        "page": page_num + 1,
                        "method": "ocr",
                        "char_count": len(ocr_text)
                    })

            doc.close()

            processing_time = time.time() - start_time
            metadata["processing_time"] = processing_time

            logger.info(f"PDF processed in {processing_time:.2f}s - {len(full_text)} characters extracted")
            return full_text.strip(), metadata

        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise

    def extract_text_from_image(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Extract text from image file using Tesseract OCR

        Args:
            file_path: Path to image file

        Returns:
            Tuple[str, Dict]: Extracted text and metadata
        """
        start_time = time.time()

        try:
            # Load and preprocess image
            image = Image.open(file_path)
            preprocessed_image = self._preprocess_image(image)

            # Extract text with confidence scores
            data = pytesseract.image_to_data(
                preprocessed_image,
                lang=self.languages,
                config='--oem 3 --psm 6',
                output_type=pytesseract.Output.DICT
            )

            # Filter confident text
            confident_text = []
            confidences = []

            for i, conf in enumerate(data['conf']):
                if int(conf) > 30:  # Confidence threshold
                    text = data['text'][i].strip()
                    if text:
                        confident_text.append(text)
                        confidences.append(int(conf))

            full_text = ' '.join(confident_text)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            processing_time = time.time() - start_time

            metadata = {
                "method": "tesseract_ocr",
                "file_size": os.path.getsize(file_path),
                "processing_time": processing_time,
                "average_confidence": avg_confidence,
                "words_detected": len(confident_text),
                "image_size": image.size
            }

            logger.info(f"Image OCR completed in {processing_time:.2f}s - confidence: {avg_confidence:.1f}%")
            return full_text, metadata

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for better OCR results

        Args:
            image: PIL Image to preprocess

        Returns:
            Image.Image: Preprocessed image
        """
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')

        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)

        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)

        # Apply slight blur to reduce noise
        image = image.filter(ImageFilter.MedianFilter(size=3))

        # Scale up if image is small (better OCR for small text)
        width, height = image.size
        if width < 1000 or height < 1000:
            scale_factor = 2
            new_size = (width * scale_factor, height * scale_factor)
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        return image

    def detect_document_type(self, text: str) -> str:
        """
        Detect document type based on extracted text

        Args:
            text: Extracted text to analyze

        Returns:
            str: Detected document type
        """
        text_lower = text.lower()

        # Define detection patterns
        patterns = {
            "constat_amiable": [
                "constat amiable", "constat d'accident", "déclaration d'accident",
                "véhicule a", "véhicule b", "conducteur", "assuré", "circonstances",
                "dégâts matériels", "témoins", "croquis"
            ],
            "formulaire_assurance": [
                "déclaration de sinistre", "formulaire", "police d'assurance",
                "numéro de contrat", "souscripteur", "bénéficiaire", "prime",
                "échéance", "garanties"
            ],
            "facture": [
                "facture", "devis", "montant", "tva", "total", "ht", "ttc",
                "réparation", "garage", "facture n°", "établissement"
            ],
            "rapport_medical": [
                "certificat médical", "rapport médical", "médecin",
                "diagnostic", "prescription", "arrêt de travail", "patient",
                "traitement", "hospitalisation"
            ],
            "proces_verbal": [
                "procès-verbal", "police", "gendarmerie", "commissariat",
                "main courante", "dépôt de plainte", "officier", "agent"
            ]
        }

        # Score each document type
        scores = {}
        for doc_type, keywords in patterns.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                scores[doc_type] = score

        # Return best match or unknown
        if scores:
            best_match = max(scores, key=scores.get)
            logger.debug(f"Type de document détecté : {best_match} (score: {scores[best_match]})")
            return best_match

        return "inconnu"

    def extract_from_bytes(self, file_bytes: bytes, file_type: str) -> Tuple[str, Dict[str, Any]]:
        """
        Extract text from file bytes

        Args:
            file_bytes: Raw file content
            file_type: MIME type of the file

        Returns:
            Tuple[str, Dict]: Extracted text and metadata
        """
        # Create temporary file
        temp_path = Path(settings.DATA_DIR) / "temp" / f"temp_file_{int(time.time())}"
        temp_path.parent.mkdir(exist_ok=True)

        try:
            # Write bytes to temporary file
            with open(temp_path, 'wb') as f:
                f.write(file_bytes)

            # Process based on file type
            if file_type.startswith('application/pdf'):
                return self.extract_text_from_pdf(str(temp_path))
            elif file_type.startswith('image/'):
                return self.extract_text_from_image(str(temp_path))
            else:
                raise ValueError(f"Type de fichier non supporté : {file_type}")

        finally:
            # Clean up temporary file
            if temp_path.exists():
                temp_path.unlink()

    def validate_extraction(self, text: str, metadata: Dict[str, Any]) -> float:
        """
        Validate extraction quality and return confidence score

        Args:
            text: Extracted text
            metadata: Extraction metadata

        Returns:
            float: Confidence score (0.0 to 1.0)
        """
        if not text.strip():
            return 0.0

        # Basic text quality checks
        confidence_factors = []

        # Length factor
        length_score = min(len(text) / 100, 1.0)  # More text = higher confidence
        confidence_factors.append(length_score)

        # Character diversity
        unique_chars = len(set(text.lower()))
        diversity_score = min(unique_chars / 20, 1.0)
        confidence_factors.append(diversity_score)

        # French/English word detection
        french_words = ["le", "la", "les", "de", "du", "des", "et", "à", "un", "une"]
        english_words = ["the", "and", "of", "to", "a", "in", "for", "is", "on", "that"]

        word_score = 0
        text_words = text.lower().split()
        for word in french_words + english_words:
            if word in text_words:
                word_score += 1

        language_score = min(word_score / 10, 1.0)
        confidence_factors.append(language_score)

        # OCR-specific confidence if available
        if "average_confidence" in metadata:
            ocr_confidence = metadata["average_confidence"] / 100
            confidence_factors.append(ocr_confidence)

        # Return weighted average
        final_confidence = sum(confidence_factors) / len(confidence_factors)
        return round(final_confidence, 3)