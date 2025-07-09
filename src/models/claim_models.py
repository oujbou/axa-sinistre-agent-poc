"""
Data models for AXA Claims Processing System
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime

class DocumentType(Enum):
    """Types of documents that can be processed"""
    DESCRIPTION_TEXTUELLE = "description_textuelle"
    CONSTAT_AMIABLE = "constat_amiable"
    FORMULAIRE_ASSURANCE = "formulaire_assurance"
    FACTURE = "facture"
    PHOTO = "photo"
    RAPPORT_MEDICAL = "rapport_medical"
    PROCES_VERBAL = "proces_verbal"
    INCONNU = "inconnu"

class ClaimType(Enum):
    """Types of insurance claims"""
    ACCIDENT_AUTO = "accident_auto"
    DEGATS_HABITATION = "degats_habitation"
    SANTE = "sante"
    VOYAGE = "voyage"
    RESPONSABILITE_CIVILE = "responsabilite_civile"
    VOL = "vol"
    INCENDIE = "incendie"
    DEGAT_DES_EAUX = "degat_des_eaux"
    CATASTROPHE_NATURELLE = "catastrophe_naturelle"
    INCONNU = "inconnu"

class Severity(Enum):
    """Claim severity levels"""
    FAIBLE = "faible"
    MODEREE = "moderee"
    ELEVEE = "elevee"
    CRITIQUE = "critique"

class OCRResult(BaseModel):
    """Result from OCR processing"""
    texte_extrait: str = Field(..., description="Texte extrait du document")
    type_document: DocumentType = Field(..., description="Type de document détecté")
    score_confiance: float = Field(..., ge=0.0, le=1.0, description="Confiance de l'OCR")
    langue_detectee: str = Field(..., description="Langue détectée")
    metadonnees: Dict[str, Any] = Field(default_factory=dict, description="Métadonnées OCR supplémentaires")
    temps_traitement: float = Field(..., description="Temps de traitement en secondes")

class ClaimClassification(BaseModel):
    """Result from claim classification"""
    type_sinistre: ClaimType = Field(..., description="Type de sinistre classifié")
    severite: Severity = Field(..., description="Sévérité estimée")
    score_confiance: float = Field(..., ge=0.0, le=1.0, description="Confiance de la classification")
    mots_cles_trouves: List[str] = Field(default_factory=list, description="Mots-clés pertinents identifiés")
    montant_estime: Optional[float] = Field(None, description="Montant estimé du sinistre si déterminable")
    necessite_enquete: bool = Field(..., description="Si une enquête détaillée est nécessaire")
    score_urgence: int = Field(..., ge=1, le=10, description="Score d'urgence (1=faible, 10=critique)")

    methode_classification: str = Field(..., description="Méthode utilisée: rules, llm, hybride")
    temps_traitement: float = Field(..., description="Temps de traitement en secondes")
    raisonnement_llm: Optional[str] = Field(None, description="Explication du raisonnement IA")

    # Business-specific flags
    necessite_expertise: bool = Field(default=False, description="Si expertise technique requise")
    potentiel_fraude: bool = Field(default=False, description="Indicateurs de fraude détectés")
    urgence_medicale: bool = Field(default=False, description="Urgence médicale identifiée")


    # Workflow information
    niveau_autorisation: str = Field(default="agent", description="Niveau d'autorisation requis")
    delai_traitement_estime: str = Field(default="1-2sem", description="Délai estimé de traitement")
    actions_immediates: List[str] = Field(default_factory=list, description="Actions à prendre immédiatement")


class ClaimReport(BaseModel):
    """Final claim processing report"""
    id_sinistre: str = Field(..., description="Identifiant unique du sinistre")
    type_sinistre: ClaimType = Field(..., description="Type de sinistre")
    severite: Severity = Field(..., description="Sévérité du sinistre")
    resume: str = Field(..., description="Résumé exécutif du sinistre")
    informations_extraites: Dict[str, Any] = Field(..., description="Informations clés extraites")
    actions_recommandees: List[str] = Field(..., description="Prochaines étapes recommandées")
    temps_traitement_estime: str = Field(..., description="Temps estimé pour traiter")
    necessite_revision_humaine: bool = Field(..., description="Si une révision humaine est requise")
    score_confiance: float = Field(..., ge=0.0, le=1.0, description="Confiance globale")
    genere_le: datetime = Field(default_factory=datetime.now, description="Date de génération du rapport")

class ProcessingInput(BaseModel):
    """Input for the claim processing pipeline"""
    saisie_texte: Optional[str] = Field(None, description="Saisie directe de texte")
    chemin_fichier: Optional[str] = Field(None, description="Chemin vers le fichier uploadé")
    contenu_fichier: Optional[bytes] = Field(None, description="Contenu brut du fichier")
    type_fichier: Optional[str] = Field(None, description="Type MIME du fichier")
    contexte_utilisateur: Dict[str, Any] = Field(default_factory=dict, description="Contexte utilisateur supplémentaire")

class AgentResponse(BaseModel):
    """Standardized response from any agent"""
    nom_agent: str = Field(..., description="Nom de l'agent qui répond")
    succes: bool = Field(..., description="Si l'opération a réussi")
    donnees: Dict[str, Any] = Field(..., description="Données de réponse")
    confiance: float = Field(..., ge=0.0, le=1.0, description="Confiance dans la réponse")
    erreurs: List[str] = Field(default_factory=list, description="Erreurs rencontrées")
    temps_traitement: float = Field(..., description="Temps de traitement en secondes")
    metadonnees: Dict[str, Any] = Field(default_factory=dict, description="Métadonnées supplémentaires")

class WorkflowState(BaseModel):
    """State shared across the entire workflow"""
    id_session: str = Field(..., description="Identifiant unique de session")
    etape_courante: str = Field(..., description="Étape de traitement courante")
    donnees_entree: ProcessingInput = Field(..., description="Données d'entrée originales")
    resultat_ocr: Optional[OCRResult] = Field(None, description="Résultat du traitement OCR")
    resultat_classification: Optional[ClaimClassification] = Field(None, description="Résultat de la classification")
    rapport_final: Optional[ClaimReport] = Field(None, description="Rapport final du sinistre")
    erreurs: List[str] = Field(default_factory=list, description="Erreurs accumulées")
    demarre_le: datetime = Field(default_factory=datetime.now, description="Heure de début du workflow")
    termine_le: Optional[datetime] = Field(None, description="Heure de fin du workflow")

# French insurance-specific terms for better classification
MOTS_CLES_ASSURANCE_FRANCAISE = {
    ClaimType.ACCIDENT_AUTO: [
        "accident", "collision", "constat", "véhicule", "voiture", "auto",
        "carambolage", "choc", "impact", "dégâts matériels", "blessé",
        "parking", "rayure", "pare-choc", "carrosserie", "conducteur",
        "permis de conduire", "plaque d'immatriculation", "assurance auto"
    ],
    ClaimType.DEGATS_HABITATION: [
        "domicile", "habitation", "maison", "appartement", "logement",
        "dégât des eaux", "inondation", "fuite", "toiture", "fenêtre",
        "porte", "mur", "plafond", "sol", "propriétaire", "locataire",
        "multirisque habitation", "copropriété"
    ],
    ClaimType.VOL: [
        "vol", "cambriolage", "effraction", "disparition", "dérobé",
        "volé", "soustraction", "larcin", "brigandage", "vandalisme",
        "tentative de vol", "plainte", "commissariat", "gendarmerie"
    ],
    ClaimType.INCENDIE: [
        "incendie", "feu", "brûlé", "combustion", "fumée", "flamme",
        "sinistre incendie", "dégâts par le feu", "pompiers", "explosion",
        "court-circuit", "surchauffe"
    ],
    ClaimType.DEGAT_DES_EAUX: [
        "dégât des eaux", "inondation", "fuite", "infiltration", "humidité",
        "débordement", "rupture de canalisation", "eau", "plomberie",
        "robinet", "radiateur", "machine à laver", "lave-vaisselle"
    ],
    ClaimType.SANTE: [
        "maladie", "accident corporel", "hospitalisation", "médecin",
        "ordonnance", "soins", "frais médicaux", "pharmacie", "dentiste",
        "optique", "kinésithérapeute", "arrêt de travail"
    ],
    ClaimType.VOYAGE: [
        "voyage", "vacances", "bagages", "annulation", "retard", "vol annulé",
        "rapatriement", "assistance", "étranger", "maladie en voyage",
        "perte bagages", "accident à l'étranger"
    ]
}