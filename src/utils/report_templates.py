"""
Report templates for AXA insurance claims
Intelligent templates with conditional sections and business logic
"""
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

try:
    from models.claim_models import ClaimType, Severity
except ImportError:
    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.models.claim_models import ClaimType, Severity


class ReportTemplate:
    """Base class for report templates"""

    def __init__(self, claim_type: ClaimType):
        self.claim_type = claim_type
        self.sections_obligatoires = []
        self.sections_conditionnelles = {}
        self.calculs_automatiques = {}
        self.mentions_legales = []

    def generate_structure(self, classification_data: Dict) -> Dict[str, Any]:
        """Generate report structure based on classification data"""
        raise NotImplementedError


class AccidentAutoTemplate(ReportTemplate):
    """Template for auto accident reports"""

    def __init__(self):
        super().__init__(ClaimType.ACCIDENT_AUTO)
        self.sections_obligatoires = [
            "resume_executif",
            "circonstances_accident",
            "vehicules_impliques",
            "degats_constates",
            "responsabilites",
            "estimation_indemnisation",
            "prochaines_etapes"
        ]

        self.sections_conditionnelles = {
            "expertise_technique": lambda data: data.get("montant_estime", 0) > 5000,
            "procedure_penale": lambda data: "blessé" in data.get("texte_original", "").lower(),
            "recours_tiers": lambda data: "responsabilité partagée" in data.get("justification", ""),
            "enquete_complementaire": lambda data: data.get("flags", {}).get("potentiel_fraude", False),
            "urgence_medicale": lambda data: data.get("flags", {}).get("urgence_medicale", False)
        }

        self.mentions_legales = [
            "Déclaration à effectuer dans les 5 jours ouvrés",
            "Droit de recours selon article L211-1 du Code des assurances",
            "Expertise contradictoire possible selon article L125-3"
        ]

    def generate_structure(self, classification_data: Dict) -> Dict[str, Any]:
        """Generate auto accident report structure"""

        montant = classification_data.get("montant_estime", 0)
        severite = classification_data.get("severite", "moderee")
        urgence = classification_data.get("score_urgence", 5)

        structure = {
            "titre": "RAPPORT DE SINISTRE AUTOMOBILE",
            "sous_titre": f"Référence: AXA-AUTO-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "sections": {}
        }

        # Sections obligatoires
        structure["sections"]["resume_executif"] = {
            "titre": "Résumé Exécutif",
            "template": "accident_auto_resume",
            "priorite": 1,
            "variables": {
                "type_sinistre": "Accident automobile",
                "severite": severite.upper(),
                "montant_estime": f"{montant:,.0f}€" if montant else "À évaluer",
                "urgence": f"{urgence}/10"
            }
        }

        structure["sections"]["circonstances"] = {
            "titre": "Circonstances de l'Accident",
            "template": "accident_auto_circonstances",
            "priorite": 2,
            "variables": {
                "date_sinistre": "À préciser",
                "lieu_sinistre": "À préciser",
                "conditions_meteo": "À préciser",
                "temoins": "À identifier"
            }
        }

        structure["sections"]["vehicules"] = {
            "titre": "Véhicules Impliqués",
            "template": "accident_auto_vehicules",
            "priorite": 3,
            "variables": {
                "vehicule_assure": "À détailler",
                "vehicule_tiers": "À détailler",
                "degats_vehicule_a": "À évaluer",
                "degats_vehicule_b": "À évaluer"
            }
        }

        structure["sections"]["responsabilites"] = {
            "titre": "Analyse des Responsabilités",
            "template": "accident_auto_responsabilites",
            "priorite": 4,
            "variables": {
                "responsabilite_assure": "À déterminer",
                "responsabilite_tiers": "À déterminer",
                "constat_amiable": "À vérifier"
            }
        }

        structure["sections"]["estimation"] = {
            "titre": "Estimation et Indemnisation",
            "template": "accident_auto_estimation",
            "priorite": 5,
            "variables": {
                "montant_degats": f"{montant:,.0f}€" if montant else "À évaluer",
                "franchise": self._calculate_franchise(montant),
                "delai_reparation": self._estimate_repair_time(montant, severite)
            }
        }

        structure["sections"]["actions"] = {
            "titre": "Prochaines Étapes",
            "template": "accident_auto_actions",
            "priorite": 6,
            "variables": {
                "actions_immediates": classification_data.get("actions_immediates", []),
                "delai_traitement": classification_data.get("delai_traitement_estime", "5-10 jours"),
                "contact_principal": "Service Sinistres AXA Auto"
            }
        }

        # Sections conditionnelles
        if self.sections_conditionnelles["expertise_technique"](classification_data):
            structure["sections"]["expertise"] = {
                "titre": "Expertise Technique Requise",
                "template": "accident_auto_expertise",
                "priorite": 7,
                "variables": {
                    "type_expertise": "Expertise contradictoire recommandée",
                    "delai_expertise": "10-15 jours ouvrés",
                    "expert_designe": "À désigner"
                }
            }

        if self.sections_conditionnelles["urgence_medicale"](classification_data):
            structure["sections"]["medical"] = {
                "titre": "Urgence Médicale",
                "template": "accident_auto_medical",
                "priorite": 1.5,  # Insert après résumé
                "variables": {
                    "soins_urgents": "Prise en charge médicale prioritaire",
                    "contact_medical": "Service Médical AXA - 01 XX XX XX XX",
                    "garanties_sante": "Vérification garanties corporelles"
                }
            }

        return structure

    def _calculate_franchise(self, montant: Optional[float]) -> str:
        """Calculate franchise based on amount"""
        if not montant:
            return "À calculer selon contrat"

        # Franchise dégressive exemple AXA
        if montant < 1000:
            return "150€"
        elif montant < 3000:
            return "300€"
        elif montant < 10000:
            return "500€"
        else:
            return "750€"

    def _estimate_repair_time(self, montant: Optional[float], severite: str) -> str:
        """Estimate repair timeline"""
        if not montant:
            return "À évaluer après expertise"

        if severite == "critique" or montant > 15000:
            return "3-6 semaines (dégâts importants)"
        elif severite == "elevee" or montant > 5000:
            return "2-4 semaines (réparations moyennes)"
        elif severite == "moderee" or montant > 1000:
            return "1-2 semaines (réparations standard)"
        else:
            return "3-7 jours (réparations mineures)"


class DegatsHabitationTemplate(ReportTemplate):
    """Template for home damage reports"""

    def __init__(self):
        super().__init__(ClaimType.DEGATS_HABITATION)
        self.sections_obligatoires = [
            "resume_executif",
            "description_logement",
            "origine_sinistre",
            "degats_constates",
            "mesures_urgence",
            "estimation_travaux",
            "prochaines_etapes"
        ]

        self.sections_conditionnelles = {
            "expertise_batiment": lambda data: data.get("montant_estime", 0) > 10000,
            "relogement_temporaire": lambda data: "inhabitable" in data.get("texte_original", "").lower(),
            "recours_voisinage": lambda data: "voisin" in data.get("texte_original", "").lower(),
            "procedures_urgence": lambda data: data.get("score_urgence", 5) >= 8
        }

    def generate_structure(self, classification_data: Dict) -> Dict[str, Any]:
        """Generate home damage report structure"""

        montant = classification_data.get("montant_estime", 0)
        severite = classification_data.get("severite", "moderee")

        structure = {
            "titre": "RAPPORT DE SINISTRE HABITATION",
            "sous_titre": f"Référence: AXA-HAB-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "sections": {}
        }

        structure["sections"]["resume_executif"] = {
            "titre": "Résumé Exécutif",
            "template": "habitation_resume",
            "priorite": 1,
            "variables": {
                "type_sinistre": "Dégâts habitation",
                "severite": severite.upper(),
                "montant_estime": f"{montant:,.0f}€" if montant else "À évaluer",
                "habitabilite": self._assess_habitability(classification_data)
            }
        }

        structure["sections"]["logement"] = {
            "titre": "Description du Logement",
            "template": "habitation_logement",
            "priorite": 2,
            "variables": {
                "type_logement": "À préciser",
                "surface": "À préciser",
                "annee_construction": "À préciser",
                "occupants": "À préciser"
            }
        }

        structure["sections"]["origine"] = {
            "titre": "Origine du Sinistre",
            "template": "habitation_origine",
            "priorite": 3,
            "variables": {
                "cause_probable": self._determine_cause(classification_data),
                "date_survenance": "À préciser",
                "decouvert_par": "À préciser"
            }
        }

        return structure

    def _assess_habitability(self, classification_data: Dict) -> str:
        """Assess if home is still habitable"""
        severite = classification_data.get("severite", "moderee")
        texte = classification_data.get("texte_original", "").lower()

        if severite == "critique" or any(word in texte for word in ["inhabitable", "évacuation", "danger"]):
            return "Inhabitable - Relogement nécessaire"
        elif severite == "elevee":
            return "Habitabilité réduite - Surveillance requise"
        else:
            return "Habitable avec précautions"

    def _determine_cause(self, classification_data: Dict) -> str:
        """Determine probable cause of damage"""
        texte = classification_data.get("texte_original", "").lower()

        if "fuite" in texte or "eau" in texte:
            return "Dégât des eaux probable"
        elif "feu" in texte or "incendie" in texte:
            return "Dommage par le feu"
        elif "vol" in texte or "effraction" in texte:
            return "Acte de vandalisme/vol"
        else:
            return "Cause à déterminer"


class GenericTemplate(ReportTemplate):
    """Generic template for unknown or complex claims"""

    def __init__(self, claim_type: ClaimType):
        super().__init__(claim_type)
        self.sections_obligatoires = [
            "resume_executif",
            "description_sinistre",
            "analyse_preliminaire",
            "actions_requises",
            "recommandations"
        ]

    def generate_structure(self, classification_data: Dict) -> Dict[str, Any]:
        """Generate generic report structure"""

        structure = {
            "titre": "RAPPORT DE SINISTRE",
            "sous_titre": f"Référence: AXA-GEN-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "sections": {}
        }

        structure["sections"]["resume_executif"] = {
            "titre": "Résumé Exécutif",
            "template": "generic_resume",
            "priorite": 1,
            "variables": {
                "type_sinistre": classification_data.get("type_sinistre", "À classifier"),
                "confiance_classification": f"{classification_data.get('score_confiance', 0):.1%}",
                "revision_requise": "Oui" if classification_data.get("score_confiance", 0) < 0.7 else "Non"
            }
        }

        structure["sections"]["description"] = {
            "titre": "Description du Sinistre",
            "template": "generic_description",
            "priorite": 2,
            "variables": {
                "texte_original": classification_data.get("texte_original", ""),
                "elements_identifies": classification_data.get("mots_cles_trouves", [])
            }
        }

        return structure


class TemplateEngine:
    """Central template engine for report generation"""

    def __init__(self):
        self.templates = {
            ClaimType.ACCIDENT_AUTO: AccidentAutoTemplate(),
            ClaimType.DEGATS_HABITATION: DegatsHabitationTemplate(),
        }
        self.generic_template = GenericTemplate

        # Content templates for sections
        self.content_templates = {
            "accident_auto_resume": """
SINISTRE: {type_sinistre}
GRAVITÉ: {severite}
ESTIMATION: {montant_estime}
URGENCE: {urgence}

Ce rapport présente l'analyse complète du sinistre automobile déclaré. 
L'évaluation préliminaire indique une gravité {severite} avec une estimation de {montant_estime}.
            """,

            "accident_auto_circonstances": """
LIEU: {lieu_sinistre}
DATE: {date_sinistre}
CONDITIONS: {conditions_meteo}
TÉMOINS: {temoins}

Les circonstances exactes de l'accident sont en cours de vérification. 
Une reconstitution détaillée sera effectuée si nécessaire.
            """,

            "habitation_resume": """
SINISTRE: {type_sinistre}
GRAVITÉ: {severite}
ESTIMATION: {montant_estime}
HABITABILITÉ: {habitabilite}

Ce rapport concerne un sinistre habitation nécessitant une évaluation approfondie.
L'habitabilité du logement a été évaluée comme: {habitabilite}.
            """
        }

    def get_template(self, claim_type: ClaimType) -> ReportTemplate:
        """Get appropriate template for claim type"""
        if claim_type in self.templates:
            return self.templates[claim_type]
        else:
            return self.generic_template(claim_type)

    def render_section(self, template_name: str, variables: Dict[str, Any]) -> str:
        """Render section content with variables"""
        if template_name in self.content_templates:
            template = self.content_templates[template_name]
            try:
                return template.format(**variables)
            except KeyError as e:
                return f"Erreur template: variable manquante {e}"
        else:
            return f"Template {template_name} non trouvé"

    def calculate_business_dates(self, claim_type: ClaimType, urgency: int) -> Dict[str, str]:
        """Calculate business-relevant dates"""
        now = datetime.now()

        # Délais légaux selon type de sinistre
        if claim_type == ClaimType.ACCIDENT_AUTO:
            declaration_limit = now + timedelta(days=5)
            prescription = now + timedelta(days=365 * 2)  # 2 ans
        elif claim_type == ClaimType.DEGATS_HABITATION:
            declaration_limit = now + timedelta(days=5)
            prescription = now + timedelta(days=365 * 2)
        else:
            declaration_limit = now + timedelta(days=5)
            prescription = now + timedelta(days=365 * 2)

        # Délais de traitement selon urgence
        if urgency >= 8:
            traitement_target = now + timedelta(hours=24)
        elif urgency >= 6:
            traitement_target = now + timedelta(days=3)
        else:
            traitement_target = now + timedelta(days=10)

        return {
            "date_declaration_limite": declaration_limit.strftime("%d/%m/%Y"),
            "date_prescription": prescription.strftime("%d/%m/%Y"),
            "date_traitement_cible": traitement_target.strftime("%d/%m/%Y %H:%M"),
            "delai_reponse_client": "48h pour accusé réception"
        }