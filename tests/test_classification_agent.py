"""
Test script for Classification Agent with REACT pattern
Run from project root with: python tests/test_classification_agent.py
"""
import sys
import json
from pathlib import Path

# Add project root and src to Python path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
src_path = project_root / "src"

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))

print(f"🔍 Project root: {project_root}")
print(f"📁 Source path: {src_path}")

# Import required modules
try:
    from openai import OpenAI

    print("✅ OpenAI imported")

    from config.settings import settings

    print("✅ Settings imported")

    from src.agents.classification_agent import ClassificationAgent

    print("✅ ClassificationAgent imported")

    from src.models.claim_models import ProcessingInput, ClaimType, Severity

    print("✅ Models imported")

except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)


def create_mock_ocr_result(texte: str, document_type: str = "description_textuelle") -> dict:
    """Create mock OCR result for testing"""
    return {
        "resultat_ocr": {
            "texte_extrait": texte,
            "type_document": document_type,
            "score_confiance": 0.95,
            "langue_detectee": "français",
            "temps_traitement": 1.2,
            "metadonnees": {
                "methode": "texte_direct",
                "nombre_caracteres": len(texte),
                "nombre_mots": len(texte.split())
            }
        },
        "donnees_ameliorees": {
            "texte_nettoye": texte,
            "statistiques_texte": {
                "nombre_caracteres": len(texte),
                "nombre_mots": len(texte.split())
            }
        }
    }


def test_auto_accident_classification():
    """Test classification of auto accident scenario"""
    print("\n🚗 Testing Auto Accident Classification")
    print("=" * 50)

    # Initialize OpenAI client
    client = OpenAI(api_key=settings.OPENAI_API_KEY, timeout=30.0)

    # Create classification agent
    classification_agent = ClassificationAgent(client)

    # Test scenario: Auto accident with specific details
    accident_text = """
    Accident de voiture hier soir vers 20h dans le parking du Carrefour d'Aulnay-sous-Bois.
    J'étais en train de me garer quand le véhicule B (plaque EF-456-GH) a reculé et percuté 
    ma portière avant droite. Dégâts visibles: rayure profonde de 25cm et enfoncement léger.
    Le conducteur du véhicule B (M. Martin) a reconnu sa responsabilité. 
    Constat amiable rempli sur place. Aucun blessé à déplorer.
    Mon véhicule: Renault Clio, plaque AB-123-CD, contrat AXA n°2024-789456.
    Estimation garage: environ 800€ de réparation.
    """

    # Create mock OCR input
    test_input = create_mock_ocr_result(accident_text, "constat_amiable")

    print("📋 Test scenario prepared:")
    print(f"   - Text length: {len(accident_text)} characters")
    print(f"   - Expected type: ACCIDENT_AUTO")
    print(f"   - Expected severity: FAIBLE-MODEREE (800€)")

    # Execute REACT cycle
    print("\n⚙️ Executing Classification REACT cycle...")
    result = classification_agent.execute_react_cycle(test_input)

    # Display results
    print(f"\n📊 Classification Results:")
    print(f"   - Status: {result.status.value}")
    print(f"   - Overall confidence: {result.confidence_score:.2f}")
    print(f"   - Actions taken: {', '.join(result.actions_taken)}")

    if result.output_data.get("succes"):
        classification = result.output_data["resultat_classification"]
        enhanced = result.output_data["donnees_ameliorees"]

        print(f"\n✅ Classification Success:")
        print(f"   - Type détecté: {classification['type_sinistre']}")
        print(f"   - Sévérité: {classification['severite']}")
        print(f"   - Confiance: {classification['score_confiance']:.2f}")
        print(f"   - Score urgence: {classification['score_urgence']}/10")
        print(f"   - Montant estimé: {classification.get('montant_estime', 'N/A')}€")
        print(f"   - Méthode: {classification['methode_classification']}")
        print(f"   - Temps traitement: {classification['temps_traitement']:.2f}s")

        print(f"\n🎯 Mots-clés identifiés:")
        for keyword in classification['mots_cles_trouves']:
            print(f"      - {keyword}")

        print(f"\n🚨 Indicateurs spéciaux:")
        indicators = enhanced['indicateurs_speciaux']
        for key, value in indicators.items():
            status = "✅" if value else "❌"
            print(f"      {status} {key}: {value}")

        print(f"\n📋 Actions immédiates:")
        for action in enhanced['actions_immediates']:
            print(f"      - {action}")

        print(f"\n🔍 Workflow recommandé: {enhanced.get('workflow_recommande', 'N/A')}")

    else:
        print(f"\n❌ Classification Failed:")
        print(f"   - Error: {result.output_data.get('erreur', 'Erreur inconnue')}")

    print(f"\n💬 Critic feedback:")
    print(f"   {result.critic_feedback}")

    return result


def test_water_damage_classification():
    """Test classification of water damage scenario"""
    print("\n💧 Testing Water Damage Classification")
    print("=" * 50)

    client = OpenAI(api_key=settings.OPENAI_API_KEY, timeout=30.0)
    classification_agent = ClassificationAgent(client)

    water_damage_text = """
    Dégât des eaux important dans mon appartement ce matin vers 6h.
    La canalisation sous l'évier de la cuisine a cédé pendant la nuit.
    Fuite massive qui a inondé la cuisine et le salon. 
    Eau partout sur le parquet, meubles trempés, électroménager endommagé.
    J'ai coupé l'eau et l'électricité par sécurité. 
    Plombier d'urgence appelé, il arrive dans 2h.
    Dégâts estimés: cuisine complète à refaire (15000€), parquet salon (3000€),
    électroménager (2000€). Total estimé: 20000€.
    Assurance habitation AXA contrat MRH-2024-456789.
    Urgence car l'eau continue de s'infiltrer chez les voisins du dessous.
    """

    test_input = create_mock_ocr_result(water_damage_text, "formulaire_assurance")

    print("📋 Test scenario: Major water damage")
    print(f"   - Expected type: DEGAT_DES_EAUX")
    print(f"   - Expected severity: CRITIQUE (20k€)")
    print(f"   - Expected urgency: HIGH (neighbor impact)")

    # Execute classification
    result = classification_agent.execute_react_cycle(test_input)

    # Quick results display
    if result.output_data.get("succes"):
        classification = result.output_data["resultat_classification"]
        print(f"\n✅ Water Damage Results:")
        print(f"   - Type: {classification['type_sinistre']}")
        print(f"   - Severity: {classification['severite']}")
        print(f"   - Urgency: {classification['score_urgence']}/10")
        print(f"   - Investigation required: {classification['necessite_enquete']}")
        print(f"   - Estimated amount: {classification.get('montant_estime', 'N/A')}€")
    else:
        print(f"❌ Water damage classification failed")

    return result


def test_ambiguous_classification():
    """Test classification of ambiguous scenario"""
    print("\n❓ Testing Ambiguous Classification")
    print("=" * 50)

    client = OpenAI(api_key=settings.OPENAI_API_KEY, timeout=30.0)
    classification_agent = ClassificationAgent(client)

    ambiguous_text = """
    Problème avec ma propriété hier. Il y a eu des dégâts.
    Quelque chose s'est cassé et maintenant c'est endommagé.
    Il faut réparer. Ce n'était pas de ma faute.
    Je pense que ça va coûter cher. Peut-être quelques milliers d'euros.
    J'ai des photos si besoin. Que dois-je faire maintenant?
    """

    test_input = create_mock_ocr_result(ambiguous_text, "inconnu")

    print("📋 Test scenario: Deliberately vague description")
    print(f"   - Expected type: INCONNU or low confidence")
    print(f"   - Expected: Manual review required")

    result = classification_agent.execute_react_cycle(test_input)

    if result.output_data.get("succes"):
        classification = result.output_data["resultat_classification"]
        enhanced = result.output_data["donnees_ameliorees"]

        print(f"\n🤔 Ambiguous Results:")
        print(f"   - Type: {classification['type_sinistre']}")
        print(f"   - Confidence: {classification['score_confiance']:.2f}")
        print(f"   - Needs investigation: {classification['necessite_enquete']}")
        print(f"   - Workflow: {enhanced.get('workflow_recommande', 'N/A')}")

    return result


def main():
    """Run all classification tests"""
    print("🚀 Classification Agent REACT Pattern Tests")
    print("=" * 60)

    # Environment validation
    if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY == "your_openai_api_key_here":
        print("❌ OpenAI API key not configured in .env file")
        return False

    print("✅ Environment validated")

    # Run test scenarios
    test_results = []

    try:
        # Test 1: Auto accident (clear case)
        auto_result = test_auto_accident_classification()
        test_results.append(("Auto Accident", auto_result.status.value == "completed"))

        # Test 2: Water damage (high severity)
        water_result = test_water_damage_classification()
        test_results.append(("Water Damage", water_result.status.value == "completed"))

        # Test 3: Ambiguous case
        ambiguous_result = test_ambiguous_classification()
        test_results.append(("Ambiguous Case", ambiguous_result.status.value == "completed"))

        # Summary
        print("\n" + "=" * 60)
        print("📊 Test Summary")
        print("=" * 60)

        for test_name, success in test_results:
            status = "✅ RÉUSSI" if success else "❌ ÉCHOUÉ"
            print(f"{status} {test_name}")

        success_rate = sum(1 for _, success in test_results if success) / len(test_results)
        print(f"\n🎯 Taux de réussite: {success_rate:.1%}")

        print("\n🎯 Prochaines étapes:")
        print("1. Vérifiez la qualité des classifications")
        print("2. Testez avec des cas réels d'assurance")
        print("3. Passez à l'Étape 4: Agent Rapporteur")

        return success_rate > 0.5

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)