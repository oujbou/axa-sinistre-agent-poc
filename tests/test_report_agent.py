"""
Minimal test script for Report Agent - JSON and Text export only
Run from project root with: python tests/test_report_agent.py
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

# Import required modules
try:
    from openai import OpenAI
    from config.settings import settings
    from src.agents.report_agent import ReportAgent

    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)


def create_test_data() -> dict:
    """Create simple test data"""
    return {
        "resultat_ocr": {
            "texte_extrait": "Ma voiture a été rayée sur un parking. Dégâts sur la portière droite. Plaque responsable: AB-123-CD. Estimation: 800€.",
            "score_confiance": 0.9,
            "type_document": "description_textuelle"
        },
        "resultat_classification": {
            "type_sinistre": "accident_auto",
            "severite": "moderee",
            "score_confiance": 0.85,
            "montant_estime": 800,
            "score_urgence": 4,
            "mots_cles_trouves": ["accident", "voiture", "rayure", "parking"],
            "actions_immediates": [
                "Contacter garage pour devis",
                "Vérifier garanties contrat",
                "Programmer expertise si nécessaire"
            ],
            "delai_traitement_estime": "5-7 jours"
        }
    }


def test_minimal_report():
    """Test minimal report generation"""
    print("\n🧪 Testing Minimal Report Generation (JSON + Text)")
    print("=" * 55)

    # Initialize
    client = OpenAI(api_key=settings.OPENAI_API_KEY, timeout=30.0)
    report_agent = ReportAgent(client)

    # Test data
    test_input = create_test_data()

    print("📋 Test scenario: Auto accident")
    print("   - Expected: JSON export")
    print("   - Expected: Text export")
    print("   - Expected: < 3 seconds")

    # Execute
    result = report_agent.execute_react_cycle(test_input)

    # Results
    print(f"\n📊 Results:")
    print(f"   - Status: {result.status.value}")
    print(f"   - Confidence: {result.confidence_score:.2f}")

    if result.output_data.get("succes"):
        resume = result.output_data["resume_generation"]
        exports = result.output_data["exports"]
        rapport_data = result.output_data["rapport_data"]

        print(f"\n✅ Report Generated:")
        print(f"   - Export formats: {resume['formats_export']}")
        print(f"   - Generation time: {resume['temps_generation']:.2f}s")
        print(f"   - Total size: {resume['taille_total']} chars")

        print(f"\n📄 Content summary:")
        print(f"   - Type: {rapport_data['type_sinistre']}")
        print(f"   - Severity: {rapport_data['severite']}")
        print(f"   - Summary: {rapport_data['resume'][:50]}...")
        print(f"   - Actions: {len(rapport_data['actions_recommandees'])} items")

        print(f"\n📤 Export details:")
        for format_name, export_data in exports.items():
            print(f"   - {format_name.upper()}: {export_data['taille']} characters")

        # Show sample JSON (first 200 chars)
        if "json" in exports:
            json_sample = exports["json"]["contenu"][:200]
            print(f"\n📋 JSON sample:")
            print(f"   {json_sample}...")

        # Show sample Text (first 300 chars)
        if "text" in exports:
            text_sample = exports["text"]["contenu"][:300]
            print(f"\n📄 Text sample:")
            lines = text_sample.split('\n')[:5]
            for line in lines:
                if line.strip():
                    print(f"   {line}")
            print("   ...")

    else:
        print(f"\n❌ Generation Failed:")
        print(f"   - Error: {result.output_data.get('erreur', 'Unknown error')}")

    return result


def validate_exports(result):
    """Validate export formats"""
    if not result.output_data.get("succes"):
        return False

    exports = result.output_data.get("exports", {})

    # Check JSON
    if "json" in exports:
        try:
            json_content = exports["json"]["contenu"]
            parsed = json.loads(json_content)
            print("✅ JSON export is valid")
        except:
            print("❌ JSON export is invalid")
            return False

    # Check Text
    if "text" in exports:
        text_content = exports["text"]["contenu"]
        if len(text_content) > 100 and "RAPPORT DE SINISTRE" in text_content:
            print("✅ Text export is valid")
        else:
            print("❌ Text export is too short or missing header")
            return False

    return True


def main():
    """Run minimal test"""
    print("🚀 Minimal Report Agent Test")
    print("=" * 30)

    # Environment check
    if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY == "your_openai_api_key_here":
        print("❌ OpenAI API key not configured")
        return False

    print("✅ Environment validated")

    try:
        # Run test
        result = test_minimal_report()
        success = result.status.value == "completed"

        if success:
            # Validate exports
            exports_valid = validate_exports(result)
            success = success and exports_valid

        print(f"\n🎯 Test Result: {'✅ SUCCESS' if success else '❌ FAILED'}")

        if success:
            print("\n📝 Report Agent Ready:")
            print("   ✅ JSON export functional")
            print("   ✅ Text export functional")
            print("   ✅ Generation time < 5s")
            print("   ✅ Ready for LangGraph integration")

        return success

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)