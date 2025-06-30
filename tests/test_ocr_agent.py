"""
Test script for OCR Agent with REACT pattern
Run from project root with: python -m pytest tests/test_ocr_agent.py -v
Or run directly with: python tests/test_ocr_agent.py
"""
import sys
import os
import json
from pathlib import Path

# Add project root and src to Python path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent  # Go up from tests/ to project root
src_path = project_root / "src"

# Add paths to Python path
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))

print(f"🔍 Project root: {project_root}")
print(f"📁 Source path: {src_path}")
print(f"✅ Structure validation:")
print(f"   - src/ exists: {src_path.exists()}")
print(f"   - config/ exists: {(project_root / 'config').exists()}")
print(f"   - tests/ exists: {(project_root / 'tests').exists()}")

# Import required modules
try:
    from openai import OpenAI

    print("✅ OpenAI imported")

    from config.settings import settings

    print("✅ Settings imported")

    # Import with absolute paths
    from src.agents.ocr_agent import OCRAgent

    print("✅ OCRAgent imported")

    from src.models.claim_models import ProcessingInput

    print("✅ Models imported")

except ImportError as e:
    print(f"❌ Import failed: {e}")

    # Try alternative import strategy
    try:
        print("🔄 Trying alternative import strategy...")

        # Make sure all paths are in sys.path
        sys.path.insert(0, str(project_root / "src"))

        import agents.ocr_agent as ocr_module

        OCRAgent = ocr_module.OCRAgent

        import models.claim_models as models_module

        ProcessingInput = models_module.ProcessingInput

        print("✅ Alternative imports successful")

    except ImportError as e2:
        print(f"❌ Alternative imports also failed: {e2}")
        print(f"💡 Current Python path: {sys.path[:3]}...")

        # Detailed debugging
        config_file = project_root / "config" / "settings.py"
        agent_file = src_path / "agents" / "ocr_agent.py"
        models_file = src_path / "models" / "claim_models.py"

        print(f"\n🔧 File check:")
        print(f"   - config/settings.py: {config_file.exists()}")
        print(f"   - src/agents/ocr_agent.py: {agent_file.exists()}")
        print(f"   - src/models/claim_models.py: {models_file.exists()}")

        print(f"\n🔍 Contents check:")
        if (src_path / "agents").exists():
            print(f"   - src/agents/ contents: {list((src_path / 'agents').iterdir())}")

        sys.exit(1)


def test_text_input():
    """Test OCR agent with direct text input"""
    print("\n🧪 Testing OCR Agent with Text Input")
    print("=" * 50)

    # Validate environment first
    if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY == "your_openai_api_key_here":
        print("❌ OpenAI API key not configured in .env file")
        return None

    # Initialize OpenAI client
    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    # Create OCR agent
    ocr_agent = OCRAgent(client)

    # Test input with French insurance scenario
    test_input = {
        "saisie_texte": "Ma voiture a été rayée sur un parking de supermarché hier soir. La rayure fait environ 20cm sur la portière avant droite. J'ai pris des photos et relevé la plaque du véhicule responsable: AB-123-CD. Contrat d'assurance numéro: AXA-2024-789456.",
        "chemin_fichier": None,
        "contenu_fichier": None,
        "type_fichier": None,
        "contexte_utilisateur": {"langue": "francais", "type_sinistre": "auto"}
    }

    print("📋 Test data prepared...")
    print(f"   - Text length: {len(test_input['saisie_texte'])} chars")
    print(f"   - Context: {test_input['contexte_utilisateur']}")

    # Execute REACT cycle
    print("\n⚙️ Executing REACT cycle...")
    result = ocr_agent.execute_react_cycle(test_input)

    # Display results
    print(f"\n📊 Results:")
    print(f"   - Status: {result.status.value}")
    print(f"   - Confidence: {result.confidence_score:.2f}")
    print(f"   - Actions taken: {', '.join(result.actions_taken)}")

    if result.output_data.get("succes"):
        ocr_result = result.output_data["resultat_ocr"]
        print(f"\n✅ OCR Success:")
        print(f"   - Extracted text: {ocr_result['texte_extrait'][:100]}...")
        print(f"   - Document type: {ocr_result['type_document']}")
        print(f"   - Processing time: {ocr_result['temps_traitement']:.2f}s")
        print(f"   - Confidence: {ocr_result['score_confiance']:.2f}")

        if "donnees_ameliorees" in result.output_data:
            enhanced = result.output_data["donnees_ameliorees"]
            if "informations_cles" in enhanced:
                print(f"\n📋 Key information extracted:")
                print(json.dumps(enhanced["informations_cles"], indent=2, ensure_ascii=False))
    else:
        print(f"\n❌ OCR Failed:")
        print(f"   - Error: {result.output_data.get('erreur', 'Erreur inconnue')}")

    print(f"\n🔍 Critic feedback:")
    print(f"   {result.critic_feedback}")

    return result


def create_sample_pdf():
    """Create a sample PDF for testing (optional)"""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter

        # Create sample PDF with French insurance content
        pdf_path = project_root / "data" / "input" / "sample_constat.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)

        c = canvas.Canvas(str(pdf_path), pagesize=letter)

        # Add sample constat content in French
        c.setFont("Helvetica", 12)
        c.drawString(100, 750, "CONSTAT AMIABLE D'ACCIDENT AUTOMOBILE")
        c.drawString(100, 720, "Date: 25/06/2025")
        c.drawString(100, 700, "Lieu: Parking Carrefour, Aulnay-sous-Bois")

        c.drawString(100, 660, "VÉHICULE A:")
        c.drawString(120, 640, "Conducteur: Jean DUPONT")
        c.drawString(120, 620, "Plaque: AB-123-CD")
        c.drawString(120, 600, "Assureur: AXA Assurances")
        c.drawString(120, 580, "Contrat: AXA-2024-789456")

        c.drawString(100, 540, "VÉHICULE B:")
        c.drawString(120, 520, "Conducteur: Marie MARTIN")
        c.drawString(120, 500, "Plaque: EF-456-GH")
        c.drawString(120, 480, "Assureur: MAIF")

        c.drawString(100, 440, "CIRCONSTANCES:")
        c.drawString(120, 420, "Collision lors d'une manoeuvre de stationnement")
        c.drawString(120, 400, "Dégâts: rayure sur portière avant droite véhicule A")

        c.save()

        print(f"✅ Sample PDF created: {pdf_path}")
        return str(pdf_path)

    except ImportError:
        print("⚠️ ReportLab not installed - skipping PDF creation")
        print("💡 Install with: pip install reportlab")
        return None


def test_pdf_input():
    """Test OCR agent with PDF input"""
    print("\n🧪 Testing OCR Agent with PDF Input")
    print("=" * 50)

    # Create sample PDF
    pdf_path = create_sample_pdf()
    if not pdf_path:
        print("❌ Skipping PDF test - no sample file")
        return None

    # Initialize OpenAI client
    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    # Create OCR agent
    ocr_agent = OCRAgent(client)

    # Test input
    test_input = {
        "saisie_texte": None,
        "chemin_fichier": pdf_path,
        "contenu_fichier": None,
        "type_fichier": "application/pdf",
        "contexte_utilisateur": {"langue": "francais", "type_attendu": "constat_amiable"}
    }

    print(f"📋 Testing with PDF: {Path(pdf_path).name}")

    # Execute REACT cycle
    result = ocr_agent.execute_react_cycle(test_input)

    # Display results (abbreviated)
    print(f"\n📊 Results:")
    print(f"   - Status: {result.status.value}")
    print(f"   - Confidence: {result.confidence_score:.2f}")

    if result.output_data.get("succes"):
        ocr_result = result.output_data["resultat_ocr"]
        print(f"\n✅ PDF OCR Success:")
        print(f"   - Extracted text: {ocr_result['texte_extrait'][:200]}...")
        print(f"   - Document type: {ocr_result['type_document']}")
        print(f"   - Processing time: {ocr_result['temps_traitement']:.2f}s")
    else:
        print(f"\n❌ PDF OCR Failed:")
        print(f"   - Error: {result.output_data.get('erreur', 'Erreur inconnue')}")

    return result


def main():
    """Run all OCR agent tests"""
    print("🚀 OCR Agent REACT Pattern Tests")
    print("=" * 60)

    # Environment validation
    try:
        if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY == "your_openai_api_key_here":
            print("❌ OpenAI API key not configured in .env file")
            print("💡 Please set your OPENAI_API_KEY in the .env file")
            return False

        print("✅ Environment validated")

        # Test 1: Text input
        text_result = test_text_input()

        # Test 2: PDF input (if possible)
        pdf_result = test_pdf_input()

        # Summary
        print("\n" + "=" * 60)
        print("📊 Test Summary")
        print("=" * 60)

        if text_result and text_result.status.value == "completed":
            print("✅ Traitement texte: RÉUSSI")
        else:
            print("❌ Traitement texte: ÉCHOUÉ")

        if pdf_result and pdf_result.status.value == "completed":
            print("✅ Traitement PDF: RÉUSSI")
        else:
            print("⚠️ Traitement PDF: IGNORÉ ou ÉCHOUÉ")

        print("\n🎯 Prochaines étapes:")
        print("1. Vérifiez la qualité des résultats OCR")
        print("2. Testez avec de vrais documents d'assurance")
        print("3. Passez à l'Étape 3: Agent de Classification")

        return True

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)