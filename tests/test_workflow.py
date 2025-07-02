"""
Test du Smart Workflow AXA - Validation complète LangGraph
Run from project root with: python tests/test_smart_workflow.py
"""
import sys
import time
from pathlib import Path

# Add project paths
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
src_path = project_root / "src"

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))

try:
    from openai import OpenAI
    from config.settings import settings
    # Import du workflow
    from src.workflows.simple_workflow import create_simple_workflow

    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)


class SimpleTestSuite:
    """Tests simplifiés pour validation rapide"""

    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY, timeout=30.0)
        self.workflow = create_simple_workflow(self.client)
        self.test_results = []

    def create_simple_scenarios(self) -> dict:
        """Scénarios simplifiés"""
        return {
            "accident_simple": {
                "name": "Accident Simple",
                "expected_final": "completed",
                "data": {
                    "saisie_texte": "Ma voiture a été rayée dans un parking. Rayure de 10cm sur la portière. Estimation: 400€. Pas de blessé.",
                    "chemin_fichier": None,
                    "contenu_fichier": None,
                    "type_fichier": None
                }
            },

            "cas_urgent": {
                "name": "Cas Urgent",
                "expected_final": "fast_track",
                "data": {
                    "saisie_texte": "URGENT - Accident avec blessé hospitalisé ce matin. Ambulance sur place. Véhicule détruit. Urgence médicale absolue.",
                    "chemin_fichier": None,
                    "contenu_fichier": None,
                    "type_fichier": None
                }
            },

            "montant_eleve": {
                "name": "Montant Élevé",
                "expected_final": "detailed_analysis",
                "data": {
                    "saisie_texte": "Incendie dans ma maison. Dégâts structurels importants. Estimation préliminaire: 50000€. Expertise requise.",
                    "chemin_fichier": None,
                    "contenu_fichier": None,
                    "type_fichier": None
                }
            },

            "description_vague": {
                "name": "Description Vague",
                "expected_final": "human_review",
                "data": {
                    "saisie_texte": "Il y a eu un problème. Quelque chose est cassé.",
                    "chemin_fichier": None,
                    "contenu_fichier": None,
                    "type_fichier": None
                }
            }
        }

    def run_test(self, scenario_key: str, scenario: dict) -> dict:
        """Test individuel simplifié"""
        print(f"\n🧪 Test: {scenario['name']}")
        print("-" * 30)

        start_time = time.time()

        try:
            result = self.workflow.execute(scenario['data'])
            execution_time = time.time() - start_time

            success = result.get('success', False)
            final_step = result.get('final_step', 'unknown')
            errors = result.get('errors', [])

            print(f"✅ Succès: {'Oui' if success else 'Non'}")
            print(f"🏁 Étape finale: {final_step}")
            print(f"⏱️ Temps: {execution_time:.2f}s")

            if errors:
                print(f"❌ Erreurs: {len(errors)}")
                for error in errors[:2]:  # Max 2 erreurs affichées
                    print(f"   - {error}")

            # Validation simple
            expected = scenario['expected_final']
            final_correct = expected in final_step or final_step == expected

            print(f"🎯 Attendu: {expected} -> {'✅' if final_correct else '❌'}")

            # Score simple
            score = 0
            if success:
                score += 5
            if final_correct:
                score += 3
            if execution_time < 15:
                score += 2

            print(f"📊 Score: {score}/10")

            return {
                "scenario": scenario_key,
                "name": scenario['name'],
                "success": success,
                "final_correct": final_correct,
                "execution_time": execution_time,
                "score": score,
                "final_step": final_step,
                "error_count": len(errors)
            }

        except Exception as e:
            print(f"❌ Erreur test: {str(e)}")
            return {
                "scenario": scenario_key,
                "name": scenario['name'],
                "success": False,
                "final_correct": False,
                "execution_time": time.time() - start_time,
                "score": 0,
                "final_step": "error",
                "error_count": 1
            }

    def run_all_tests(self):
        """Tous les tests"""
        print("🚀 Tests Simplifiés Workflow AXA")
        print("=" * 35)

        if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY == "your_openai_api_key_here":
            print("❌ OpenAI API key non configurée")
            return False

        scenarios = self.create_simple_scenarios()

        for scenario_key, scenario in scenarios.items():
            test_result = self.run_test(scenario_key, scenario)
            self.test_results.append(test_result)

        self._display_summary()
        return self._calculate_success()

    def _display_summary(self):
        """Résumé simple"""
        print(f"\n" + "=" * 40)
        print("📊 RÉSUMÉ FINAL")
        print("=" * 40)

        total = len(self.test_results)
        successful = sum(1 for r in self.test_results if r['success'])
        correct_final = sum(1 for r in self.test_results if r['final_correct'])
        avg_time = sum(r['execution_time'] for r in self.test_results) / total if total > 0 else 0
        avg_score = sum(r['score'] for r in self.test_results) / total if total > 0 else 0

        print(f"🧪 Tests: {total}")
        print(f"✅ Réussis: {successful}/{total} ({successful / total:.0%})")
        print(f"🎯 Fin correcte: {correct_final}/{total} ({correct_final / total:.0%})")
        print(f"⏱️ Temps moyen: {avg_time:.1f}s")
        print(f"📊 Score moyen: {avg_score:.1f}/10")

        print(f"\n📋 Détail:")
        for result in self.test_results:
            status = "✅" if result['success'] else "❌"
            final_status = "🎯" if result['final_correct'] else "❌"
            print(f"   {status}{final_status} {result['name']}: {result['score']}/10 ({result['execution_time']:.1f}s)")

        # Recommandation
        if avg_score >= 7:
            print(f"\n🎉 Workflow opérationnel! Prêt pour Streamlit")
        elif avg_score >= 5:
            print(f"\n✅ Workflow acceptable, peut continuer")
        else:
            print(f"\n⚠️ Workflow nécessite ajustements")

    def _calculate_success(self) -> bool:
        """Succès global simple"""
        if not self.test_results:
            return False

        avg_score = sum(r['score'] for r in self.test_results) / len(self.test_results)
        success_rate = sum(1 for r in self.test_results if r['success']) / len(self.test_results)

        return avg_score >= 5.0 and success_rate >= 0.75


def main():
    """Point d'entrée"""
    test_suite = SimpleTestSuite()
    success = test_suite.run_all_tests()

    print(f"\n🎯 Résultat: {'🎉 SUCCÈS' if success else '❌ ÉCHEC'}")

    if success:
        print("\n📝 Prochaine étape:")
        print("   🎨 Créer interface Streamlit")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)