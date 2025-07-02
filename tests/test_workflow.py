"""
Test du Smart Workflow AXA - Validation complète LangGraph
Run from project root with: python tests/test_smart_workflow.py
"""
import sys
import time
from pathlib import Path
from datetime import datetime

# Add project paths
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
src_path = project_root / "src"

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))

try:
    from openai import OpenAI
    from config.settings import settings
    from src.workflows.simple_workflow import create_smart_workflow

    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)


class WorkflowTestSuite:
    """Suite de tests pour le workflow intelligent"""

    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY, timeout=30.0)
        self.workflow = create_smart_workflow(self.client)
        self.test_results = []

    def create_test_scenarios(self) -> dict:
        """Créer les scénarios de test"""
        return {
            "simple_accident": {
                "name": "Accident Simple",
                "description": "Rayure parking, faible montant",
                "expected_route": ["ocr", "classification", "report"],
                "expected_final": "completed",
                "data": {
                    "saisie_texte": "Ma voiture a été rayée dans un parking hier. Rayure de 15cm sur la portière droite. Pas de blessé. Estimation garage: 450€. Responsable inconnu.",
                    "chemin_fichier": None,
                    "contenu_fichier": None,
                    "type_fichier": None,
                    "contexte_utilisateur": {"type_attendu": "accident_auto"}
                }
            },

            "degats_eau_urgent": {
                "name": "Dégâts des Eaux Urgent",
                "description": "Fuite importante, montant élevé, urgence",
                "expected_route": ["ocr", "classification"],
                "expected_final": "detailed_analysis",
                "data": {
                    "saisie_texte": "URGENT - Dégât des eaux massif ce matin ! Canalisation principale a cédé. Inondation complète appartement + voisins. Dégâts estimés 25000€. Plombier sur place. Expertise urgente requise.",
                    "chemin_fichier": None,
                    "contenu_fichier": None,
                    "type_fichier": None,
                    "contexte_utilisateur": {"urgence": "elevee"}
                }
            },

            "urgence_medicale": {
                "name": "Urgence Médicale",
                "description": "Accident avec blessés, traitement prioritaire",
                "expected_route": ["ocr", "classification"],
                "expected_final": "fast_track",
                "data": {
                    "saisie_texte": "Accident grave ce matin à 8h. Collision frontale, j'ai été hospitalisé en urgence. Ambulance, pompiers. Blessures importantes mais stables. Mon véhicule est détruit. Urgence médicale absolue.",
                    "chemin_fichier": None,
                    "contenu_fichier": None,
                    "type_fichier": None,
                    "contexte_utilisateur": {"urgence": "medicale"}
                }
            },

            "description_vague": {
                "name": "Description Vague",
                "description": "Informations insuffisantes, révision humaine",
                "expected_route": ["ocr", "classification"],
                "expected_final": "human_review",
                "data": {
                    "saisie_texte": "Il y a eu un problème. Quelque chose s'est cassé. Il faut réparer. Je ne sais pas combien ça va coûter.",
                    "chemin_fichier": None,
                    "contenu_fichier": None,
                    "type_fichier": None,
                    "contexte_utilisateur": {}
                }
            },

            "cas_complexe": {
                "name": "Cas Complexe Multi-Facteurs",
                "description": "Montant élevé + enquête + expertise",
                "expected_route": ["ocr", "classification"],
                "expected_final": "detailed_analysis",
                "data": {
                    "saisie_texte": "Incendie dans ma maison hier soir. Origine suspecte - investigation police en cours. Dégâts structurels importants, maison inhabitable. Estimation préliminaire: 180000€. Expertise technique et enquête fraude nécessaires.",
                    "chemin_fichier": None,
                    "contenu_fichier": None,
                    "type_fichier": None,
                    "contexte_utilisateur": {"complexite": "elevee"}
                }
            }
        }

    def run_single_test(self, scenario_key: str, scenario: dict) -> dict:
        """Exécuter un test individuel"""
        print(f"\n🧪 Test: {scenario['name']}")
        print("=" * 50)

        start_time = time.time()

        print(f"📋 Description: {scenario['description']}")
        print(f"🎯 Route attendue: {' → '.join(scenario['expected_route'])}")
        print(f"🏁 Fin attendue: {scenario['expected_final']}")

        # Exécution du workflow
        try:
            print(f"\n⚙️ Exécution du workflow...")
            result = self.workflow.execute(scenario['data'])

            execution_time = time.time() - start_time

            # Analyse des résultats
            success = result.get('success', False)
            final_step = result.get('final_step', 'unknown')
            routing_decisions = result.get('routing_decisions', [])
            workflow_path = result.get('workflow_path', [])

            print(f"\n📊 Résultats:")
            print(f"   ✅ Succès: {'Oui' if success else 'Non'}")
            print(f"   🏁 Étape finale: {final_step}")
            print(f"   ⏱️ Temps d'exécution: {execution_time:.2f}s")
            print(f"   🔀 Décisions de routage: {len(routing_decisions)}")

            # Affichage du chemin pris
            if workflow_path:
                print(f"\n🛣️ Chemin parcouru:")
                print(f"   {' → '.join(workflow_path)}")

            # Détail des décisions de routage
            if routing_decisions:
                print(f"\n🧠 Décisions de routage:")
                for i, decision in enumerate(routing_decisions, 1):
                    print(f"   {i}. Après {decision['step']}: {decision['decision']}")
                    print(f"      Raison: {decision['reason']}")

                    # Facteurs métier si disponibles
                    if 'factors' in decision:
                        factors = decision['factors']
                        print(f"      Facteurs: conf={factors.get('confidence', 0):.2f}, "
                              f"montant={factors.get('montant', 0)}€, "
                              f"urgence={factors.get('urgence', 0)}")

            # Validation des attentes
            route_valid = self._validate_route(workflow_path, scenario['expected_route'])
            final_valid = scenario['expected_final'] in final_step or final_step == scenario['expected_final']

            print(f"\n✅ Validation:")
            print(f"   🛣️ Route correcte: {'Oui' if route_valid else 'Non'}")
            print(f"   🏁 Fin correcte: {'Oui' if final_valid else 'Non'}")

            # Résultats spécifiques selon le type de fin
            self._display_specific_results(result, final_step)

            # Score du test
            test_score = self._calculate_test_score(success, route_valid, final_valid, execution_time)

            print(f"\n🎯 Score du test: {test_score:.1f}/10")

            return {
                "scenario": scenario_key,
                "name": scenario['name'],
                "success": success,
                "route_valid": route_valid,
                "final_valid": final_valid,
                "execution_time": execution_time,
                "test_score": test_score,
                "final_step": final_step,
                "routing_decisions": len(routing_decisions),
                "errors": result.get('errors', [])
            }

        except Exception as e:
            print(f"\n❌ Erreur durant le test: {str(e)}")
            return {
                "scenario": scenario_key,
                "name": scenario['name'],
                "success": False,
                "route_valid": False,
                "final_valid": False,
                "execution_time": time.time() - start_time,
                "test_score": 0.0,
                "final_step": "error",
                "routing_decisions": 0,
                "errors": [str(e)]
            }

    def _validate_route(self, actual_path: list, expected_route: list) -> bool:
        """Valider que le chemin contient les étapes attendues"""
        if not actual_path:
            return False

        # Vérifier que toutes les étapes attendues sont présentes
        return all(step in actual_path for step in expected_route)

    def _display_specific_results(self, result: dict, final_step: str):
        """Afficher les résultats spécifiques selon le type de finalisation"""
        results = result.get('results', {})

        if final_step == "fast_track" and 'fast_report' in results:
            fast_data = results['fast_report']
            print(f"\n🚀 Résultat Fast Track:")
            print(f"   - ID: {fast_data.get('id_rapport', 'N/A')}")
            print(f"   - Traitement: {fast_data.get('traitement', 'N/A')}")
            print(f"   - Confiance: {fast_data.get('confiance', 0):.2f}")

        elif final_step == "detailed_analysis" and 'detailed_report' in results:
            detail_data = results['detailed_report']
            print(f"\n🔬 Résultat Analyse Détaillée:")
            print(f"   - ID: {detail_data.get('id_rapport', 'N/A')}")
            print(f"   - Analyse approfondie: {detail_data.get('analyse_approfondie', False)}")
            print(f"   - Impact montant: €{detail_data.get('montant_impact', 0):,.0f}")
            print(f"   - Délai: {detail_data.get('delai_traitement', 'N/A')}")

        elif final_step == "human_review" and 'human_review' in results:
            review_data = results['human_review']
            print(f"\n👤 Résultat Révision Humaine:")
            print(f"   - ID: {review_data.get('id_rapport', 'N/A')}")
            print(f"   - Raison escalade: {review_data.get('raison_escalade', 'N/A')}")
            print(f"   - Statut: {review_data.get('statut', 'N/A')}")

        elif final_step == "completed" and 'report' in results:
            print(f"\n📋 Résultat Standard:")
            report_data = results['report'].get('rapport_data', {})
            print(f"   - Type: {report_data.get('type_sinistre', 'N/A')}")
            print(f"   - Sévérité: {report_data.get('severite', 'N/A')}")
            print(f"   - Délai: {report_data.get('delai_traitement', 'N/A')}")

    def _calculate_test_score(self, success: bool, route_valid: bool, final_valid: bool,
                              execution_time: float) -> float:
        """Calculer le score du test"""
        score = 0.0

        # Succès de base (4 points)
        if success:
            score += 4.0

        # Routage correct (3 points)
        if route_valid:
            score += 3.0

        # Finalisation correcte (2 points)
        if final_valid:
            score += 2.0

        # Performance (1 point)
        if execution_time < 10:
            score += 1.0
        elif execution_time < 20:
            score += 0.5

        return score

    def run_performance_test(self):
        """Test de performance rapide"""
        print(f"\n⚡ Test de Performance")
        print("=" * 30)

        simple_data = {
            "saisie_texte": "Test rapide - rayure voiture, 300€",
            "chemin_fichier": None,
            "contenu_fichier": None,
            "type_fichier": None,
            "contexte_utilisateur": {}
        }

        # Mesure du temps
        start_time = time.time()
        result = self.workflow.execute(simple_data)
        execution_time = time.time() - start_time

        print(f"📊 Résultats Performance:")
        print(f"   - Temps total: {execution_time:.2f}s")
        print(f"   - Succès: {'✅' if result.get('success') else '❌'}")

        # Évaluation performance
        if execution_time < 5:
            print(f"   🚀 Performance excellente (< 5s)")
        elif execution_time < 10:
            print(f"   ✅ Performance bonne (< 10s)")
        elif execution_time < 20:
            print(f"   ⚠️ Performance acceptable (< 20s)")
        else:
            print(f"   ❌ Performance lente (> 20s)")

        return {
            "execution_time": execution_time,
            "success": result.get('success', False)
        }

    def run_all_tests(self):
        """Exécuter tous les tests"""
        print("🚀 Suite de Tests Smart Workflow AXA")
        print("=" * 45)

        # Vérification environnement
        if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY == "your_openai_api_key_here":
            print("❌ OpenAI API key non configurée")
            return False

        print("✅ Environnement validé")
        print(f"🕒 Début des tests: {datetime.now().strftime('%H:%M:%S')}")

        # Tests des scénarios
        scenarios = self.create_test_scenarios()

        for scenario_key, scenario in scenarios.items():
            test_result = self.run_single_test(scenario_key, scenario)
            self.test_results.append(test_result)

        # Test de performance
        perf_result = self.run_performance_test()

        # Résumé final
        self._display_final_summary(perf_result)

        return self._calculate_overall_success()

    def _display_final_summary(self, perf_result: dict):
        """Afficher le résumé final"""
        print(f"\n" + "=" * 60)
        print("📊 RÉSUMÉ FINAL DES TESTS")
        print("=" * 60)

        # Statistiques globales
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r['success'])
        valid_routes = sum(1 for r in self.test_results if r['route_valid'])
        valid_finals = sum(1 for r in self.test_results if r['final_valid'])
        avg_time = sum(r['execution_time'] for r in self.test_results) / total_tests
        avg_score = sum(r['test_score'] for r in self.test_results) / total_tests

        print(f"🧪 Tests exécutés: {total_tests}")
        print(f"✅ Tests réussis: {successful_tests}/{total_tests} ({successful_tests / total_tests:.1%})")
        print(f"🛣️ Routages corrects: {valid_routes}/{total_tests} ({valid_routes / total_tests:.1%})")
        print(f"🏁 Finalisations correctes: {valid_finals}/{total_tests} ({valid_finals / total_tests:.1%})")
        print(f"⏱️ Temps moyen: {avg_time:.2f}s")
        print(f"🎯 Score moyen: {avg_score:.1f}/10")

        # Détail par test
        print(f"\n📋 Détail par test:")
        for result in self.test_results:
            status = "✅" if result['success'] else "❌"
            route_status = "🛣️" if result['route_valid'] else "❌"
            final_status = "🏁" if result['final_valid'] else "❌"

            print(f"   {status}{route_status}{final_status} {result['name']}: "
                  f"{result['test_score']:.1f}/10 ({result['execution_time']:.1f}s)")

        # Performance
        print(f"\n⚡ Performance globale:")
        print(f"   - Test rapide: {perf_result['execution_time']:.2f}s")
        print(f"   - Workflow optimisé: {'✅' if perf_result['execution_time'] < 10 else '⚠️'}")

        # Recommandations
        print(f"\n💡 Recommandations:")
        if avg_score >= 8:
            print("   🎉 Workflow excellemment opérationnel!")
            print("   📝 Prêt pour l'interface Streamlit")
        elif avg_score >= 6:
            print("   ✅ Workflow fonctionnel avec améliorations mineures")
            print("   🔧 Optimiser le routage pour certains cas")
        else:
            print("   ⚠️ Workflow nécessite des ajustements")
            print("   🔧 Revoir la logique de routage")

    def _calculate_overall_success(self) -> bool:
        """Calculer le succès global"""
        if not self.test_results:
            return False

        avg_score = sum(r['test_score'] for r in self.test_results) / len(self.test_results)
        success_rate = sum(1 for r in self.test_results if r['success']) / len(self.test_results)

        return avg_score >= 6.0 and success_rate >= 0.8


def main():
    """Point d'entrée principal"""
    test_suite = WorkflowTestSuite()
    success = test_suite.run_all_tests()

    print(f"\n🎯 Résultat global: {'🎉 SUCCÈS' if success else '❌ ÉCHEC'}")

    if success:
        print("\n📝 Prochaines étapes:")
        print("   1. ✅ Workflow LangGraph opérationnel")
        print("   2. 🎨 Créer l'interface Streamlit")
        print("   3. 🚀 Préparer la démonstration AXA")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)