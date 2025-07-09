"""
🚀 AXA SINISTRE AGENT - Interface Streamlit Professionnelle
Démonstration POC Multi-Agents avec LangGraph
"""
import streamlit as st
import time
import json
import io
from datetime import datetime
from pathlib import Path
import sys

# Configuration de la page
st.set_page_config(
    page_title="AXA Sinistre Agent",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour design professionnel AXA
st.markdown("""
<style>
    /* Style AXA */
    .main-header {
        background: linear-gradient(90deg, #00008f 0%, #0066cc 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }

    .workflow-step {
        background: #f8f9fa;
        border-left: 4px solid #00008f;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }

    .step-active {
        background: #e3f2fd;
        border-left-color: #2196f3;
        animation: pulse 2s infinite;
    }

    .step-completed {
        background: #e8f5e8;
        border-left-color: #4caf50;
    }

    .step-error {
        background: #ffebee;
        border-left-color: #f44336;
    }

    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }

    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }

    .upload-zone {
        border: 2px dashed #00008f;
        border-radius: 10px;
        padding: 40px;
        text-align: center;
        background: #fafafa;
        margin: 20px 0;
    }

    .result-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    .confidence-bar {
        background: #e0e0e0;
        border-radius: 10px;
        overflow: hidden;
        height: 20px;
        margin: 10px 0;
    }

    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #ff5252 0%, #ff9800 50%, #4caf50 100%);
        transition: width 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)


# Configuration des chemins
@st.cache_resource
def setup_paths():
    """Configuration des chemins du projet"""
    current_dir = Path(__file__).parent
    project_root = current_dir
    src_path = project_root / "src"

    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    return project_root, src_path


# Initialisation
project_root, src_path = setup_paths()


# Import du workflow
@st.cache_resource
def load_workflow():
    """Charge le workflow AXA"""
    try:
        from openai import OpenAI
        from config.settings import settings
        from workflows.simple_workflow import create_simple_workflow

        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        workflow = create_simple_workflow(client)
        return workflow, True, "Workflow chargé avec succès"
    except ImportError as e:
        return None, False, f"Erreur d'import: {str(e)}"
    except Exception as e:
        return None, False, f"Erreur de configuration: {str(e)}"


def main():
    """Interface principale"""

    # En-tête AXA
    st.markdown("""
    <div class="main-header">
        <h1>🏢 AXA SINISTRE AGENT</h1>
        <h3>Système Multi-Agents Intelligent de Traitement des Sinistres</h3>
        <p>POC - Démonstration LangGraph avec Pattern REACT</p>
    </div>
    """, unsafe_allow_html=True)

    # Chargement du workflow
    workflow, workflow_ok, workflow_msg = load_workflow()

    if not workflow_ok:
        st.error(f"❌ Erreur de chargement: {workflow_msg}")
        st.info("🔧 Vérifiez que la configuration OpenAI est correcte dans settings.py")
        return

    st.success(f"✅ {workflow_msg}")

    # Sidebar - Configuration et informations
    with st.sidebar:
        st.markdown("### 🔧 Configuration")

        # Mode de démo
        demo_mode = st.selectbox(
            "Mode de démonstration",
            ["Production", "Démo rapide", "Debug détaillé"],
            index=1
        )

        # Seuils de confidence
        st.markdown("### 📊 Seuils")
        confidence_threshold = st.slider("Seuil de confiance", 0.0, 1.0, 0.7, 0.1)

        # Statistiques de session
        if 'session_stats' not in st.session_state:
            st.session_state.session_stats = {
                'total_processed': 0,
                'successful': 0,
                'avg_time': 0
            }

        stats = st.session_state.session_stats
        st.markdown("### 📈 Statistiques Session")
        st.metric("Sinistres traités", stats['total_processed'])
        st.metric("Taux de succès", f"{(stats['successful'] / max(stats['total_processed'], 1) * 100):.1f}%")
        st.metric("Temps moyen", f"{stats['avg_time']:.1f}s")

        # Informations système
        st.markdown("### ℹ️ Système")
        st.info("""
        **Architecture:**
        - OCR Agent (Tesseract + PyMuPDF)
        - Classification Agent (Rules + LLM)
        - Report Agent (JSON + Text)
        - LangGraph Orchestration

        **Modèle:** GPT-4o-mini
        **Langues:** Français, Anglais
        """)

    # Interface principale - Onglets
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Déclaration", "⚙️ Traitement", "📊 Résultats", "📋 Historique"])

    with tab1:
        declaration_interface()

    with tab2:
        if st.session_state.get('processing_data'):
            processing_interface(workflow, demo_mode)
        else:
            st.info("👆 Commencez par saisir un sinistre dans l'onglet Déclaration")

    with tab3:
        if st.session_state.get('workflow_result'):
            results_interface()
        else:
            st.info("⚙️ Lancez un traitement pour voir les résultats")

    with tab4:
        history_interface()


def declaration_interface():
    """Interface de déclaration de sinistre"""
    st.markdown("## 📝 Déclaration de Sinistre")

    # Méthode de saisie
    input_method = st.radio(
        "Comment souhaitez-vous déclarer le sinistre ?",
        ["💬 Saisie texte", "📄 Upload fichier", "🎯 Exemple rapide"],
        horizontal=True
    )

    processing_data = None

    if input_method == "💬 Saisie texte":
        st.markdown("### ✍️ Décrivez votre sinistre")

        text_input = st.text_area(
            "Description détaillée du sinistre",
            placeholder="Exemple: Ma voiture a été rayée dans un parking ce matin. Rayure de 15cm sur la portière droite. Estimation garage: 450€. Pas de blessé, responsable inconnu.",
            height=150
        )

        if st.button("🚀 Traiter ce sinistre", type="primary", use_container_width=True):
            if text_input.strip():
                processing_data = {
                    "saisie_texte": text_input,
                    "chemin_fichier": None,
                    "contenu_fichier": None,
                    "type_fichier": None,
                    "source": "saisie_texte"
                }
            else:
                st.error("⚠️ Veuillez saisir une description du sinistre")

    elif input_method == "📄 Upload fichier":
        st.markdown("### 📤 Upload de document")

        uploaded_file = st.file_uploader(
            "Formats supportés: PDF, Images (JPG, PNG), Texte",
            type=['pdf', 'jpg', 'jpeg', 'png', 'txt'],
            help="Glissez-déposez votre document ou cliquez pour sélectionner"
        )

        if uploaded_file is not None:
            st.success(f"✅ Fichier chargé: {uploaded_file.name}")

            # Aperçu du fichier
            if uploaded_file.type.startswith('image'):
                st.image(uploaded_file, caption="Aperçu du document", use_container_width=True)
            elif uploaded_file.type == 'text/plain':
                content = str(uploaded_file.read(), "utf-8")
                st.text_area("Contenu du fichier:", content, height=100)

            if st.button("🚀 Traiter ce document", type="primary", use_container_width=True):
                processing_data = {
                    "saisie_texte": None,
                    "chemin_fichier": None,
                    "contenu_fichier": uploaded_file.read(),
                    "type_fichier": uploaded_file.type,
                    "source": f"upload_{uploaded_file.name}"
                }

    elif input_method == "🎯 Exemple rapide":
        st.markdown("### 🎯 Exemples de Démonstration")

        examples = {
            "🚗 Accident Auto Simple": "Ma voiture a été rayée dans un parking. Rayure de 10cm sur la portière droite. Estimation: 400€. Pas de blessé.",
            "🚨 Urgence Médicale": "Accident grave ce matin avec hospitalisation d'urgence. Collision frontale, ambulance sur place. Véhicule détruit.",
            "🏠 Dégâts Importants": "Incendie dans ma maison hier soir. Dégâts structurels importants. Estimation: 50000€. Expertise requise.",
            "❓ Description Vague": "Il y a eu un problème. Quelque chose est cassé. Je ne sais pas trop quoi dire.",
            "💰 Montant Élevé": "Dégât des eaux massif - canalisation principale. Inondation complète. Dégâts estimés 25000€."
        }

        selected_example = st.selectbox("Choisissez un exemple:", list(examples.keys()))

        st.text_area("Texte de l'exemple:", examples[selected_example], height=100)

        if st.button("🚀 Traiter cet exemple", type="primary", use_container_width=True):
            processing_data = {
                "saisie_texte": examples[selected_example],
                "chemin_fichier": None,
                "contenu_fichier": None,
                "type_fichier": None,
                "source": f"exemple_{selected_example}"
            }

    # Stockage des données pour traitement
    if processing_data:
        st.session_state.processing_data = processing_data
        st.session_state.processing_started = False
        st.success("✅ Sinistre préparé pour traitement")
        st.info("👉 Passez à l'onglet 'Traitement' pour lancer l'analyse")


def processing_interface(workflow, demo_mode):
    """Interface de traitement en temps réel"""
    st.markdown("## ⚙️ Traitement Intelligent")

    if not st.session_state.get('processing_started', False):
        # Affichage des données à traiter
        data = st.session_state.processing_data

        st.markdown("### 📋 Données à traiter")
        with st.expander("Voir les détails", expanded=True):
            st.json({
                "Source": data['source'],
                "Type": "Texte" if data['saisie_texte'] else "Fichier",
                "Contenu": data['saisie_texte'][:200] + "..." if data['saisie_texte'] and len(
                    data['saisie_texte']) > 200 else data['saisie_texte']
            })

        # Lancement du traitement
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 LANCER LE TRAITEMENT", type="primary", use_container_width=True):
                st.session_state.processing_started = True
                st.rerun()

    else:
        # Traitement en cours
        st.markdown("### 🔄 Traitement en cours...")

        # Conteneurs pour les étapes
        step_container = st.container()
        progress_container = st.container()

        # Simulation temps réel avec workflow réel
        with st.spinner("Traitement en cours..."):
            # Barre de progression
            progress_bar = progress_container.progress(0)
            status_text = progress_container.empty()

            # Conteneur des étapes
            steps_display = step_container.container()

            # Étapes du workflow
            workflow_steps = [
                ("🔍 OCR", "Extraction du texte"),
                ("🏷️ Classification", "Analyse du type de sinistre"),
                ("📋 Rapport", "Génération du rapport final")
            ]

            step_placeholders = []
            for i, (icon, desc) in enumerate(workflow_steps):
                placeholder = steps_display.empty()
                step_placeholders.append(placeholder)

            # Exécution réelle du workflow
            start_time = time.time()

            try:
                # Lancement du workflow
                result = workflow.execute(st.session_state.processing_data)

                # Simulation de progression pour UX
                for i, (icon, desc) in enumerate(workflow_steps):
                    progress = (i + 1) / len(workflow_steps)
                    progress_bar.progress(progress)
                    status_text.text(f"Étape {i + 1}/{len(workflow_steps)}: {desc}")

                    # Affichage de l'étape
                    step_class = "step-active" if i == len(workflow_steps) - 1 else "step-completed"
                    step_placeholders[i].markdown(f"""
                    <div class="workflow-step {step_class}">
                        <strong>{icon} {desc}</strong><br>
                        <small>✅ Terminé</small>
                    </div>
                    """, unsafe_allow_html=True)

                    time.sleep(0.5)  # Animation

                # Finalisation
                execution_time = time.time() - start_time
                progress_bar.progress(1.0)
                status_text.text("✅ Traitement terminé avec succès !")

                # Stockage des résultats
                st.session_state.workflow_result = result
                st.session_state.execution_time = execution_time

                # Mise à jour des statistiques
                stats = st.session_state.session_stats
                stats['total_processed'] += 1
                if result.get('success'):
                    stats['successful'] += 1
                stats['avg_time'] = (stats['avg_time'] * (stats['total_processed'] - 1) + execution_time) / stats[
                    'total_processed']

                # Message de succès
                if result.get('success'):
                    st.success(f"🎉 Sinistre traité avec succès en {execution_time:.1f}s !")
                    st.info("👉 Consultez les résultats dans l'onglet 'Résultats'")
                else:
                    st.error("❌ Erreur durant le traitement")
                    st.error(f"Erreurs: {result.get('errors', [])}")

                # Bouton pour nouveau traitement
                if st.button("🔄 Nouveau sinistre", use_container_width=True):
                    st.session_state.processing_started = False
                    st.session_state.processing_data = None
                    st.rerun()

            except Exception as e:
                st.error(f"❌ Erreur système: {str(e)}")
                st.session_state.processing_started = False


def results_interface():
    """Interface des résultats"""
    st.markdown("## 📊 Résultats de l'Analyse")

    result = st.session_state.workflow_result
    exec_time = st.session_state.get('execution_time', 0)

    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Statut",
            "✅ Succès" if result.get('success') else "❌ Échec",
            f"{exec_time:.1f}s"
        )

    with col2:
        st.metric(
            "Étape finale",
            result.get('final_step', 'Inconnu').title(),
            f"{len(result.get('workflow_path', []))} étapes"
        )

    with col3:
        # Confiance globale (estimation)
        confidence = 0.8 if result.get('success') else 0.2
        st.metric("Confiance globale", f"{confidence:.1%}")

    with col4:
        error_count = len(result.get('errors', []))
        st.metric("Erreurs", error_count, "🟢" if error_count == 0 else "🔴")

    # Onglets de résultats détaillés
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Classification", "📄 Rapport", "⚙️ Workflow", "💾 Export"])

    with tab1:
        show_classification_results(result)

    with tab2:
        show_report_results(result)

    with tab3:
        show_workflow_details(result)

    with tab4:
        show_export_options(result)


def show_classification_results(result):
    """Affichage des résultats de classification"""
    st.markdown("### 🏷️ Résultats de Classification")

    # Extraction des données de classification
    results_data = result.get('results', {})
    classification = results_data.get('classification', {})

    if classification:
        class_data = classification.get('resultat_classification', {})

        if class_data:
            # Informations principales
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 📊 Classification")

                # CORRECTION: Conversion sécurisée des enums
                type_sinistre_raw = class_data.get('type_sinistre', 'Inconnu')
                type_sinistre = str(
                    type_sinistre_raw.value if hasattr(type_sinistre_raw, 'value') else type_sinistre_raw)

                severite_raw = class_data.get('severite', 'Inconnue')
                severite = str(severite_raw.value if hasattr(severite_raw, 'value') else severite_raw)

                st.markdown(f"""
                <div class="result-card">
                    <h4>🏷️ Type de sinistre</h4>
                    <h3 style="color: #00008f;">{type_sinistre.replace('_', ' ').title()}</h3>

                    <h4>⚠️ Sévérité</h4>
                    <h3 style="color: #ff6b35;">{severite.upper()}</h3>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown("#### 📈 Métriques")

                # Confiance
                confidence = class_data.get('score_confiance', 0)
                st.markdown("**Confiance de classification:**")
                st.markdown(f"""
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {confidence * 100}%"></div>
                </div>
                <center>{confidence:.1%}</center>
                """, unsafe_allow_html=True)

                # Autres métriques
                montant = class_data.get('montant_estime', 0)
                urgence = class_data.get('score_urgence', 0)

                if montant:
                    st.metric("💰 Montant estimé", f"{montant:,.0f}€")
                st.metric("🚨 Score d'urgence", f"{urgence}/10")

            # Mots-clés identifiés
            mots_cles = class_data.get('mots_cles_trouves', [])
            if mots_cles:
                st.markdown("#### 🔍 Mots-clés identifiés")
                st.write(" • ".join(mots_cles))

        else:
            st.warning("⚠️ Données de classification non disponibles")
    else:
        st.info("ℹ️ Classification non effectuée ou en cours")


def show_report_results(result):
    """Affichage du rapport final"""
    st.markdown("### 📄 Rapport Généré")

    results_data = result.get('results', {})

    # Tentative de récupération du rapport depuis différentes sources
    report_data = None

    # Source 1: Report classique
    if 'report' in results_data:
        report_data = results_data['report'].get('rapport_data')

    # Source 2: Fast track
    elif 'fast_report' in results_data:
        report_data = results_data['fast_report']
        st.info("🚀 Rapport en traitement accéléré")

    # Source 3: Detailed analysis
    elif 'detailed_report' in results_data:
        report_data = results_data['detailed_report']
        st.info("🔬 Rapport d'analyse détaillée")

    # Source 4: Human review
    elif 'human_review' in results_data:
        report_data = results_data['human_review']
        st.warning("👤 En attente de révision humaine")

    if report_data:
        # Affichage du rapport
        st.markdown("#### 📋 Résumé du rapport")

        # Informations principales
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**ID Rapport:**")
            st.code(report_data.get('id_rapport', 'N/A'))

            # CORRECTION: Conversion sécurisée pour type_sinistre
            if 'type_sinistre' in report_data:
                type_raw = report_data.get('type_sinistre', 'N/A')
                type_str = str(type_raw.value if hasattr(type_raw, 'value') else type_raw)
                st.markdown("**Type de sinistre:**")
                st.write(type_str.replace('_', ' ').title())
            else:
                st.markdown("**Type de sinistre:**")
                st.write("N/A")

        with col2:
            st.markdown("**Statut:**")
            st.write(report_data.get('statut', report_data.get('traitement', 'En cours')))

            if 'confiance' in report_data:
                st.markdown("**Confiance:**")
                st.write(f"{report_data['confiance']:.1%}")

        # Actions recommandées
        actions = report_data.get('actions', report_data.get('actions_immediates', []))
        if actions:
            st.markdown("#### 📝 Actions recommandées")
            for i, action in enumerate(actions, 1):
                st.write(f"{i}. {action}")

        # Détails techniques
        with st.expander("🔧 Détails techniques"):
            st.json(report_data)

    else:
        st.info("ℹ️ Rapport en cours de génération ou non disponible")


def show_workflow_details(result):
    """Affichage des détails du workflow"""
    st.markdown("### ⚙️ Détails du Workflow")

    # Chemin parcouru
    workflow_path = result.get('workflow_path', [])
    if workflow_path:
        st.markdown("#### 🛣️ Chemin parcouru")

        path_display = " → ".join([step.replace('_', ' ').title() for step in workflow_path])
        st.markdown(f"**{path_display}**")

        # Visualisation des étapes
        for i, step in enumerate(workflow_path):
            icon = "🔍" if "ocr" in step else "🏷️" if "classification" in step else "📋"
            st.markdown(f"{i + 1}. {icon} {step.replace('_', ' ').title()}")

    # Erreurs
    errors = result.get('errors', [])
    if errors:
        st.markdown("#### ❌ Erreurs rencontrées")
        for error in errors:
            st.error(error)
    else:
        st.success("✅ Aucune erreur")

    # Métadonnées complètes
    with st.expander("🔍 Données complètes du workflow"):
        st.json(result)


def safe_json_serialization(obj):
    """Convertit récursivement les enums en strings pour JSON"""
    if hasattr(obj, 'value'):  # Enum
        return obj.value
    elif isinstance(obj, dict):
        return {k: safe_json_serialization(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_json_serialization(item) for item in obj]
    else:
        return obj


def show_export_options(result):
    """Options d'export"""
    st.markdown("### 💾 Export des Résultats")

    # JSON Export
    st.markdown("#### 📄 Export JSON")

    json_data = {
        "session_id": result.get('session_id'),
        "timestamp": datetime.now().isoformat(),
        "success": result.get('success'),
        "execution_time": st.session_state.get('execution_time'),
        "final_step": result.get('final_step'),
        "workflow_path": result.get('workflow_path'),
        "results": result.get('results', {})
    }

    # CORRECTION: Nettoyage des enums avant sérialisation
    clean_json_data = safe_json_serialization(json_data)

    try:
        json_str = json.dumps(clean_json_data, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Erreur sérialisation: {e}")
        # Fallback: données simplifiées
        json_str = json.dumps({
            "session_id": result.get('session_id', 'unknown'),
            "success": result.get('success', False),
            "final_step": str(result.get('final_step', 'unknown')),
            "error": "Données complexes - voir aperçu simplifié"
        }, ensure_ascii=False, indent=2)

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="📥 Télécharger JSON",
            data=json_str,
            file_name=f"rapport_sinistre_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

    with col2:
        st.download_button(
            label="📋 Copier JSON",
            data=json_str,
            file_name="rapport.json",
            mime="text/plain",
            use_container_width=True
        )

    # Aperçu
    with st.expander("👀 Aperçu JSON"):
        try:
            st.json(clean_json_data)
        except:
            # Fallback si problème d'affichage
            st.code(json_str, language="json")


def history_interface():
    """Interface historique"""
    st.markdown("## 📋 Historique des Traitements")

    # Statistiques globales
    stats = st.session_state.session_stats

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total traité", stats['total_processed'])
    with col2:
        st.metric("Taux de succès", f"{(stats['successful'] / max(stats['total_processed'], 1) * 100):.0f}%")
    with col3:
        st.metric("Temps moyen", f"{stats['avg_time']:.1f}s")

    # Simuler un historique simple
    if stats['total_processed'] > 0:
        st.markdown("### 📊 Derniers traitements")

        # Données simulées pour la démo
        history_data = []
        for i in range(min(5, stats['total_processed'])):
            history_data.append({
                "ID": f"AXA-{datetime.now().strftime('%Y%m%d')}-{i + 1:03d}",
                "Type": ["Accident Auto", "Dégâts Habitation", "Vol", "Incendie"][i % 4],
                "Statut": "✅ Succès" if i < stats['successful'] else "❌ Échec",
                "Temps": f"{stats['avg_time']:.1f}s"
            })

        st.table(history_data)
    else:
        st.info("📝 Aucun traitement effectué dans cette session")

    # Bouton de réinitialisation
    if st.button("🔄 Réinitialiser l'historique"):
        st.session_state.session_stats = {
            'total_processed': 0,
            'successful': 0,
            'avg_time': 0
        }
        st.rerun()


if __name__ == "__main__":
    main()