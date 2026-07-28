"""
streamlit_app.py
Application unique du projet, avec deux modes accessibles depuis la barre
latérale :

- 🧠 Générer un quiz : la démo finale (upload PDF -> quiz interactif corrigé).
- 🏷️ Annoter des passages : outil de travail pour construire le jeu de
  données annoté (data/annotations.jsonl), utilisé ensuite par
  `scripts/train_classifier.py` pour entraîner le classifieur du bloc 2.

Les deux modes partagent la même app mais des états de session distincts
(préfixés `quiz_` et `annotate_`) pour éviter toute interférence entre eux.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Permet d'importer le package `src` depuis app/, quel que soit le
# répertoire courant depuis lequel `streamlit run` est lancé.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification import QuizableClassifier, select_quizable_passages  # noqa: E402
from src.dataset import (  # noqa: E402
    Annotation,
    already_annotated_ids,
    append_annotation,
    corpus_statistics,
    load_annotations,
)
from src.evaluation import evaluate_quiz_quality  # noqa: E402
from src.extraction import extract_passages  # noqa: E402
from src.generation import Quiz, generate_quiz  # noqa: E402
from src.utils import CONFIG  # noqa: E402

st.set_page_config(page_title="Générateur de quiz PDF", page_icon="📝", layout="centered")


# --------------------------------------------------------------------------- #
# État de session (préfixé par mode pour éviter toute interférence)
# --------------------------------------------------------------------------- #

def init_state() -> None:
    defaults = {
        # Mode "Générer un quiz"
        "quiz_result": None,        # objet Quiz généré
        "quiz_answers": {},         # {index_question: index_option_choisi}
        "quiz_validated": False,
        "quiz_quality_report": None,
        # Mode "Annoter"
        "annotate_passages": [],           # passages restants à annoter pour le doc en cours
        "annotate_doc_name": None,
        "annotate_n_labeled_session": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_quiz() -> None:
    st.session_state.quiz_result = None
    st.session_state.quiz_answers = {}
    st.session_state.quiz_validated = False
    st.session_state.quiz_quality_report = None


init_state()


# --------------------------------------------------------------------------- #
# En-tête + vérification de la clé API (commune aux deux modes)
# --------------------------------------------------------------------------- #

st.title("📝 Générateur automatique de quiz")

if not CONFIG.openai_api_key:
    st.error(
        "⚠️ Clé API OpenAI manquante. Crée un fichier `.env` à la racine du projet "
        "contenant :\n\n`OPENAI_API_KEY=sk-...`\n\npuis relance l'app.",
        icon="🚫",
    )
    st.stop()


# --------------------------------------------------------------------------- #
# Barre latérale : sélecteur de mode + paramètres propres à chaque mode
# --------------------------------------------------------------------------- #

with st.sidebar:
    mode = st.radio(
        "Mode",
        ["🧠 Générer un quiz", "🏷️ Annoter des passages"],
        help="« Générer un quiz » est la démo finale. « Annoter » sert à construire "
             "le jeu de données pour entraîner le classifieur (bloc 2 du pipeline).",
    )
    st.divider()

    if mode == "🧠 Générer un quiz":
        st.header("Paramètres")
        n_questions = st.slider("Nombre de questions", min_value=3, max_value=15, value=5)
        threshold = st.slider(
            "Seuil de sélection des passages",
            min_value=0.0, max_value=1.0, value=0.5, step=0.05,
            help="Plus le seuil est élevé, plus les passages retenus sont sélectifs "
                 "(moins nombreux mais plus riches).",
        )
        st.divider()
        st.caption(
            "Classification : "
            + ("modèle entraîné chargé ✅" if Path(CONFIG.classifier_path).exists()
               else "heuristique (pas de modèle entraîné) ℹ️")
        )
    else:
        st.header("📊 Corpus annoté")
        stats = corpus_statistics(load_annotations())
        st.metric("Passages annotés", stats["n_passages"])
        col1, col2 = st.columns(2)
        col1.metric("Quizables", stats["n_quizable"])
        col2.metric("Non quizables", stats["n_non_quizable"])
        st.caption(f"Documents distincts : {stats['n_documents']}")
        st.caption(f"Longueur moyenne : {stats['avg_passage_length']:.0f} caractères")

        if stats["n_passages"] < 40:
            st.warning(f"Objectif recommandé : ≥ 80 passages (actuellement {stats['n_passages']}).")
        else:
            st.success(f"{stats['n_passages']} passages : de quoi entraîner un premier classifieur.")

        st.divider()
        annotations_path = PROJECT_ROOT / "data" / "annotations.jsonl"
        if annotations_path.exists():
            st.download_button(
                "⬇️ Télécharger annotations.jsonl",
                data=annotations_path.read_bytes(),
                file_name="annotations.jsonl",
                mime="application/jsonl",
            )


# =============================================================================
# MODE 1 : Générer un quiz (démo finale)
# =============================================================================

def render_quiz_mode() -> None:
    st.caption("Dépose un PDF de cours ou d'article : l'app extrait le contenu, "
               "sélectionne les passages pertinents et génère un QCM.")

    uploaded_file = st.file_uploader("Document PDF", type=["pdf"], on_change=reset_quiz, key="quiz_uploader")

    if uploaded_file is not None and st.session_state.quiz_result is None:
        if st.button("🚀 Générer le quiz", type="primary"):
            tmp_path = PROJECT_ROOT / "app" / "_tmp_upload.pdf"
            tmp_path.write_bytes(uploaded_file.getvalue())

            try:
                with st.spinner("Extraction du texte du PDF..."):
                    passages = extract_passages(str(tmp_path))
                st.success(f"{len(passages)} passage(s) extrait(s) du document.")

                with st.spinner("Sélection des passages pertinents..."):
                    classifier = QuizableClassifier()
                    if Path(CONFIG.classifier_path).exists():
                        classifier.load()
                    results = select_quizable_passages(
                        passages, classifier=classifier, threshold=threshold, top_k=n_questions * 3
                    )
                if not results:
                    st.warning(
                        "Aucun passage jugé assez informatif pour générer une question. "
                        "Essaie de baisser le seuil dans la barre latérale."
                    )
                    st.stop()
                st.success(f"{len(results)} passage(s) retenu(s) comme pertinents.")

                progress_bar = st.progress(0.0, text="Génération des questions...")

                def _on_progress(current: int, total: int) -> None:
                    progress_bar.progress(current / total, text=f"Génération des questions... ({current}/{total})")

                quiz = generate_quiz(results, n_questions=n_questions, progress_callback=_on_progress)
                progress_bar.empty()

                if not quiz.questions:
                    st.error("Le modèle n'a pas réussi à générer de question exploitable sur ce document.")
                    st.stop()

                st.session_state.quiz_result = quiz
                st.session_state.quiz_quality_report = evaluate_quiz_quality(quiz)

            except ValueError as exc:
                st.error(str(exc))
            finally:
                tmp_path.unlink(missing_ok=True)

            st.rerun()

    if st.session_state.quiz_result is not None:
        _render_quiz(st.session_state.quiz_result)


def _render_quiz(quiz: Quiz) -> None:
    st.divider()
    st.subheader(f"Quiz généré — {len(quiz.questions)} question(s)")

    for i, q in enumerate(quiz.questions):
        st.markdown(f"**{i + 1}. {q.question}**")
        choice = st.radio(
            label=f"question_{i}",
            options=list(range(len(q.options))),
            format_func=lambda idx, opts=q.options: opts[idx],
            key=f"radio_{i}",
            label_visibility="collapsed",
            disabled=st.session_state.quiz_validated,
        )
        st.session_state.quiz_answers[i] = choice

        if st.session_state.quiz_validated:
            if choice == q.correct_index:
                st.success(f"✅ Correct — {q.explanation}")
            else:
                st.error(f"❌ Réponse attendue : « {q.options[q.correct_index]} » — {q.explanation}")
            st.caption(f"Source : page {q.source_page} du document")
        st.write("")

    col1, col2 = st.columns(2)
    with col1:
        if not st.session_state.quiz_validated:
            if st.button("✔️ Valider mes réponses", type="primary"):
                st.session_state.quiz_validated = True
                st.rerun()
        else:
            score = sum(
                1 for i, q in enumerate(quiz.questions) if st.session_state.quiz_answers.get(i) == q.correct_index
            )
            st.metric("Score", f"{score} / {len(quiz.questions)}")

    with col2:
        if st.button("🔄 Nouveau quiz"):
            reset_quiz()
            st.rerun()

    with st.expander("📊 Rapport qualité (analyse des résultats)"):
        st.text(st.session_state.quiz_quality_report.report())


# =============================================================================
# MODE 2 : Annoter des passages (construction du jeu de données)
# =============================================================================

def render_annotate_mode() -> None:
    st.caption(
        "Pour chaque passage, indique s'il contient assez d'information factuelle "
        "(fait, définition, date, chiffre) pour qu'une bonne question de quiz en soit extraite. "
        "Objectif recommandé : 80-150 passages, en variant les documents sources."
    )

    uploaded_file = st.file_uploader("Document PDF à annoter", type=["pdf"], key="annotate_uploader")

    if uploaded_file is not None and uploaded_file.name != st.session_state.annotate_doc_name:
        tmp_path = PROJECT_ROOT / "app" / "_tmp_annotate.pdf"
        tmp_path.write_bytes(uploaded_file.getvalue())

        try:
            all_passages = extract_passages(str(tmp_path))
        except ValueError as exc:
            st.error(str(exc))
            all_passages = []
        finally:
            tmp_path.unlink(missing_ok=True)

        already_done = already_annotated_ids()
        remaining = [p for p in all_passages if p.id not in already_done]

        st.session_state.annotate_passages = remaining
        st.session_state.annotate_doc_name = uploaded_file.name
        st.session_state.annotate_n_labeled_session = 0

        skipped = len(all_passages) - len(remaining)
        if skipped:
            st.info(f"{skipped} passage(s) de ce document déjà annoté(s) précédemment, ignoré(s).")

    def _label_current(label: int) -> None:
        passage = st.session_state.annotate_passages.pop(0)
        append_annotation(
            Annotation(
                passage_id=passage.id,
                text=passage.text,
                source_doc=st.session_state.annotate_doc_name,
                page=passage.page,
                label=label,
                annotated_by="manuel",
            )
        )
        st.session_state.annotate_n_labeled_session += 1

    if st.session_state.annotate_passages:
        passage = st.session_state.annotate_passages[0]
        remaining_count = len(st.session_state.annotate_passages)

        st.progress(
            st.session_state.annotate_n_labeled_session
            / max(1, st.session_state.annotate_n_labeled_session + remaining_count),
            text=f"{remaining_count} passage(s) restant(s) dans ce document",
        )

        st.markdown(f"**Page {passage.page}**")
        st.info(passage.text)

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("✅ Quizable", type="primary", use_container_width=True):
                _label_current(1)
                st.rerun()
        with col2:
            if st.button("❌ Non quizable", use_container_width=True):
                _label_current(0)
                st.rerun()
        with col3:
            if st.button("⏭️ Passer (ne pas annoter)", use_container_width=True):
                st.session_state.annotate_passages.pop(0)
                st.rerun()

    elif st.session_state.annotate_doc_name is not None:
        st.success(
            f"Document « {st.session_state.annotate_doc_name} » entièrement annoté "
            f"({st.session_state.annotate_n_labeled_session} passage(s) labellisé(s) dans cette session). "
            "Dépose un autre PDF pour continuer."
        )
    else:
        st.info("Dépose un PDF ci-dessus pour commencer à annoter.")


# --------------------------------------------------------------------------- #
# Aiguillage vers le mode sélectionné
# --------------------------------------------------------------------------- #

if mode == "🧠 Générer un quiz":
    render_quiz_mode()
else:
    render_annotate_mode()