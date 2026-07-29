"""
human_eval.py
Évaluation humaine de la qualité des questions générées (bloc 3/4).

C'est la méthode d'évaluation qualité RECOMMANDÉE pour ce projet (cf.
evaluation.py) tant qu'aucun jeu de questions de référence n'existe : un
humain note chaque question générée sur 3 critères indépendants, ce qui
donne une mesure directe de qualité pédagogique -- contrairement aux
contrôles structurels (QuizQualityReport) qui ne détectent que les erreurs
de format, ou à BLEU/ROUGE qui mesurent une similarité de surface avec une
référence.

Format retenu (même logique que dataset.py) : un fichier JSONL
(`data/human_eval.jsonl`), une évaluation par ligne :
{
    "question_text": "...",
    "source_passage_id": "p0007",
    "source_doc": "cours_deep_learning.pdf",
    "pertinence": 4,               # 1-5 : porte sur une info importante du passage ?
    "clarte": 5,                   # 1-5 : formulation claire, sans ambiguïté ?
    "distracteurs_plausibles": 3,  # 1-5 : les mauvaises réponses sont-elles plausibles ?
    "commentaire": "Distracteur B trop évident"
}
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

DEFAULT_HUMAN_EVAL_PATH = "data/human_eval.jsonl"

CRITERIA = ("pertinence", "clarte", "distracteurs_plausibles")

CRITERIA_LABELS = {
    "pertinence": "Pertinence (porte sur une info importante du passage)",
    "clarte": "Clarté (formulation sans ambiguïté)",
    "distracteurs_plausibles": "Distracteurs plausibles (ni évidents, ni absurdes)",
}


@dataclass
class HumanRating:
    question_text: str
    source_passage_id: str
    source_doc: str
    pertinence: int              # échelle 1 (mauvais) à 5 (excellent)
    clarte: int
    distracteurs_plausibles: int
    commentaire: str = ""


def append_rating(rating: HumanRating, path: str = DEFAULT_HUMAN_EVAL_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(rating), ensure_ascii=False) + "\n")


def load_ratings(path: str = DEFAULT_HUMAN_EVAL_PATH) -> list[HumanRating]:
    file_path = Path(path)
    if not file_path.exists():
        return []

    ratings = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ratings.append(HumanRating(**json.loads(line)))
    return ratings


@dataclass
class HumanEvalReport:
    n_ratings: int
    avg_pertinence: float
    avg_clarte: float
    avg_distracteurs_plausibles: float
    pct_faible_qualite: float  # % de questions avec au moins un critère noté <= 2

    def report(self) -> str:
        if self.n_ratings == 0:
            return "Aucune évaluation humaine enregistrée pour le moment."
        return (
            f"Questions évaluées          : {self.n_ratings}\n"
            f"Pertinence moyenne           : {self.avg_pertinence:.2f} / 5\n"
            f"Clarté moyenne               : {self.avg_clarte:.2f} / 5\n"
            f"Distracteurs plausibles (moy): {self.avg_distracteurs_plausibles:.2f} / 5\n"
            f"Questions de faible qualité  : {self.pct_faible_qualite:.0%} "
            f"(au moins un critère noté ≤ 2)\n"
        )


def aggregate_ratings(ratings: list[HumanRating]) -> HumanEvalReport:
    if not ratings:
        return HumanEvalReport(0, 0.0, 0.0, 0.0, 0.0)

    n_faible = sum(
        1 for r in ratings
        if min(r.pertinence, r.clarte, r.distracteurs_plausibles) <= 2
    )

    return HumanEvalReport(
        n_ratings=len(ratings),
        avg_pertinence=float(np.mean([r.pertinence for r in ratings])),
        avg_clarte=float(np.mean([r.clarte for r in ratings])),
        avg_distracteurs_plausibles=float(np.mean([r.distracteurs_plausibles for r in ratings])),
        pct_faible_qualite=n_faible / len(ratings),
    )