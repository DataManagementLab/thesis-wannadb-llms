import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from util import consider_overlap_as_match
from wannadb.data.data import InformationNugget
from wannadb.data.signals import CachedDistanceSignal
from wannadb.interaction import BaseInteractionCallback

# which row of the ranked list gets the next answer; the list is sorted by falling distance
PICK_RULES = ("random", "nearest", "surest", "least_sure", "rotate", "disagree")
ROTATION = ("surest", "nearest", "least_sure")
RULE_VARIANTS = tuple(f"rule_{rule}" for rule in PICK_RULES)
# which documents the LLM reads, chosen from the texts alone: at random, the middle of N clusters, or far from each other
READ_ORDERS = ("random", "typical", "diverse")

Spans = List[List[int]]


def document_key(name: str) -> str:
    return Path(name.replace("\\", "/")).stem


def stored_setting(variant: str) -> Optional[Tuple[str, Optional[int]]]:
    """(pick rule, documents the LLM may read) of a grid variant, None if it asks a real LLM."""
    if variant.startswith("rule_"):
        return variant[len("rule_"):], None
    if variant.startswith("reads_"):  # the rows of the read documents are all answered, their order does not cost
        return "random", int(variant[len("reads_"):])
    return None


def occurrences(text: str, value: str) -> Spans:
    spans, start = [], text.find(value)
    while start != -1:
        spans.append([start, start + len(value)])
        start = text.find(value, start + 1)
    return spans


def load_llm_answers(path: str, seed_index: int, ground_truth: List[Dict[str, Any]]) -> Dict[str, Dict[str, Spans]]:
    """The answers of one repeat of llm_only_baseline.py; seeds beyond the repeats use them again. Every place of the
    copied text counts, because the baseline only located the first one."""
    text_of = {d["id"]: d["text"] for d in ground_truth}
    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    repeats = sorted({record["repeat"] for record in records})
    repeat = repeats[seed_index % len(repeats)]
    answers: Dict[str, Dict[str, Spans]] = {}
    for record in records:
        if record["repeat"] != repeat:
            continue
        text = text_of[record["document"]]
        answers[record["document"]] = {attribute: occurrences(text, text[span[0]:span[1]]) if span else []
                                       for attribute, span in record["cells"].items()}
    return answers


def gold_answers(ground_truth: List[Dict[str, Any]]) -> Dict[str, Dict[str, Spans]]:
    return {d["id"]: {attribute: [[m["start_char"], m["end_char"]] for m in mentions]
                      for attribute, mentions in d["mentions"].items()} for d in ground_truth}


def read_documents(ground_truth: List[Dict[str, Any]], count: int, seed: int, order: str = "random") -> Set[str]:
    documents = sorted(ground_truth, key=lambda d: d["id"])
    names = [d["id"] for d in documents]
    if order == "random":  # the same shuffle for every count, so a smaller read set is part of every larger one
        random.Random(seed).shuffle(names)
        return set(names[:count])
    if count >= len(names):
        return set(names)
    vectors = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(d["text"] for d in documents)
    if order == "typical":
        clusters = KMeans(n_clusters=count, random_state=seed % 2 ** 31, n_init=10).fit(vectors)
        return {names[int(i)] for i in clusters.transform(vectors).argmin(axis=0)}
    if order == "diverse":  # farthest first, starting from a random document
        similarity = (vectors @ vectors.T).toarray()
        chosen = [random.Random(seed).randrange(len(names))]
        while len(chosen) < count:
            chosen.append(int(np.argmin(similarity[:, chosen].max(axis=1))))
        return {names[i] for i in chosen}
    raise ValueError(f"unknown read order {order!r}")


class StoredAnswerFeedback(BaseInteractionCallback):
    """Answers from a table instead of an LLM: no request is sent, the LLM's noise is gone and every pick rule gets
    the same answers. An answer follows the simulated user: the guess if it fits, else another fitting nugget of the
    document, else a custom match, or no value if the table has none. With read_documents only these documents are
    answered, and the column ends when none of them is left."""

    def __init__(self, answers: Dict[str, Dict[str, Spans]], pick_rule: str = "random",
                 read_documents: Optional[Set[str]] = None) -> None:
        if pick_rule not in PICK_RULES:
            raise ValueError(f"unknown pick rule {pick_rule!r}")
        self._answers = answers
        self._pick_rule = pick_rule
        self._read_documents = read_documents
        self.pick_log: List[Dict[str, Any]] = []

    def _call(self, pipeline_element_identifier: str, data: Dict[str, Any]) -> Dict[str, Any]:
        if "do-attribute-request" in data.keys():
            return {"do-attribute": True}

        attribute_name: str = data["attribute"].name
        rows: List[InformationNugget] = list(data["nuggets"])
        if self._read_documents is not None:
            rows = [n for n in rows if document_key(n.document.name) in self._read_documents]
            if not rows:
                return {"message": "stop-interactive-matching"}

        pick = self._pick(rows, data)
        result = self._answer(attribute_name, pick)
        threshold: float = data["max-distance"]
        self.pick_log.append({
            "attribute": attribute_name, "round": data["num-feedback"], "document": pick.document.name,
            "position": next(i for i, n in enumerate(data["nuggets"]) if n is pick), "rows": len(data["nuggets"]),
            "distance": float(pick[CachedDistanceSignal]), "threshold": float(threshold),
            "cell_filled": bool(pick[CachedDistanceSignal] <= threshold),
            "answer_differs": not self._guess_stands(attribute_name, pick, threshold),
            "message": result["message"],
        })
        return result

    def _pick(self, rows: List[InformationNugget], data: Dict[str, Any]) -> InformationNugget:
        threshold: float = data["max-distance"]
        rule = self._pick_rule
        if rule == "rotate":
            rule = ROTATION[(data["num-feedback"] - 1) % len(ROTATION)]
        if rule == "surest":
            return rows[-1]
        if rule == "least_sure":
            return rows[0]
        if rule == "nearest":
            return min(rows, key=lambda n: abs(n[CachedDistanceSignal] - threshold))
        if rule == "disagree":
            differing = [n for n in rows if not self._guess_stands(data["attribute"].name, n, threshold)]
            rows = differing or rows
        # global random on purpose: the gold oracle draws its document the same way
        return random.choice(rows)

    def _spans(self, attribute_name: str, nugget: InformationNugget) -> Spans:
        return self._answers.get(document_key(nugget.document.name), {}).get(attribute_name, [])

    @staticmethod
    def _fits(nugget: InformationNugget, spans: Spans) -> bool:
        return any(consider_overlap_as_match(start, end, nugget.start_char, nugget.end_char) for start, end in spans)

    def _guess_stands(self, attribute_name: str, nugget: InformationNugget, threshold: float) -> bool:
        """Whether the cell WannaDB would write for this row agrees with the table."""
        spans = self._spans(attribute_name, nugget)
        filled = nugget[CachedDistanceSignal] <= threshold
        return self._fits(nugget, spans) if spans and filled else not spans and not filled

    def _answer(self, attribute_name: str, pick: InformationNugget) -> Dict[str, Any]:
        spans = self._spans(attribute_name, pick)
        if not spans:
            return {"message": "no-match-in-document", "nugget": pick, "not-a-match": pick}
        if self._fits(pick, spans):
            return {"message": "is-match", "nugget": pick, "not-a-match": None}
        for nugget in pick.document.nuggets:
            if self._fits(nugget, spans):
                return {"message": "is-match", "nugget": nugget, "not-a-match": pick}
        return {"message": "custom-match", "document": pick.document, "start": spans[0][0], "end": spans[0][1]}
