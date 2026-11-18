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


PICK_RULES = ("random", "nearest", "surest", "least_sure", "rotate", "disagree", "multi", "multi_rotate")
ROTATION = ("surest", "nearest", "least_sure")
MULTI_ROTATION = ("multi_surest", "nearest", "least_sure")
MULTI_ROWS = 10
RULE_VARIANTS = tuple(f"rule_{rule}" for rule in PICK_RULES)
READ_ORDERS = ("random", "typical", "diverse")

CACHE_MODES = ("plain", "prefer", "mix")
CACHE_RULE = "rotate"
SHORTLIST = 10


def threshold_window(rows: List[InformationNugget], threshold: float, size: int = SHORTLIST) -> List[InformationNugget]:

    ix_lower = 0
    while ix_lower < len(rows) and rows[ix_lower][CachedDistanceSignal] > threshold:
        ix_lower += 1
    half = size // 2
    higher_left, higher_right = max(0, ix_lower - half), ix_lower
    lower_left, lower_right = ix_lower, min(len(rows), ix_lower + half)
    if lower_right - lower_left < half:
        higher_left = max(0, higher_left - (half - (lower_right - lower_left)))
    elif higher_right - higher_left < half:
        lower_right = min(len(rows), lower_right + (half - (higher_right - higher_left)))
    return rows[higher_left:lower_right]

Spans = List[List[int]]


def document_key(name: str) -> str:
    return Path(name.replace("\\", "/")).stem


def stored_setting(variant: str) -> Optional[Tuple[str, Optional[int]]]:
    """(pick rule, documents the LLM may read) of a grid variant, None if it asks a real LLM."""
    if variant.startswith("rule_"):
        return variant[len("rule_"):], None
    if variant.startswith("reads_"):  # the rows of the read documents are all answered, their order does not cost
        return "random", int(variant[len("reads_"):])
    if cache_setting(variant) is not None:
        return CACHE_RULE, None
    return None


def cache_setting(variant: str) -> Optional[Tuple[str, int]]:
    if not variant.startswith("cache_"):
        return None
    parts = variant.split("_")
    if len(parts) != 3 or parts[1] not in CACHE_MODES or not parts[2].startswith("r") or not parts[2][1:].isdigit():
        raise ValueError(f"cache variant must be cache_<{'|'.join(CACHE_MODES)}>_r<rounds>, got {variant!r}")
    return parts[1], int(parts[2][1:])


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
    answered, and the column ends when none of them is left.

    With cache_mode, one object must answer all columns of a table in order (one matcher run with all attributes),
    because the cache of read documents carries over from column to column. A row needs a read of the whole document
    unless the answer confirms the row's own nugget (that only needs the row in its sentence). The answers stay the
    same table either way; the cache only decides which reads are new and, with "prefer", which row is answered."""

    def __init__(self, answers: Dict[str, Dict[str, Spans]], pick_rule: str = "random",
                 read_documents: Optional[Set[str]] = None, multi_min: int = 3,
                 cache_mode: Optional[str] = None, steered_budget: Optional[int] = None) -> None:
        if pick_rule not in PICK_RULES:
            raise ValueError(f"unknown pick rule {pick_rule!r}")
        if cache_mode is not None and cache_mode not in CACHE_MODES:
            raise ValueError(f"unknown cache mode {cache_mode!r}")
        self._answers = answers
        self._pick_rule = pick_rule
        self._read_documents = read_documents
        # fewest right rows for a multi-match
        self._multi_min = multi_min
        self._cache_mode = cache_mode
        if cache_mode == "mix" and steered_budget is None:
            raise ValueError("cache mode mix needs steered_budget (rounds per column that may cost a check call)")
        self._steered_budget = steered_budget
        self._steered_rounds: Dict[str, int] = {}
        # documents the LLM has read completely, with the answers of all columns
        self.cache: Set[str] = set()
        self.pick_log: List[Dict[str, Any]] = []

    def _call(self, pipeline_element_identifier: str, data: Dict[str, Any]) -> Dict[str, Any]:
        if "do-attribute-request" in data.keys():
            return {"do-attribute": True}

        if self._cache_mode == "mix":
            return self._mix_round(data)

        attribute_name: str = data["attribute"].name
        rows: List[InformationNugget] = list(data["nuggets"])
        if self._read_documents is not None:
            rows = [n for n in rows if document_key(n.document.name) in self._read_documents]
            if not rows:
                return {"message": "stop-interactive-matching"}

        rule = self._round_rule(data["num-feedback"])
        threshold: float = data["max-distance"]
        if rule in ("multi", "multi_surest"):
            checked = rows if rule == "multi" else rows[-MULTI_ROWS:]
            right = [n for n in checked if self._fits(n, self._spans(attribute_name, n))]
            if len(right) >= self._multi_min:
                self.pick_log.append({"attribute": attribute_name, "round": data["num-feedback"],
                                      "rows": len(data["nuggets"]), "rows_checked": len(checked),
                                      "rows_confirmed": len(right), "threshold": float(threshold),
                                      "message": "multi-match"})
                return {"message": "multi-match", "nuggets": right}
            rule = "random" if rule == "multi" else "nearest"

        pick = self._pick(rule, rows, data)
        cache_entry: Dict[str, Any] = {}
        if self._cache_mode is not None:
            pick, cache_entry = self._use_cache(rule, rows, pick, data)
        result = self._answer(attribute_name, pick)
        self.pick_log.append({
            "attribute": attribute_name, "round": data["num-feedback"], "document": pick.document.name,
            "position": next(i for i, n in enumerate(data["nuggets"]) if n is pick), "rows": len(data["nuggets"]),
            "distance": float(pick[CachedDistanceSignal]), "threshold": float(threshold),
            "cell_filled": bool(pick[CachedDistanceSignal] <= threshold),
            "answer_differs": not self._guess_stands(attribute_name, pick, threshold),
            "message": result["message"],
            **cache_entry,
        })
        return result

    def _needs_read(self, attribute_name: str, nugget: InformationNugget) -> bool:
        """Whether answering this row needs the whole document: everything except confirming the row's own nugget
        (another nugget, a custom match and "no value in this document" all need the full text)."""
        spans = self._spans(attribute_name, nugget)
        return not (spans and self._fits(nugget, spans))

    def _use_cache(self, rule: str, rows: List[InformationNugget], pick: InformationNugget,
                   data: Dict[str, Any]) -> Tuple[InformationNugget, Dict[str, Any]]:
        attribute_name: str = data["attribute"].name
        threshold: float = data["max-distance"]
        rule_pick = pick
        switched = False
        if (self._cache_mode == "prefer" and self._needs_read(attribute_name, pick)
                and document_key(pick.document.name) not in self.cache):
            # the rows the check marks wrong whose document is already read; the pick rule chooses among them
            cached_wrong = [n for n in rows if document_key(n.document.name) in self.cache
                            and not self._guess_stands(attribute_name, n, threshold)]
            if cached_wrong:
                pick = self._pick(rule, cached_wrong, data)
                switched = True
        needs_read = self._needs_read(attribute_name, pick)
        new_read = needs_read and document_key(pick.document.name) not in self.cache
        if new_read:
            self.cache.add(document_key(pick.document.name))
        return pick, {"needs_read": needs_read, "new_read": new_read, "cache_hit": needs_read and not new_read,
                      "switched": switched, "rule_pick_document": rule_pick.document.name,
                      "cache_size": len(self.cache)}

    def _mix_round(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """One round of cache mode "mix". data["nuggets"] must be the whole ranked list."""
        if data["num-nuggets-above"] != 0 or data["num-nuggets-below"] > 0:
            raise ValueError("cache mode mix needs the whole ranked list (--len-ranked-list >= number of documents)")
        attribute_name: str = data["attribute"].name
        threshold: float = data["max-distance"]
        rows: List[InformationNugget] = list(data["nuggets"])
        cached = [n for n in rows if document_key(n.document.name) in self.cache]
        steered = self._steered_rounds.get(attribute_name, 0)
        if cached:
            # as the read limit: any row of a read document, no LLM call; global random like the read limit
            pick, free = random.choice(cached), True
            needs_read = new_read = False
        else:
            if steered >= self._steered_budget:
                return {"message": "stop-interactive-matching"}
            self._steered_rounds[attribute_name] = steered + 1
            window = threshold_window(rows, threshold)
            pick = self._pick(ROTATION[steered % len(ROTATION)], window, data)
            free = False
            needs_read = new_read = self._needs_read(attribute_name, pick)  # no row of a cached document is left
            if new_read:
                self.cache.add(document_key(pick.document.name))
        result = self._answer(attribute_name, pick)
        self.pick_log.append({
            "attribute": attribute_name, "round": data["num-feedback"], "document": pick.document.name,
            "position": next(i for i, n in enumerate(data["nuggets"]) if n is pick), "rows": len(data["nuggets"]),
            "distance": float(pick[CachedDistanceSignal]), "threshold": float(threshold),
            "cell_filled": bool(pick[CachedDistanceSignal] <= threshold),
            "answer_differs": not self._guess_stands(attribute_name, pick, threshold),
            "message": result["message"], "free": free, "steered_round": None if free else steered + 1,
            "needs_read": needs_read, "new_read": new_read, "cache_hit": free, "switched": False,
            "cache_size": len(self.cache),
        })
        return result

    def _round_rule(self, round_no: int) -> str:
        if self._pick_rule == "rotate":
            return ROTATION[(round_no - 1) % len(ROTATION)]
        if self._pick_rule == "multi_rotate":
            return MULTI_ROTATION[(round_no - 1) % len(MULTI_ROTATION)]
        return self._pick_rule

    def _pick(self, rule: str, rows: List[InformationNugget], data: Dict[str, Any]) -> InformationNugget:
        threshold: float = data["max-distance"]
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
