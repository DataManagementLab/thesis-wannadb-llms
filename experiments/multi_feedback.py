import json
import time
from typing import Any, Dict, List, Optional

from automatic_feedback import AutomaticCustomMatchesRandomRankingBasedMatchingFeedback
from llm_humanlike_feedback import HumanLikeLLMCallback
from util import consider_overlap_as_match, get_document_by_name
from wannadb.data.data import InformationNugget

# what each row of the list question shows around the guess
MULTI_PROMPT_VERSIONS = {"sentence": "multi-v1", "paragraph": "multi-v2-paragraph"}
# a paragraph longer than this is cut around the guess
MAX_PARAGRAPH_CHARS = 2000


class GoldMultiFeedback(AutomaticCustomMatchesRandomRankingBasedMatchingFeedback):
    """Simulated user with multi-feedback: confirms every row of the ranked list whose value is right (one
    multi-match). When no row is right, it answers like the single-feedback oracle: another nugget of a random row's
    document, a custom match or no value."""

    def _call(self, pipeline_element_identifier: str, data: Dict[str, Any]) -> Dict[str, Any]:
        if "do-attribute-request" in data.keys():
            return {"do-attribute": True}
        attribute_name = self._user_attribute_name2dataset_attribute_name[data["attribute"].name]
        right = [n for n in data["nuggets"] if self._is_right(n, attribute_name)]
        if right:
            return {"message": "multi-match", "nuggets": right}
        return super()._call(pipeline_element_identifier, data)

    def _is_right(self, nugget: InformationNugget, attribute_name: str) -> bool:
        document = get_document_by_name(self._documents, nugget.document.name)
        mentions = document["mentions"].get(attribute_name, []) if document else []
        return any(consider_overlap_as_match(m["start_char"], m["end_char"], nugget.start_char, nugget.end_char)
                   for m in mentions)


class MultiFeedbackLLMCallback(HumanLikeLLMCallback):
    """The LLM sees the whole ranked list, every guess marked in its sentence as in the GUI, and names all rows whose
    value is right (one multi-match). When it names none, the round goes on as a normal human-like round on one random
    row, so the LLM can still pick another nugget or copy a value from the text. Each row shows the sentence of the
    guess, or with multi_context="paragraph" its whole paragraph, so the LLM can tell which role a value plays."""

    def __init__(self, *args, multi_context: str = "sentence", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if multi_context not in MULTI_PROMPT_VERSIONS:
            raise ValueError(f"unknown multi_context {multi_context!r}")
        self._multi_context = multi_context
        # the list question of a round that then fell back to a single answer, logged with that answer
        self._pending_multi: Optional[Dict[str, Any]] = None

    def _give_feedback(self, data: Dict[str, Any]) -> Dict[str, Any]:
        nuggets: List[InformationNugget] = list(data["nuggets"])
        attribute_name: str = data["attribute"].name
        t0 = time.time()

        prompt = self._build_multi_prompt(attribute_name, nuggets)
        answer, step = self._ask_json(prompt)
        if answer is None:
            answer, step["retry"] = self._ask_json(prompt + "\nKeep the reason very short.")
        rows = self._row_numbers(answer, len(nuggets))
        step["rows"] = rows
        step["gold_rows"] = self._gold_rows(attribute_name, nuggets)

        if not rows:
            self._pending_multi = step
            return super()._give_feedback(data)

        confirmed = [nuggets[i - 1] for i in rows]
        for nugget in confirmed:
            self._accepted_values.setdefault(attribute_name, []).append(nugget.text.strip())
        self._last_accepted = None
        self._log_multi(attribute_name, data, step, time.time() - t0)
        return {"message": "multi-match", "nuggets": confirmed}

    def _build_multi_prompt(self, attribute_name: str, nuggets: List[InformationNugget]) -> str:
        lines = [
            f"An information extraction system fills the table column {self._attribute_label(attribute_name)} with "
            "values taken from documents.",
            f"Below are its current guesses for {len(nuggets)} different documents. Each row shows the "
            f"{self._multi_context} in which the guessed text appears, with the guessed text marked by [[ and ]].",
            "",
            *(f'{i}. "{self._marked_context(n)}"' for i, n in enumerate(nuggets, 1)),
            "",
            *self._example_lines(attribute_name),
            "Judge only from the text, not from outside knowledge.",
            f'Which rows mark the right value of "{attribute_name}" for their document? Small differences at the '
            'boundaries, such as a missing "the" or "Inc.", do not matter. A row is wrong if the marked text names '
            "something else, or if it contains clearly more or clearly less than the value.",
            "List every right row. If no row is right, answer an empty list.",
            "",
            'Respond with exactly one JSON object and nothing else: '
            '{"reason": "<one short sentence>", "right_rows": [<row numbers>]}',
        ]
        return "\n".join(lines)

    def _marked_context(self, nugget: InformationNugget) -> str:
        if self._multi_context == "sentence":
            return self._marked_sentence(nugget)
        text = nugget.document.text
        left = text.rfind("\n", 0, nugget.start_char) + 1
        right = text.find("\n", nugget.end_char)
        right = len(text) if right == -1 else right
        if right - left > MAX_PARAGRAPH_CHARS:
            half = MAX_PARAGRAPH_CHARS // 2
            left, right = max(left, nugget.start_char - half), min(right, nugget.end_char + half)
        marked = (text[left:nugget.start_char] + "[[" + text[nugget.start_char:nugget.end_char] + "]]"
                  + text[nugget.end_char:right])
        return " ".join(marked.split())

    @staticmethod
    def _row_numbers(answer: Optional[Dict[str, Any]], num_rows: int) -> List[int]:
        rows = (answer or {}).get("right_rows")
        if not isinstance(rows, list):
            return []
        valid = []
        for row in rows:
            try:
                row = int(row)
            except (TypeError, ValueError):
                continue
            if 1 <= row <= num_rows and row not in valid:
                valid.append(row)
        return valid

    def _gold_rows(self, attribute_name: str, nuggets: List[InformationNugget]) -> Optional[List[int]]:
        # only logged, never used to decide anything
        rows = []
        for i, nugget in enumerate(nuggets, 1):
            mentions = self._gold_mentions(attribute_name, nugget.document)
            if mentions is None:
                return None
            if any(consider_overlap_as_match(m["start_char"], m["end_char"], nugget.start_char, nugget.end_char)
                   for m in mentions):
                rows.append(i)
        return rows

    def _extra_log_fields(self) -> Dict[str, Any]:
        pending, self._pending_multi = self._pending_multi, None
        if pending is None:
            return {}
        return {"multi": self._multi_summary(pending), "multi_step": pending}

    def _log(self, attribute_name: str, data: Dict[str, Any], record: Dict[str, Any], result: Dict[str, Any],
             duration: float) -> None:
        if self._pending_multi is not None:  # its call and time belong to this round
            record["step_a"].setdefault("extra_calls", []).insert(0, self._pending_multi)
            duration += self._pending_multi["seconds"]
        super()._log(attribute_name, data, record, result, duration)

    @staticmethod
    def _multi_summary(step: Dict[str, Any]) -> Dict[str, Any]:
        chosen, gold = step["rows"], step["gold_rows"]
        summary: Dict[str, Any] = {"chosen": chosen, "right": gold}
        if gold is not None:
            summary.update({"chosen_right": len(set(chosen) & set(gold)), "chosen_wrong": len(set(chosen) - set(gold)),
                            "right_missed": len(set(gold) - set(chosen))})
        return summary

    def _log_multi(self, attribute_name: str, data: Dict[str, Any], step: Dict[str, Any], duration: float) -> None:
        steps = [step] + ([step["retry"]] if step.get("retry") else [])
        usage = None
        for s in steps:
            usage = self._combine_usage(usage, s["usage"])
        summary = self._multi_summary(step)
        out = {
            "timestamp": time.time(),
            "mode": self.mode + "+multi",
            "prompt_version": MULTI_PROMPT_VERSIONS[self._multi_context],
            "include_description": self._include_description,
            "llm_calls": len(steps),
            "num_feedback": data.get("num-feedback"),
            "confirmed_examples_shown": self._examples_shown(attribute_name),
            "attribute": attribute_name,
            "num_candidates": len(data["nuggets"]),
            "pick": None,
            "decision": "multi",
            "step_a": step,
            "step_b": None,
            "raw_response": step["raw_response"],
            "stage2": None,
            "usage": usage,
            "total_tokens_used_so_far": self._total_tokens_used,
            "duration_seconds": duration,
            "result_message": "multi-match",
            "result_span": None,
            "agrees_with_gold": None if step["gold_rows"] is None else summary["chosen_wrong"] == 0,
            "oracle_answer": None,
            "parse_failures_so_far": self._parse_failures,
            "truncated_so_far": self._truncated_responses,
            "signals": {},
            "multi": summary,
        }
        self.last_round = out
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
