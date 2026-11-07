
import random
from typing import Any, Dict, List, Optional, Set, Tuple
from wannadb.interaction import BaseInteractionCallback

SCHEDULES = ("handoff", "examples", "llmfirst", "interleave", "sandwich", "random")
ESCALATIONS = ("esc_vote", "esc_review", "esc_oracle")


def human_round_numbers(variant: str, k: int, max_rounds: int, seed: int = 0) -> Set[int]:
    if variant == "llmfirst":
        return set(range(max_rounds - k + 1, max_rounds + 1))
    if variant == "interleave":  # spread evenly, starting with the human
        return {i * max_rounds // k + 1 for i in range(k)} if k else set()
    if variant == "sandwich":  # half at the start, half at the end
        return set(range(1, (k + 1) // 2 + 1)) | set(range(max_rounds - k // 2 + 1, max_rounds + 1))
    if variant == "random":  # own generator, the global one decides which document is asked about
        return set(random.Random(seed).sample(range(1, max_rounds + 1), min(k, max_rounds)))
    return set(range(1, k + 1))  # handoff, examples and the escalations: the human first


def is_unsure(escalation: str, llm_round: Dict[str, Any]) -> bool:
    signals = llm_round.get("signals") or {}
    if escalation == "esc_vote":
        return bool(signals.get("probe_disagrees"))
    if escalation == "esc_review":
        return signals.get("review_verdict") == "revise"
    if escalation == "esc_oracle":  # upper bound, it knows which answers are wrong
        return llm_round.get("agrees_with_gold") is False
    raise ValueError(f"unknown escalation {escalation}")


class HybridInteractionCallback(BaseInteractionCallback):
    """Rounds 1 to human_rounds of every attribute are answered by the human, the rest by the LLM"""

    def __init__(self, human: BaseInteractionCallback, llm: Optional[BaseInteractionCallback],
                 human_rounds: int, human_round_numbers: Optional[Set[int]] = None,
                 escalation: Optional[str] = None) -> None:
        self._human = human
        self._llm = llm
        self._human_rounds = human_rounds
        self._human_round_numbers = human_round_numbers
        self._escalation = escalation
        # (attribute, round, answerer, message, document)
        self.answers: List[Tuple[str, int, str, str, Optional[str]]] = []

    def _is_human_round(self, round_no: int) -> bool:
        if self._human_round_numbers is not None:
            return round_no in self._human_round_numbers
        return round_no <= self._human_rounds

    def _call(self, pipeline_element_identifier: str, data: Dict[str, Any]) -> Dict[str, Any]:
        if "do-attribute-request" in data.keys():
            return {"do-attribute": True}

        attribute_name: str = data["attribute"].name
        round_no: int = data["num-feedback"]

        if self._is_human_round(round_no) or self._llm is None:
            return self._ask_human(pipeline_element_identifier, data, "human")

        state = random.getstate()
        result = self._llm(pipeline_element_identifier, data)
        if self._escalation and is_unsure(self._escalation, self._llm.last_round):
            random.setstate(state)
            self._llm.withdraw_last_answer(attribute_name)
            return self._ask_human(pipeline_element_identifier, data, "human_escalated")
        self.answers.append((attribute_name, round_no, "llm", result["message"], self._document_of(result)))
        return result

    def _ask_human(self, pipeline_element_identifier: str, data: Dict[str, Any], answerer: str) -> Dict[str, Any]:
        result = self._human(pipeline_element_identifier, data)
        self.answers.append((data["attribute"].name, data["num-feedback"], answerer, result["message"],
                             self._document_of(result)))
        self._remember_confirmed_value(data["attribute"].name, result)
        return result

    @staticmethod
    def _document_of(result: Dict[str, Any]) -> Optional[str]:
        if result["message"] == "multi-match":
            return ",".join(n.document.name for n in result["nuggets"])
        source = result.get("document") or getattr(result.get("nugget"), "document", None)
        return getattr(source, "name", None)

    def _remember_confirmed_value(self, attribute_name: str, result: Dict[str, Any]) -> None:
        if self._llm is None or not hasattr(self._llm, "add_confirmed_example"):
            return
        if result["message"] == "multi-match":
            for nugget in result["nuggets"]:
                self._llm.add_confirmed_example(attribute_name, nugget.text)
            return
        if result["message"] == "is-match":
            value = result["nugget"].text
        elif result["message"] == "custom-match":
            value = result["document"].text[result["start"]:result["end"]]
        else:  # a rejection gives no example value
            return
        self._llm.add_confirmed_example(attribute_name, value)

    @property
    def failure_counts(self) -> Optional[Dict[str, int]]:
        return getattr(self._llm, "failure_counts", None)

    @property
    def pick_log(self) -> Optional[List[Dict[str, Any]]]:
        return getattr(self._llm, "pick_log", None)
