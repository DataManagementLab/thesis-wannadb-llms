
from typing import Any, Dict, List, Optional, Set, Tuple
from wannadb.interaction import BaseInteractionCallback

SCHEDULES = ("handoff", "examples", "llmfirst", "interleave", "sandwich")


def human_round_numbers(variant: str, k: int, max_rounds: int) -> Set[int]:
    if variant == "llmfirst":
        return set(range(max_rounds - k + 1, max_rounds + 1))
    if variant == "interleave":  # spread evenly, starting with the human
        return {i * max_rounds // k + 1 for i in range(k)} if k else set()
    if variant == "sandwich":  # half at the start, half at the end
        return set(range(1, (k + 1) // 2 + 1)) | set(range(max_rounds - k // 2 + 1, max_rounds + 1))
    return set(range(1, k + 1))  # handoff and examples: the human first


class HybridInteractionCallback(BaseInteractionCallback):
    """Rounds 1 to human_rounds of every attribute are answered by the human, the rest by the LLM.
    human_round_numbers replaces that with any set of rounds, e.g. the LLM first or alternating."""

    def __init__(self, human: BaseInteractionCallback, llm: Optional[BaseInteractionCallback],
                 human_rounds: int, human_round_numbers: Optional[Set[int]] = None) -> None:
        self._human = human
        self._llm = llm
        self._human_rounds = human_rounds
        self._human_round_numbers = human_round_numbers
        # (attribute, round, answerer, message)
        self.answers: List[Tuple[str, int, str, str]] = []

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
            result = self._human(pipeline_element_identifier, data)
            self.answers.append((attribute_name, round_no, "human", result["message"]))
            self._remember_confirmed_value(attribute_name, result)
            return result

        result = self._llm(pipeline_element_identifier, data)
        self.answers.append((attribute_name, round_no, "llm", result["message"]))
        return result

    def _remember_confirmed_value(self, attribute_name: str, result: Dict[str, Any]) -> None:
        if self._llm is None or not hasattr(self._llm, "add_confirmed_example"):
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
