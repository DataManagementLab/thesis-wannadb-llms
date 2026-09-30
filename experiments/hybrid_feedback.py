
from typing import Any, Dict, List, Optional, Tuple
from wannadb.interaction import BaseInteractionCallback


class HybridInteractionCallback(BaseInteractionCallback):
    """Rounds 1 to human_rounds of every attribute are answered by the human, the rest by the LLM"""

    def __init__(self, human: BaseInteractionCallback, llm: Optional[BaseInteractionCallback],
                 human_rounds: int) -> None:
        self._human = human
        self._llm = llm
        self._human_rounds = human_rounds
        # (attribute, round, answerer, message)
        self.answers: List[Tuple[str, int, str, str]] = []

    def _call(self, pipeline_element_identifier: str, data: Dict[str, Any]) -> Dict[str, Any]:
        if "do-attribute-request" in data.keys():
            return {"do-attribute": True}

        attribute_name: str = data["attribute"].name
        round_no: int = data["num-feedback"]

        if round_no <= self._human_rounds or self._llm is None:
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
