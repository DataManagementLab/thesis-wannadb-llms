import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import APIConnectionError, APITimeoutError, OpenAI, InternalServerError, RateLimitError

from util import consider_overlap_as_match, get_document_by_name
from wannadb.data.data import InformationNugget
from wannadb.data.signals import CachedContextSentenceSignal
from wannadb.interaction import BaseInteractionCallback

logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_BASE_URL: str = "http://10.0.21.72:13505/v1"
DEFAULT_MODEL: str = "unsloth/DeepSeek-V4-Flash-0731"


class TokenBudgetExceeded(RuntimeError):
    """Raised when the configured token cap for a run would be exceeded by another call"""


class LLMInteractionCallback(BaseInteractionCallback):
    """Interaction callback that lets DeepSeek answer instead of a human or the ground-truth oracle"""

    def __init__(
            self,
            documents: Optional[List[Dict[str, Any]]] = None,
            user_attribute_name2dataset_attribute_name: Optional[Dict[str, str]] = None,
            *,
            base_url: str = DEFAULT_BASE_URL,
            model: str = DEFAULT_MODEL,
            api_key: Optional[str] = None,
            extra_body: Optional[Dict[str, Any]] = None,
            timeout_seconds: float = 300.0,
            include_description: bool = False,
            attribute_descriptions: Optional[Dict[str, str]] = None,
            include_context: bool = False,
            max_response_tokens: int = 16_384,
            max_total_tokens: Optional[int] = 2_000_000,
            log_path: Optional[str] = None,
            random_seed: Optional[int] = None,
            random_rejection_pick: bool = True,
            search_document_on_reject: bool = True,
            max_document_nuggets: int = 60,
            max_retries: int = 5,
            show_confirmed_examples: bool = False,
    ) -> None:
        self._documents = documents
        self._user_attribute_name2dataset_attribute_name = user_attribute_name2dataset_attribute_name or {}

        self._client = OpenAI(base_url=base_url, api_key=api_key or "not-needed", timeout=timeout_seconds)
        self._model = model
        self._extra_body = extra_body

        self._include_description = include_description
        self._attribute_descriptions = attribute_descriptions or {}
        self._include_context = include_context

        self._max_response_tokens = max_response_tokens
        self._max_total_tokens = max_total_tokens
        self._total_tokens_used = 0

        # unusable answers still produce a valid message, but are counted so they can be reported
        self._parse_failures = 0
        self._truncated_responses = 0

        self._rng = random.Random(random_seed)
        self._random_rejection_pick = random_rejection_pick
        self._search_document_on_reject = search_document_on_reject
        self._max_document_nuggets = max_document_nuggets
        self._max_retries = max_retries

        self._show_confirmed_examples = show_confirmed_examples
        self.confirmed_examples: Dict[str, List[str]] = {}

        if log_path is None:
            log_dir = Path(__file__).parent / "llm_feedback_logs"
            log_dir.mkdir(exist_ok=True)
            log_path = str(log_dir / f"run_{int(time.time())}.jsonl")
        self._log_path = log_path
        logger.info(f"LLMInteractionCallback logging interactions to '{self._log_path}'.")

    def _call(self, pipeline_element_identifier: str, data: Dict[str, Any]) -> Dict[str, Any]:
        if "do-attribute-request" in data.keys():
            # the gold oracles never skip an attribute either
            return {"do-attribute": True}

        return self._give_feedback(data)

    def _give_feedback(self, data: Dict[str, Any]) -> Dict[str, Any]:
        nuggets: List[InformationNugget] = list(data["nuggets"])
        attribute_name: str = data["attribute"].name

        # global random on purpose, so a round uses up as much randomness as a gold oracle round
        rejection_pick: Optional[InformationNugget] = None
        if self._random_rejection_pick:
            rejection_pick = random.choice(nuggets)

        # shuffled against position bias, the order is logged
        order: List[int] = list(range(len(nuggets)))
        self._rng.shuffle(order)

        prompt: str = self._build_prompt(attribute_name, nuggets, order)

        t0: float = time.time()
        response = self._ask(prompt)
        duration: float = time.time() - t0

        chosen_shuffled_ix, reasoning, raw_content, usage = self._parse_response(response, len(nuggets))

        usage_dict = self._usage_to_dict(usage)
        stage2: Optional[Dict[str, Any]] = None

        if chosen_shuffled_ix is not None:
            chosen_ix: Optional[int] = order[chosen_shuffled_ix]
            chosen_nugget: InformationNugget = nuggets[chosen_ix]
            result: Dict[str, Any] = {"message": "is-match", "nugget": chosen_nugget, "not-a-match": None}
        else:
            chosen_ix = None
            reject_target: InformationNugget = rejection_pick if rejection_pick is not None else nuggets[0]

            if self._search_document_on_reject:
                # stage 2: check all nuggets of the document before rejecting it
                doc_nuggets = list(reject_target.document.nuggets)
                if len(doc_nuggets) > self._max_document_nuggets:
                    logger.warning(
                        f"Document '{reject_target.document.name}' has {len(doc_nuggets)} nuggets, "
                        f"capping stage-2 prompt at {self._max_document_nuggets}."
                    )
                    doc_nuggets = doc_nuggets[: self._max_document_nuggets]

                doc_order: List[int] = list(range(len(doc_nuggets)))
                self._rng.shuffle(doc_order)
                doc_prompt = self._build_document_prompt(attribute_name, doc_nuggets, doc_order)

                t1 = time.time()
                doc_response = self._ask(doc_prompt)
                stage2_duration = time.time() - t1
                doc_ix, doc_reasoning, doc_raw, doc_usage = self._parse_response(doc_response, len(doc_nuggets))
                doc_usage_dict = self._usage_to_dict(doc_usage)

                duration += stage2_duration
                usage_dict = self._combine_usage(usage_dict, doc_usage_dict)
                stage2 = {
                    "num_candidates": len(doc_nuggets),
                    "candidate_order": doc_order,
                    "reasoning": doc_reasoning,
                    "raw_response": doc_raw,
                    "usage": doc_usage_dict,
                    "duration_seconds": stage2_duration,
                    "resolved_index": None,
                }

                if doc_ix is not None:
                    found_nugget = doc_nuggets[doc_order[doc_ix]]
                    stage2["resolved_index"] = doc_order[doc_ix]
                    stage2["resolved_nugget"] = {"text": found_nugget.text, "start_char": found_nugget.start_char,
                                                 "end_char": found_nugget.end_char}
                    result = {"message": "is-match", "nugget": found_nugget, "not-a-match": reject_target}
                else:
                    result = {"message": "no-match-in-document", "nugget": reject_target, "not-a-match": reject_target}
            else:
                result = {"message": "no-match-in-document", "nugget": reject_target, "not-a-match": reject_target}

        self._log_interaction(
            attribute_name=attribute_name,
            nuggets=nuggets,
            order=order,
            reasoning=reasoning,
            raw_content=raw_content,
            usage=usage_dict,
            duration=duration,
            result_message=result["message"],
            chosen_ix=chosen_ix,
            rejection_pick=rejection_pick,
            stage2=stage2,
            num_feedback=data.get("num-feedback"),
            examples_shown=self._examples_shown(attribute_name),
            confirmed=result["nugget"] if result["message"] == "is-match" else None,
        )

        return result

    def add_confirmed_example(self, attribute_name: str, value: str) -> None:
        values = self.confirmed_examples.setdefault(attribute_name, [])
        value = value.strip()
        if value and value not in values:
            values.append(value)

    def _examples_shown(self, attribute_name: str) -> Optional[List[str]]:
        if not self._show_confirmed_examples:
            return None
        return list(self.confirmed_examples.get(attribute_name, []))

    def _example_lines(self, attribute_name: str) -> List[str]:
        examples = self._examples_shown(attribute_name)
        if not examples:
            return []
        return [
            "In other documents, a human has already confirmed these values for this column:",
            ", ".join(f'"{v}"' for v in examples),
            "",
        ]

    def _build_prompt(self, attribute_name: str, nuggets: List[InformationNugget], order: List[int]) -> str:
        attribute_label = f'"{attribute_name}"'
        if self._include_description:
            description = self._attribute_descriptions.get(attribute_name)
            if description:
                attribute_label += f' (meaning: {description})'

        lines = [
            f"An information extraction system is filling a table column {attribute_label}.",
            "Each snippet below is the system's current best guess for that column in a "
            "DIFFERENT document, so several of them may be correct at the same time.",
            "",
            *self._example_lines(attribute_name),
            "Pick the index of one snippet that really is the value of that column in its own "
            "document. If none of them is correct, answer with null instead of a number.",
            "",
        ]

        for position, original_ix in enumerate(order):
            nugget = nuggets[original_ix]
            document_id = Path(nugget.document.name.replace("\\", "/")).stem
            line = f'{position}: [{document_id}] "{nugget.text}"'
            if self._include_context:
                context = self._context_sentence(nugget)
                if context:
                    line += f'\n     context: "{context}"'
            lines.append(line)

        lines.append("")
        lines.append(
            "Respond with exactly one JSON object and nothing else, reasoning before the answer: "
            '{"reasoning": "<your reasoning>", "matching_index": <int or null>}'
        )
        return "\n".join(lines)

    def _build_document_prompt(
            self, attribute_name: str, nuggets: List[InformationNugget], order: List[int]
    ) -> str:
        attribute_label = f'"{attribute_name}"'
        if self._include_description:
            description = self._attribute_descriptions.get(attribute_name)
            if description:
                attribute_label += f' (meaning: {description})'

        doc_name = Path(nuggets[0].document.name.replace("\\", "/")).stem if nuggets else "?"
        lines = [
            f"None of an earlier shortlist was the value of the table column {attribute_label} "
            f"in document {doc_name}.",
            "Below are ALL text snippets that were extracted from that same document, not just a "
            "shortlist. Check whether any of them is actually the value of that column.",
            "",
            *self._example_lines(attribute_name),
            "Pick the index of the one that is. If truly none of them is correct, answer with "
            "null instead of a number.",
            "",
        ]

        for position, original_ix in enumerate(order):
            nugget = nuggets[original_ix]
            line = f'{position}: "{nugget.text}"'
            if self._include_context:
                context = self._context_sentence(nugget)
                if context:
                    line += f'\n     context: "{context}"'
            lines.append(line)

        lines.append("")
        lines.append(
            "Respond with exactly one JSON object and nothing else, reasoning before the answer: "
            '{"reasoning": "<your reasoning>", "matching_index": <int or null>}'
        )
        return "\n".join(lines)

    @staticmethod
    def _context_sentence(nugget: InformationNugget) -> str:
        try:
            return str(nugget[CachedContextSentenceSignal]["text"])
        except Exception:
            return ""

    def _ask(self, prompt: str):
        if self._max_total_tokens is not None and self._total_tokens_used >= self._max_total_tokens:
            raise TokenBudgetExceeded(
                f"Token budget of {self._max_total_tokens} already used up "
                f"({self._total_tokens_used} tokens so far); refusing to send another request."
            )

        last_error: Optional[BaseException] = None
        for attempt in range(self._max_retries):
            try:
                return self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self._max_response_tokens,
                    temperature=0,
                    extra_body=self._extra_body,
                )
            except (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError) as e:
                last_error = e
                wait = min(2 ** attempt, 300)
                logger.warning(
                    f"DeepSeek request failed ({type(e).__name__}: {e}), "
                    f"attempt {attempt + 1}/{self._max_retries}, retrying in {wait}s."
                )
                if attempt + 1 < self._max_retries:
                    time.sleep(wait)

        logger.error(f"DeepSeek endpoint unreachable after {self._max_retries} attempts: {last_error}")
        raise last_error

    def _parse_response(self, response, num_candidates: int) -> Tuple[Optional[int], str, str, Any]:
        usage = response.usage
        if usage is not None:
            self._total_tokens_used += usage.total_tokens

        choice = response.choices[0]
        raw_content: str = choice.message.content or ""

        if getattr(choice, "finish_reason", None) == "length":
            self._truncated_responses += 1
            logger.warning(
                f"LLM response hit the {self._max_response_tokens}-token limit and was cut off. "
                "Raise max_response_tokens -- this answer is not trustworthy."
            )

        parsed: Optional[Dict[str, Any]]
        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError:
            parsed = self._salvage_json(raw_content)

        reasoning: str = ""
        matching_ix: Optional[int] = None

        if parsed is not None:
            reasoning = str(parsed.get("reasoning", ""))
            raw_ix = parsed.get("matching_index")
            if raw_ix is not None:
                try:
                    matching_ix = int(raw_ix)
                except (TypeError, ValueError):
                    logger.warning(f"LLM returned a non-integer matching_index: {raw_ix!r}. Treating as no-match.")
                    matching_ix = None

            if matching_ix is not None and not (0 <= matching_ix < num_candidates):
                logger.warning(
                    f"LLM returned out-of-range matching_index {matching_ix} for {num_candidates} candidates. "
                    "Treating as no-match."
                )
                matching_ix = None
        else:
            self._parse_failures += 1
            logger.warning(f"Could not parse a JSON object out of the LLM response: {raw_content!r}")

        return matching_ix, reasoning, raw_content, usage

    @property
    def failure_counts(self) -> Dict[str, int]:
        return {
            "parse_failures": self._parse_failures,
            "truncated_responses": self._truncated_responses,
        }

    @staticmethod
    def _usage_to_dict(usage: Any) -> Optional[Dict[str, int]]:
        if usage is None:
            return None
        return {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }

    @staticmethod
    def _combine_usage(
            a: Optional[Dict[str, int]], b: Optional[Dict[str, int]]
    ) -> Optional[Dict[str, int]]:
        if a is None:
            return b
        if b is None:
            return a
        return {k: a[k] + b[k] for k in a}

    @staticmethod
    def _salvage_json(text: str) -> Optional[Dict[str, Any]]:
        """For answers that wrap the JSON object in text or a code block."""
        if not text:
            return None
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    def _log_interaction(
            self,
            attribute_name: str,
            nuggets: List[InformationNugget],
            order: List[int],
            reasoning: str,
            raw_content: str,
            usage: Optional[Dict[str, int]],
            duration: float,
            result_message: str,
            chosen_ix: Optional[int],
            rejection_pick: Optional[InformationNugget] = None,
            stage2: Optional[Dict[str, Any]] = None,
            num_feedback: Optional[int] = None,
            examples_shown: Optional[List[str]] = None,
            confirmed: Optional[InformationNugget] = None,
    ) -> None:
        record: Dict[str, Any] = {
            "timestamp": time.time(),
            "num_feedback": num_feedback,
            "confirmed_examples_shown": examples_shown,
            "attribute": attribute_name,
            "num_candidates": len(nuggets),
            "candidate_order": order,
            "candidates": [
                {"document": n.document.name, "text": n.text, "start_char": n.start_char, "end_char": n.end_char}
                for n in nuggets
            ],
            "reasoning": reasoning,
            "raw_response": raw_content,
            "usage": usage,
            "total_tokens_used_so_far": self._total_tokens_used,
            "duration_seconds": duration,
            "result_message": result_message,
            "chosen_index": chosen_ix,
            "parse_failures_so_far": self._parse_failures,
            "truncated_so_far": self._truncated_responses,
            "rejection_pick": None if rejection_pick is None else {
                "document": rejection_pick.document.name,
                "text": rejection_pick.text,
                "start_char": rejection_pick.start_char,
                "end_char": rejection_pick.end_char,
            },
            "stage2": stage2,
        }

        # stage 1 or stage 2 confirmation; only logged, never used to decide anything
        if self._documents is not None and confirmed is not None:
            record["agrees_with_gold"] = self._check_agreement(attribute_name, confirmed)

        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _check_agreement(self, attribute_name: str, nugget: InformationNugget) -> Optional[bool]:
        dataset_attribute_name = self._user_attribute_name2dataset_attribute_name.get(attribute_name)
        if dataset_attribute_name is None:
            return None
        document = get_document_by_name(self._documents, nugget.document.name)
        if document is None:
            return None
        for mention in document["mentions"].get(dataset_attribute_name, []):
            if consider_overlap_as_match(mention["start_char"], mention["end_char"], nugget.start_char, nugget.end_char):
                return True
        return False
