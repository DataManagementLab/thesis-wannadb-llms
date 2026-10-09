import json
import logging
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from util import consider_overlap_as_match, get_document_by_name
from wannadb.data.data import Document, InformationNugget
from wannadb.data.signals import SentenceStartCharsSignal
from llm_feedback import LLMInteractionCallback

logger: logging.Logger = logging.getLogger(__name__)

HIDDEN_REASONING_LOG_CHARS = 1500
PROMPT_VERSION = "humanlike-v1"


class HumanLikeLLMCallback(LLMInteractionCallback):


    def __init__(self, *args, allow_custom_text: bool = False, check_in_full_text: bool = False,
                 vote_samples: int = 1, vote_temperature: float = 0.7, self_check_rounds: int = 0,
                 cautious_review: bool = False, show_accepted_in_prompt: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._allow_custom_text = allow_custom_text
        # one call on the tagged full text instead of first asking about the sentence alone
        self._check_in_full_text = check_in_full_text
        # majority over several sampled answers of the edit step
        self._vote_samples = vote_samples
        self._vote_temperature = vote_temperature
        # a reviewer call checks the answer against the values accepted so far, a rejected answer is asked again
        self._self_check_rounds = self_check_rounds
        # the reviewer only asks for a revision when it is sure, a doubtful revision breaks right answers
        self._cautious_review = cautious_review
        # control for the self-check: the answering call itself sees the values accepted so far
        self._show_accepted_in_prompt = show_accepted_in_prompt
        self._accepted_values: Dict[str, List[str]] = {}
        self._custom_not_found = 0

    @property
    def mode(self) -> str:
        if self._check_in_full_text:
            return "humanlike-direct"
        return "humanlike-custom" if self._allow_custom_text else "humanlike-nuggets"

    def _attribute_label(self, attribute_name: str) -> str:
        label = f'"{attribute_name}"'
        description = self._attribute_descriptions.get(attribute_name) if self._include_description else None
        return f"{label} (meaning: {description})" if description else label

    def _give_feedback(self, data: Dict[str, Any]) -> Dict[str, Any]:
        nuggets: List[InformationNugget] = list(data["nuggets"])
        attribute_name: str = data["attribute"].name

        # global random on purpose: the gold oracle draws its document the same way
        pick: InformationNugget = random.choice(nuggets)
        document: Document = pick.document

        record: Dict[str, Any] = {"pick": self._span_dict(pick), "step_a": None, "step_b": None}
        t0 = time.time()

        if self._check_in_full_text:
            result, step = self._edit(attribute_name, document, pick)
            step["decision"] = "confirm" if result.get("nugget") is pick and result["message"] == "is-match" else "edit"
            record["step_a"] = step
            self._remember_accepted(attribute_name, result)
            self._log(attribute_name, data, record, result, time.time() - t0)
            return result

        confirm_prompt = self._build_confirm_prompt(attribute_name, pick)
        answer, step_a = self._ask_json(confirm_prompt)
        decision = str((answer or {}).get("decision", "")).strip().lower()
        step_a["decision"] = decision
        record["step_a"] = step_a

        if decision == "confirm":
            result: Dict[str, Any] = {"message": "is-match", "nugget": pick, "not-a-match": None}
        else:
            # an unusable confirm answer goes to the edit step as well, which can still pick the guess
            result, record["step_b"] = self._edit(attribute_name, document, pick)

        self._remember_accepted(attribute_name, result)
        self._log(attribute_name, data, record, result, time.time() - t0)
        return result

    def _edit(self, attribute_name: str, document: Document, pick: InformationNugget
              ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        tagged_text, tag_of = self._tagged_document(document)
        prompt = self._build_edit_prompt(attribute_name, tagged_text, tag_of.get(id(pick)), pick)
        nugget_of = {tag: n for n in document.nuggets for tag in [tag_of.get(id(n))] if tag}

        if self._vote_samples > 1:
            answers = [self._answer_once(prompt, document, nugget_of, self._vote_temperature)
                       for _ in range(self._vote_samples)]
            span, step = self._majority(answers)
            step["votes"] = [a[0] for a in answers]
            step["extra_calls"] = [a[1] for a in answers if a[1] is not step]
        else:
            span, step = self._answer_once(prompt, document, nugget_of)

        checks: List[Dict[str, Any]] = []
        for _ in range(self._self_check_rounds):
            verdict, problem, check_step = self._review(attribute_name, tagged_text, tag_of, document, span)
            checks.append(check_step)
            if verdict != "revise":
                break
            revised_prompt = (prompt + f"\n\nA reviewer checked an earlier answer, {self._describe(span, document)}, "
                                       f"and found this problem: {problem}\nTake this into account and answer again.")
            span, revised = self._answer_once(revised_prompt, document, nugget_of)
            checks.append(revised)
        if checks:
            step["checks"] = checks
            step["final_span"] = span
        return self._result_of(span, document, pick), step

    def _answer_once(self, prompt: str, document: Document, nugget_of: Dict[str, InformationNugget],
                     temperature: float = 0) -> Tuple[Optional[List[int]], Dict[str, Any]]:
        """One edit answer, as the character span it points to, or None for "no value"."""
        answer, step = self._ask_json(prompt, temperature)
        if answer is None:
            answer, retry = self._ask_json(prompt + "\nKeep the reason very short.", temperature)
            step["retry"] = retry
        answer = answer or {}

        span_id = self._normalise_span_id(answer.get("span"))
        text = answer.get("text") if self._allow_custom_text else None
        step["span"], step["text"] = span_id, text

        if span_id is not None and span_id in nugget_of:
            chosen = nugget_of[span_id]
            return [chosen.start_char, chosen.end_char], step

        if isinstance(text, str) and text.strip():
            located = self._locate(document, text)
            if located is None:
                # one more chance, a paraphrase cannot be placed in the text
                answer2, retry = self._ask_json(
                    prompt + f'\nYour copied text "{text.strip()}" does not occur word for word in the document. '
                             "Copy the value exactly as it is written, or answer null.", temperature)
                step["retry_not_found"] = retry
                text2 = (answer2 or {}).get("text")
                span2 = self._normalise_span_id((answer2 or {}).get("span"))
                if span2 is not None and span2 in nugget_of:
                    chosen = nugget_of[span2]
                    return [chosen.start_char, chosen.end_char], step
                if isinstance(text2, str) and text2.strip():
                    located = self._locate(document, text2)
                if located is None:
                    self._custom_not_found += 1
            if located is not None:
                step["located"] = list(located)
                return list(located), step
        return None, step

    @staticmethod
    def _majority(answers: List[Tuple[Optional[List[int]], Dict[str, Any]]]
                  ) -> Tuple[Optional[List[int]], Dict[str, Any]]:
        """Answers whose spans overlap like a correct cell count as the same answer"""
        groups: List[List[int]] = []
        for i, (span, _) in enumerate(answers):
            for group in groups:
                other = answers[group[0]][0]
                if (span is None and other is None) or (span is not None and other is not None and
                                                        consider_overlap_as_match(other[0], other[1], span[0], span[1])):
                    group.append(i)
                    break
            else:
                groups.append([i])
        best = max(groups, key=lambda g: (len(g), -g[0]))
        return answers[best[0]]

    def _describe(self, span: Optional[List[int]], document: Document) -> str:
        if span is None:
            return "that the document contains no value for this column"
        return f'the value "{document.text[span[0]:span[1]]}"'

    def _recent_accepted(self, attribute_name: str) -> List[str]:
        return list(dict.fromkeys(reversed(self._accepted_values.get(attribute_name, []))))[:20]

    def _examples_shown(self, attribute_name: str) -> Optional[List[str]]:
        if self._show_accepted_in_prompt:
            return self._recent_accepted(attribute_name)
        return super()._examples_shown(attribute_name)

    def _example_lines(self, attribute_name: str) -> List[str]:
        if not self._show_accepted_in_prompt:
            return super()._example_lines(attribute_name)
        accepted = self._recent_accepted(attribute_name)
        if not accepted:
            return []
        return ["Values accepted so far for this column in other documents (they may contain mistakes): "
                + ", ".join(f'"{v}"' for v in accepted), ""]

    def _review(self, attribute_name: str, tagged_text: str, tag_of: Dict[int, str], document: Document,
                span: Optional[List[int]]) -> Tuple[str, str, Dict[str, Any]]:
        accepted = self._recent_accepted(attribute_name)
        lines = [
            f"An information extraction system fills the table column {self._attribute_label(attribute_name)} with "
            "values taken from documents.",
            f"For the document below, a first answer proposes {self._describe(span, document)}.",
        ]
        if accepted:
            lines += ["Values accepted so far for this column in other documents: "
                      + ", ".join(f'"{v}"' for v in accepted)]
        lines += [
            "",
            "Document:",
            '"""',
            tagged_text,
            '"""',
            "",
            "Check the proposal like a careful reviewer. Judge only from the text, not from outside knowledge.",
            f'Is it the value of "{attribute_name}" for what this document is about, and not a value that belongs '
            "to something else in the text? Does it cover the value without clearly extra or missing words?"
            + (" Does it have the same kind of form as the accepted values?" if accepted else ""),
            "If the proposal is fine, answer accept. Otherwise answer revise and say briefly what is wrong.",
            *(["Answer revise only if you are sure that the proposal is wrong. If in doubt, answer accept. The "
               "accepted values show the usual form, but they may contain mistakes themselves."]
              if self._cautious_review else []),
            "",
            'Respond with exactly one JSON object and nothing else: '
            '{"problem": "<one short sentence, empty if none>", "verdict": "accept" or "revise"}',
        ]
        answer, step = self._ask_json("\n".join(lines))
        verdict = str((answer or {}).get("verdict", "accept")).strip().lower()
        problem = str((answer or {}).get("problem", "")).strip()
        step["verdict"], step["problem"] = verdict, problem
        return verdict, problem, step

    @staticmethod
    def _result_of(span: Optional[List[int]], document: Document, pick: InformationNugget) -> Dict[str, Any]:
        if span is None:
            return {"message": "no-match-in-document", "nugget": pick, "not-a-match": pick}
        start, end = span
        if pick.start_char == start and pick.end_char == end:
            return {"message": "is-match", "nugget": pick, "not-a-match": None}
        same = next((n for n in document.nuggets if n.start_char == start and n.end_char == end), None)
        if same is not None:  # like a human clicking the existing nugget
            return {"message": "is-match", "nugget": same, "not-a-match": pick}
        return {"message": "custom-match", "document": document, "start": start, "end": end}

    def add_confirmed_example(self, attribute_name: str, value: str) -> None:
        super().add_confirmed_example(attribute_name, value)
        if value.strip():
            self._accepted_values.setdefault(attribute_name, []).append(value.strip())

    def seed_accepted_values(self, attribute_name: str, values: List[str]) -> None:
        """For replaying a logged round: the values the run had accepted before it."""
        self._accepted_values[attribute_name] = [v for v in values if v.strip()]

    def _remember_accepted(self, attribute_name: str, result: Dict[str, Any]) -> None:
        if result["message"] == "is-match":
            value = result["nugget"].text
        elif result["message"] == "custom-match":
            value = result["document"].text[result["start"]:result["end"]]
        else:
            return
        self._accepted_values.setdefault(attribute_name, []).append(value.strip())

    def _build_confirm_prompt(self, attribute_name: str, pick: InformationNugget) -> str:
        lines = [
            f"An information extraction system fills the table column {self._attribute_label(attribute_name)} with "
            "values taken from documents.",
            f'For one document it suggests the text marked with [[ and ]] as the value of "{attribute_name}". '
            "This is the sentence in which the text appears:",
            "",
            f'"{self._marked_sentence(pick)}"',
            "",
            *self._example_lines(attribute_name),
            f'Is the marked text the value of "{attribute_name}" for this document? Judge only from the text, not '
            "from outside knowledge.",
            'Answer confirm if it names the right value. Small differences at the boundaries, such as a missing '
            '"the" or "Inc.", do not matter.',
            "Answer edit if it names something else, or if it contains clearly more or clearly less than the value.",
            "",
            'Respond with exactly one JSON object and nothing else: '
            '{"reason": "<one short sentence>", "decision": "confirm" or "edit"}',
        ]
        return "\n".join(lines)

    def _build_edit_prompt(self, attribute_name: str, tagged_text: str, pick_tag: Optional[str],
                           pick: InformationNugget) -> str:
        suggestion = f'<{pick_tag}> ("{pick.text}")' if pick_tag else f'"{pick.text}"'
        lines = [
            f"An information extraction system fills the table column {self._attribute_label(attribute_name)} with "
            "values taken from documents.",
            "Below is one complete document. Every text span the system extracted is enclosed in tags like "
            "<n7>...</n7>.",
        ]
        if self._check_in_full_text:
            lines.append(f'The system currently suggests {suggestion} as the value of "{attribute_name}". Check this '
                         "suggestion against the whole document.")
        else:
            lines.append(f'The system currently suggests {suggestion} as the value of "{attribute_name}", and this '
                         "may be wrong.")
        lines += [
            "",
            "Document:",
            '"""',
            tagged_text,
            '"""',
            "",
            *self._example_lines(attribute_name),
        ]
        if self._check_in_full_text:
            lines += [
                "Judge only from the text, not from outside knowledge.",
                f'If the suggestion is the value of "{attribute_name}" in this document, answer with its id. Small '
                'differences at the boundaries, such as a missing "the" or "Inc.", do not matter.',
                "If another tagged span is the value, answer with that id. If several tagged spans contain the value, "
                "take the one that matches it best.",
            ]
        else:
            lines.append(f'Which tagged span is the value of "{attribute_name}" in this document? Judge only from the '
                         "text, not from outside knowledge. If several tagged spans contain the value, take the one "
                         "that matches it best.")
        if self._allow_custom_text:
            lines += [
                "If no tagged span fits, or the best one contains clearly more than the value, but the value is "
                'written in the document, copy only the value word for word from the document into "text" and set '
                '"span" to null.',
                f'If the document does not contain a value for "{attribute_name}" at all, answer null for both.',
                "",
                'Respond with exactly one JSON object and nothing else: {"reason": "<one short sentence>", '
                '"span": "<id such as n7>" or null, "text": "<exact words from the document>" or null}',
            ]
        else:
            lines += [
                "If none of the tagged spans is the value, answer null.",
                "",
                'Respond with exactly one JSON object and nothing else: '
                '{"reason": "<one short sentence>", "span": "<id such as n7>" or null}',
            ]
        return "\n".join(lines)

    @staticmethod
    def _sentence_bounds(document: Document, start: int, end: int) -> Tuple[int, int]:
        try:
            starts = sorted(document[SentenceStartCharsSignal])
        except KeyError:
            return max(0, start - 200), min(len(document.text), end + 200)
        left = max([s for s in starts if s <= start], default=0)
        right = min([s for s in starts if s >= end], default=len(document.text))
        return left, right

    def _marked_sentence(self, nugget: InformationNugget) -> str:
        document = nugget.document
        left, right = self._sentence_bounds(document, nugget.start_char, nugget.end_char)
        text = document.text
        marked = (text[left:nugget.start_char] + "[[" + text[nugget.start_char:nugget.end_char] + "]]"
                  + text[nugget.end_char:right])
        return " ".join(marked.split())

    @staticmethod
    def _tagged_document(document: Document) -> Tuple[str, Dict[int, str]]:
        tag_of: Dict[int, str] = {}
        parts: List[str] = []
        position = 0
        number = 0
        for nugget in sorted(document.nuggets, key=lambda n: (n.start_char, -n.end_char)):
            if nugget.start_char < position:
                continue  # overlapping spans cannot be tagged inline
            number += 1
            tag = f"n{number}"
            tag_of[id(nugget)] = tag
            parts.append(document.text[position:nugget.start_char])
            parts.append(f"<{tag}>{document.text[nugget.start_char:nugget.end_char]}</{tag}>")
            position = nugget.end_char
        parts.append(document.text[position:])
        return "".join(parts), tag_of

    @staticmethod
    def _normalise_span_id(value: Any) -> Optional[str]:
        if value is None:
            return None
        match = re.search(r"\d+", str(value))
        return f"n{int(match.group(0))}" if match else None

    @staticmethod
    def _locate(document: Document, text: str) -> Optional[Tuple[int, int]]:
        needle = text.strip().strip('"').strip("'").strip()
        if not needle:
            return None
        start = document.text.find(needle)
        if start != -1:
            return start, start + len(needle)
        # line breaks and case may differ from the copy
        pattern = r"\s+".join(re.escape(token) for token in needle.split())
        match = re.search(pattern, document.text, re.IGNORECASE)
        if match:
            return match.start(), match.end()
        trimmed = needle.rstrip(".,;:")
        if trimmed != needle:
            return HumanLikeLLMCallback._locate(document, trimmed)
        return None

    def _ask_json(self, prompt: str, temperature: float = 0) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        t0 = time.time()
        response = self._ask(prompt, temperature)
        seconds = time.time() - t0

        usage = response.usage
        if usage is not None:
            self._total_tokens_used += usage.total_tokens
        choice = response.choices[0]
        content: str = choice.message.content or ""
        if getattr(choice, "finish_reason", None) == "length":
            self._truncated_responses += 1
            logger.warning(f"LLM response hit the {self._max_response_tokens}-token limit and was cut off.")

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = self._salvage_json(content)
        if not isinstance(parsed, dict):
            parsed = None
            self._parse_failures += 1
            logger.warning(f"Could not parse a JSON object out of the LLM response: {content[:300]!r}")

        extra = getattr(choice.message, "model_extra", None) or {}
        hidden = extra.get("reasoning_content") or extra.get("reasoning") or ""
        step = {
            "prompt": prompt,
            "temperature": temperature,
            "raw_response": content,
            "reason": (parsed or {}).get("reason"),
            "hidden_reasoning": str(hidden)[:HIDDEN_REASONING_LOG_CHARS],
            "usage": self._usage_to_dict(usage),
            "seconds": seconds,
        }
        return parsed, step

    @staticmethod
    def _span_dict(nugget: InformationNugget) -> Dict[str, Any]:
        return {"document": nugget.document.name, "text": nugget.text,
                "start_char": nugget.start_char, "end_char": nugget.end_char}

    @property
    def failure_counts(self) -> Dict[str, int]:
        return {**super().failure_counts, "custom_text_not_found": self._custom_not_found}

    def _gold_mentions(self, attribute_name: str, document: Document) -> Optional[List[Dict[str, int]]]:
        dataset_attribute_name = self._user_attribute_name2dataset_attribute_name.get(attribute_name)
        if self._documents is None or dataset_attribute_name is None:
            return None
        gold_document = get_document_by_name(self._documents, document.name)
        if gold_document is None:
            return None
        return gold_document["mentions"].get(dataset_attribute_name, [])

    def _log(self, attribute_name: str, data: Dict[str, Any], record: Dict[str, Any], result: Dict[str, Any],
             duration: float) -> None:
        if result["message"] == "custom-match":
            document, start, end = result["document"], result["start"], result["end"]
        else:
            nugget = result["nugget"]
            document, start, end = nugget.document, nugget.start_char, nugget.end_char

        # only logged, never used to decide anything
        mentions = self._gold_mentions(attribute_name, document)
        agrees: Optional[bool] = None
        oracle_answer: Optional[str] = None
        if mentions is not None:
            def hit(s: int, e: int) -> bool:
                return any(consider_overlap_as_match(m["start_char"], m["end_char"], s, e) for m in mentions)

            agrees = (not mentions) if result["message"] == "no-match-in-document" else hit(start, end)
            pick = record["pick"]
            if hit(pick["start_char"], pick["end_char"]):
                oracle_answer = "confirm"
            elif any(hit(n.start_char, n.end_char) for n in document.nuggets):
                oracle_answer = "other_nugget"
            elif mentions:
                oracle_answer = "custom"
            else:
                oracle_answer = "no_value"

        def calls(step: Dict[str, Any]):
            yield step
            nested = [step[key] for key in ("retry", "retry_not_found") if step.get(key)]
            for s in nested + step.get("extra_calls", []) + step.get("checks", []):
                yield from calls(s)

        usage = None
        n_calls = 0
        for top in (record["step_a"], record["step_b"]):
            for s in calls(top) if top else []:
                usage = self._combine_usage(usage, s["usage"])
                n_calls += 1

        step_b = record["step_b"]
        out = {
            "timestamp": time.time(),
            "mode": self.mode,
            "prompt_version": PROMPT_VERSION,
            "include_description": self._include_description,
            "vote_samples": self._vote_samples,
            "self_check_rounds": self._self_check_rounds,
            "cautious_review": self._cautious_review,
            "show_accepted_in_prompt": self._show_accepted_in_prompt,
            "llm_calls": n_calls,
            "num_feedback": data.get("num-feedback"),
            "confirmed_examples_shown": self._examples_shown(attribute_name),
            "attribute": attribute_name,
            "num_candidates": len(data["nuggets"]),
            "pick": record["pick"],
            "decision": record["step_a"]["decision"],
            "step_a": record["step_a"],
            "step_b": step_b,
            # names kept from the list mode so the grid runner reads both
            "raw_response": record["step_a"]["raw_response"],
            "stage2": None if step_b is None else {"raw_response": step_b["raw_response"]},
            "usage": usage,
            "total_tokens_used_so_far": self._total_tokens_used,
            "duration_seconds": duration,
            "result_message": result["message"],
            "result_span": {"document": document.name, "start_char": start, "end_char": end,
                            "text": document.text[start:end]},
            "agrees_with_gold": agrees,
            "oracle_answer": oracle_answer,
            "parse_failures_so_far": self._parse_failures,
            "truncated_so_far": self._truncated_responses,
        }
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
