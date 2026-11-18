import abc
import json
import logging
import re
from typing import Any, Dict, List

import requests
from spacy.tokens import Doc

from wannadb import resources
from wannadb.configuration import register_configurable_element, BasePipelineElement
from wannadb.data.data import DocumentBase, InformationNugget
from wannadb.data.signals import LabelSignal, POSTagsSignal, SentenceStartCharsSignal, ExtractorNameSignal
from wannadb.interaction import BaseInteractionCallback
from wannadb.resources import StanzaNERPipeline, FigerNERPipeline
from wannadb.statistics import Statistics
from wannadb.status import BaseStatusCallback

logger: logging.Logger = logging.getLogger(__name__)


class BaseExtractor(BasePipelineElement, abc.ABC):
    """
    Base class for all extractors.

    Extractors derive the information nuggets from the documents.
    """
    identifier: str = "BaseExtractor"


########################################################################################################################
# actual extractors
########################################################################################################################


@register_configurable_element
class SpacyNERExtractor(BaseExtractor):
    """Extractor based on spacy's NER models."""

    identifier: str = "SpacyNERExtractor"

    required_signal_identifiers: Dict[str, List[str]] = {
        "nuggets": [],
        "attributes": [],
        "documents": []
    }

    generated_signal_identifiers: Dict[str, List[str]] = {
        "nuggets": [LabelSignal.identifier, POSTagsSignal.identifier, ExtractorNameSignal.identifier],
        "attributes": [],
        "documents": [SentenceStartCharsSignal.identifier]
    }

    def __init__(self, spacy_resource_identifier: str) -> None:
        """
        Initialize the SpacyNERExtractor.

        :param spacy_resource_identifier: identifier of the spacy model resource
        """
        super(SpacyNERExtractor, self).__init__()
        self._spacy_resource_identifier: str = spacy_resource_identifier

        # preload required resources
        resources.MANAGER.load(self._spacy_resource_identifier)
        logger.debug(f"Initialized '{self.identifier}'.")

    def _call(
            self,
            document_base: DocumentBase,
            interaction_callback: BaseInteractionCallback,
            status_callback: BaseStatusCallback,
            statistics: Statistics
    ) -> None:
        statistics["num_documents"] = len(document_base.documents)

        for ix, document in enumerate(document_base.documents):
            self._use_status_callback(status_callback, ix, len(document_base.documents))

            spacy_output: Doc = resources.MANAGER[self._spacy_resource_identifier](document.text)
            sentence_start_chars: List[int] = []

            # transform the spacy output into the document and nuggets
            for sentence in spacy_output.sents:
                sentence_start_chars.append(sentence.start_char)

            document[SentenceStartCharsSignal] = SentenceStartCharsSignal(sentence_start_chars)

            for entity in spacy_output.ents:
                nugget: InformationNugget = InformationNugget(
                    document=document,
                    start_char=entity.start_char,
                    end_char=entity.end_char
                )

                nugget[POSTagsSignal] = POSTagsSignal([])  # TODO: gather pos tags
                nugget[LabelSignal] = LabelSignal(entity.label_)
                nugget[ExtractorNameSignal] = ExtractorNameSignal(self.identifier)

                document.nuggets.append(nugget)

                statistics["num_nuggets"] += 1
                statistics["spacy_entity_type_dist"][entity.label_] += 1

    def to_config(self) -> Dict[str, Any]:
        return {
            "identifier": self.identifier,
            "spacy_resource_identifier": self._spacy_resource_identifier
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SpacyNERExtractor":
        return cls(config["spacy_resource_identifier"])


@register_configurable_element
class StanzaNERExtractor(BaseExtractor):
    """Extractor based on Stanza's NER model."""

    identifier: str = "StanzaNERExtractor"

    required_signal_identifiers: Dict[str, List[str]] = {
        "nuggets": [],
        "attributes": [],
        "documents": []
    }

    generated_signal_identifiers: Dict[str, List[str]] = {
        "nuggets": [LabelSignal.identifier, POSTagsSignal.identifier, ExtractorNameSignal.identifier],
        "attributes": [],
        "documents": [SentenceStartCharsSignal.identifier]
    }

    def __init__(self) -> None:
        """Initialize the StanzaNERExtractor."""
        super(StanzaNERExtractor, self).__init__()

        # preload required resources
        resources.MANAGER.load(StanzaNERPipeline)
        logger.debug(f"Initialized '{self.identifier}'.")

    def _call(
            self,
            document_base: DocumentBase,
            interaction_callback: BaseInteractionCallback,
            status_callback: BaseStatusCallback,
            statistics: Statistics
    ) -> None:
        statistics["num_documents"] = len(document_base.documents)

        for ix, document in enumerate(document_base.documents):
            self._use_status_callback(status_callback, ix, len(document_base.documents))

            stanza_output = resources.MANAGER[StanzaNERPipeline](document.text)

            sentence_start_chars: List[int] = []

            # transform the stanza output into the document and nuggets
            for sentence in stanza_output.sentences:
                sentence_start_chars.append(sentence.tokens[0].start_char)

                for entity in sentence.entities:
                    nugget: InformationNugget = InformationNugget(
                        document=document,
                        start_char=entity.start_char,
                        end_char=entity.start_char + len(entity.text)
                    )

                    nugget[POSTagsSignal] = POSTagsSignal([word.xpos for word in entity.words])
                    nugget[LabelSignal] = LabelSignal(entity.type)
                    nugget[ExtractorNameSignal] = ExtractorNameSignal(self.identifier)

                    document.nuggets.append(nugget)

                    statistics["num_nuggets"] += 1
                    statistics["stanza_entity_type_dist"][entity.type] += 1

            document[SentenceStartCharsSignal] = SentenceStartCharsSignal(sentence_start_chars)

    def to_config(self) -> Dict[str, Any]:
        return {
            "identifier": self.identifier
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "StanzaNERExtractor":
        return cls()


@register_configurable_element
class NumberPairExtractor(BaseExtractor):
    """
    Rule-based extractor that adds both numbers of a number pair as nuggets, e.g. "114" and "110" of the score
    "114 - 110" or "30" and "7" of the record "( 30 - 7 )".

    THESIS CHANGE (LLM feedback thesis, 2026-11-13), not part of the original WannaDB, see CORE_CHANGES.md.
    Why: Stanza and spaCy tag such a pair as ONE entity. The single numbers are then no nuggets, so WannaDB can fill
    a column like the points of the losing team only through custom matches. On the Rotowire test games the points of
    the losing team were a nugget in 39 % of the games and are one in 100 % with this extractor.

    The extractor must run after the NER extractors. A number that already is a nugget with exactly this span is
    skipped. The new nuggets get the label CARDINAL, which is the label the NER extractors give most such pairs.
    Chains of more than two numbers (dates like 2014-11-05), decimal numbers and numbers inside words are left alone.
    """

    identifier: str = "NumberPairExtractor"

    # two whole numbers joined by a hyphen or an en dash on one line; neither number may be part of a word or of a
    # number with a decimal point or comma, and no third number may follow after a further hyphen
    PAIR_PATTERN = re.compile(
        r"(?<![\w.,])(\d+)[ \t]*[\-\u2013][ \t]*(\d+)(?!\w|[.,]\d|[ \t]*[\-\u2013][ \t]*\d)")
    # a number and a hyphen right before a pair: the pair is the end of a chain like 1 - 2 - 3
    CHAIN_BEFORE_PATTERN = re.compile(r"\d[ \t]*[\-\u2013][ \t]*$")

    required_signal_identifiers: Dict[str, List[str]] = {
        "nuggets": [],
        "attributes": [],
        "documents": []
    }

    generated_signal_identifiers: Dict[str, List[str]] = {
        "nuggets": [LabelSignal.identifier, POSTagsSignal.identifier, ExtractorNameSignal.identifier],
        "attributes": [],
        "documents": []
    }

    def __init__(self) -> None:
        super(NumberPairExtractor, self).__init__()
        logger.debug(f"Initialized '{self.identifier}'.")

    def _call(
            self,
            document_base: DocumentBase,
            interaction_callback: BaseInteractionCallback,
            status_callback: BaseStatusCallback,
            statistics: Statistics
    ) -> None:
        statistics["num_documents"] = len(document_base.documents)

        for ix, document in enumerate(document_base.documents):
            self._use_status_callback(status_callback, ix, len(document_base.documents))

            existing_spans = {(nugget.start_char, nugget.end_char) for nugget in document.nuggets}

            for pair in self.PAIR_PATTERN.finditer(document.text):
                # search(text, pos, endpos) lets "$" match at the start of the pair
                if self.CHAIN_BEFORE_PATTERN.search(document.text, max(0, pair.start() - 20), pair.start()):
                    continue
                statistics["num_pairs"] += 1
                for group in (1, 2):
                    span = (pair.start(group), pair.end(group))
                    if span in existing_spans:
                        statistics["num_numbers_already_nuggets"] += 1
                        continue
                    existing_spans.add(span)

                    nugget: InformationNugget = InformationNugget(
                        document=document,
                        start_char=span[0],
                        end_char=span[1]
                    )

                    nugget[POSTagsSignal] = POSTagsSignal(["CD"])  # Penn Treebank tag of a cardinal number
                    nugget[LabelSignal] = LabelSignal("CARDINAL")
                    nugget[ExtractorNameSignal] = ExtractorNameSignal(self.identifier)

                    document.nuggets.append(nugget)

                    statistics["num_nuggets"] += 1

    def to_config(self) -> Dict[str, Any]:
        return {
            "identifier": self.identifier
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "NumberPairExtractor":
        return cls()


@register_configurable_element
class FigerNERExtractor(BaseExtractor):
    """
    Extractor based on Figer's NER model
    (using CoreNLP for basic extraction and fine-graned labeling on top).
    """

    identifier: str = "FigerNERExtractor"

    required_signal_identifiers: Dict[str, List[str]] = {
        "nuggets": [],
        "attributes": [],
        "documents": []
    }

    generated_signal_identifiers: Dict[str, List[str]] = {
        "nuggets": [LabelSignal.identifier],
        "attributes": [],
        "documents": [SentenceStartCharsSignal.identifier]
    }

    def __init__(self) -> None:
        """Initialize the FigerNERExtractor."""
        super().__init__()

        # preload required resources
        resources.MANAGER.load(FigerNERPipeline)
        logger.debug(f"Initialized '{self.identifier}'.")

    def _call(
            self,
            document_base: DocumentBase,
            interaction_callback: BaseInteractionCallback,
            status_callback: BaseStatusCallback,
            statistics: Statistics
    ) -> None:
        statistics["num_documents"] = len(document_base.documents)

        base_url = resources.MANAGER[FigerNERPipeline]

        for ix, document in enumerate(document_base.documents):
            self._use_status_callback(status_callback, ix, len(document_base.documents))

            # Run FIGER on document (truncate to first 8000 chars if necessary due to server limitations)
            r = requests.get(base_url, params={'text': document.text[:8000]})
            if r.status_code == 200:
                answer = json.loads(r.text)
                if answer["status"] == 200:
                    sentence_start_chars: List[int] = answer["sentence_offsets"]

                    for raw_nugget in answer["data"]:
                        nugget: InformationNugget = InformationNugget(
                            document=document,
                            start_char=raw_nugget["start_char"],
                            end_char=raw_nugget["end_char"]
                        )

                        # nugget[POSTagsSignal] = POSTagsSignal([word.xpos for word in entity.words])

                        # Label format from FIGER is e.g.
                        # "/location@1.4898770776826524,/organization/company@0.17639383484191654,/location/country@0.25034040521054085"
                        # Extract first label (without numeric value)
                        label_string = raw_nugget['label'].split(',')[0].split('@')[0][1:].replace("/", " ")
                        nugget[LabelSignal] = LabelSignal(label_string)
                        document.nuggets.append(nugget)

                        statistics["num_nuggets"] += 1
                        statistics["figer_label_dist"][label_string] += 1

                    document[SentenceStartCharsSignal] = SentenceStartCharsSignal(sentence_start_chars)
                else:
                    print(logger.warning(f"Failed to run FIGER on document '{document.name}' with error '{answer['error']}'"))
            else:
                logger.warning(f"Failed to run FIGER on document '{document.name}'")

    def to_config(self) -> Dict[str, Any]:
        return {
            "identifier": self.identifier
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FigerNERExtractor":
        return cls()
