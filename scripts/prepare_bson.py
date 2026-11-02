
import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wannadb.data.data import Attribute, DocumentBase
from wannadb.interaction import EmptyInteractionCallback
from wannadb.preprocessing.embedding import SBERTDocumentSentenceEmbedder, SBERTLabelEmbedder
from wannadb.preprocessing.label_paraphrasing import SplitAttributeNameLabelParaphraser
from wannadb.preprocessing.other_processing import ContextSentenceCacher
from wannadb.resources import ResourceManager
from wannadb.statistics import Statistics
from wannadb.status import EmptyStatusCallback


def dataset_attributes(name: str):
    path = ROOT / "wannadb_datsets" / "datasets" / name / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_dataset", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.ATTRIBUTES)


def main() -> int:
    name = sys.argv[1]
    source = ROOT / "bson_files" / f"{name}-preprocessed.bson"
    target = ROOT / "bson_files" / f"{name}-preprocessed-fixed.bson"
    t0 = time.time()
    db = DocumentBase.from_bson(source.read_bytes())
    db._attributes = [Attribute(a) for a in dataset_attributes(name)]
    print(f"{name}: {len(db.documents)} documents, {len(db.nuggets)} nuggets, loaded in {time.time() - t0:.0f}s")
    with ResourceManager():
        for element in (SplitAttributeNameLabelParaphraser(do_lowercase=True, splitters=[" ", "_"]),
                        ContextSentenceCacher(),
                        SBERTLabelEmbedder("SBERTBertLargeNliMeanTokensResource"),
                        SBERTDocumentSentenceEmbedder("SBERTBertLargeNliMeanTokensResource")):
            t1 = time.time()
            element(db, EmptyInteractionCallback(), EmptyStatusCallback(), Statistics(do_collect=False))
            print(f"{type(element).__name__} done in {time.time() - t1:.0f}s")
    # the runner sets its own attributes, only the documents with their signals are stored
    tmp = target.with_suffix(".bson.tmp")  # a grid may be waiting for the file
    tmp.write_bytes(DocumentBase(db.documents, []).to_bson())
    tmp.replace(target)
    print(f"wrote {target.name} ({target.stat().st_size / 1e6:.0f} MB) after {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
