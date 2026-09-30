import sys

sys.path.insert(0, r"C:\Users\hamid\Desktop\Uni\BachelorThesis")
sys.path.insert(0, r"C:\Users\hamid\Desktop\Uni\BachelorThesis\experiments")

from wannadb.data.data import Attribute, DocumentBase
from llm_feedback import LLMInteractionCallback

BSON_PATH = r"C:\Users\hamid\Desktop\Uni\BachelorThesis\bson_files\aviation-preprocessed.bson"
TARGET_DOC = "AAB-00-01"
ATTRIBUTE_NAME = "air_carrier"


def main() -> None:
    with open(BSON_PATH, "rb") as f:
        document_base = DocumentBase.from_bson(f.read())

    candidates = [d for d in document_base.documents if TARGET_DOC in d.name]
    if not candidates:
        print(f"Document '{TARGET_DOC}' not found.")
        sys.exit(1)
    doc = candidates[0]
    print(f"Document: {doc.name}  ({len(doc.nuggets)} nuggets)")

    nuggets_by_span = {(n.start_char, n.end_char): n for n in doc.nuggets}
    picked = [
        nuggets_by_span[(100, 121)],  # "Sunjet Aviation, Inc.", the correct one
        nuggets_by_span[(126, 133)],  # "Sanford", a city
        nuggets_by_span[(469, 483)],  # "U.S. Air Force", wrong organization
        nuggets_by_span[(495, 513)],  # "Air National Guard", wrong organization
    ]

    attribute = Attribute(ATTRIBUTE_NAME)
    data = {
        "max-distance": 0.2,
        "max-distance-change": 0.0,
        "nuggets": tuple(picked),
        "nugget-updates-context": None,
        "all-guessed-nugget-matches": tuple(picked),
        "attribute": attribute,
        "num-feedback": 1,
        "num-nuggets-above": 0,
        "num-nuggets-below": 0,
        "sampling-mode": "smoke-test",
    }

    callback = LLMInteractionCallback(random_seed=42)

    print("\n--- do-attribute-request ---")
    gate = callback(pipeline_element_identifier="smoke-test", data={"do-attribute-request": None, "attribute": attribute})
    print(gate)

    print("\n--- feedback round ---")
    print("Candidates shown (original order):")
    for i, n in enumerate(picked):
        print(f"  [{i}] {n.text!r}")

    result = callback(pipeline_element_identifier="smoke-test", data=data)
    print("\nResult:")
    print(" message:     ", result["message"])
    print(" nugget:      ", result["nugget"].text)
    print(" not-a-match: ", result["not-a-match"].text if result["not-a-match"] is not None else None)
    print(f"\nTotal tokens used: {callback._total_tokens_used}")
    print(f"Log file: {callback._log_path}")


if __name__ == "__main__":
    main()
