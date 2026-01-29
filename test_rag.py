# test_rag.py — runnable test wrapper (prints output when run as script)
from query_data import query_rag
import re
import string
import sys
import traceback

# Try to import Ollama, but fail gracefully if it's not installed.
try:
    from langchain_community.llms.ollama import Ollama  # type: ignore
except Exception:
    Ollama = None

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response?
"""


def test_monopoly_rules():
    return query_and_validate(
        question="How many MINGOS is there in campus?",
        expected_response="1",
    )

def test_monopoly_rules2():
    return query_and_validate(
        question="Is there a University in the campus",
        expected_response="Yes",
    )

def test_monopoly_rules3():
    return query_and_validate(
        question="Where is the library?",
        expected_response=" M&M foods"
    )

def test_ticket_to_ride_rules():
    return query_and_validate(
        question="Which departement is with the AIML dept?",
        expected_response="MCA  is besides along with the BT  and BT Quad",
    )


def _normalize_text(s: str) -> str:
    """Lowercase, strip, remove punctuation and extra whitespace."""
    if s is None:
        return ""
    s = s.lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s)
    return s


def _extract_number(s: str):
    """Return first integer found in string or None."""
    if s is None:
        return None
    m = re.search(r"-?\d+", s)
    return int(m.group(0)) if m else None


def _fallback_evaluate(expected: str, actual: str) -> bool:
    """
    Deterministic fallback if Ollama is unavailable.
    - Numeric compare if expected contains number.
    - Otherwise substring / token overlap checks.
    """
    print("\n[Fallback evaluator] Running deterministic checks (no LLM).")

    expected_norm = _normalize_text(expected)
    actual_norm = _normalize_text(actual)

    expected_num = _extract_number(expected)
    actual_num = _extract_number(actual)

    if expected_num is not None:
        print(f"[Fallback evaluator] Expected number: {expected_num}, Actual number found: {actual_num}")
        return expected_num == actual_num

    if expected_norm == actual_norm:
        print(f"[Fallback evaluator] Exact normalized match: '{expected_norm}' == '{actual_norm}'")
        return True

    if expected_norm in actual_norm:
        print(f"[Fallback evaluator] Expected '{expected_norm}' found in actual response.")
        return True

    expected_tokens = set(expected_norm.split())
    actual_tokens = set(actual_norm.split())
    if expected_tokens:
        common = expected_tokens.intersection(actual_tokens)
        overlap_ratio = len(common) / len(expected_tokens)
        print(f"[Fallback evaluator] Token overlap: {len(common)}/{len(expected_tokens)} = {overlap_ratio:.2f}")
        if overlap_ratio >= 0.5:
            return True

    print("[Fallback evaluator] Deterministic checks failed.")
    return False


def query_and_validate(question: str, expected_response: str):
    # Run the RAG query
    print(f"\n[Running RAG] Question: {question}")
    response_text = query_rag(question, debug=True)  # debug=True prints retrieval + prompt
    print(f"[RAG returned] {response_text!r}")

    # Format prompt for LLM evaluation
    prompt = EVAL_PROMPT.format(expected_response=expected_response, actual_response=response_text)
    print("\n--- Evaluation Prompt ---")
    print(prompt)

    # If Ollama is available, try it; otherwise use deterministic fallback
    if Ollama is None:
        print("\n[Info] Ollama not installed or unavailable — using deterministic fallback evaluator.")
        return _fallback_evaluate(expected_response, response_text)

    # Try using Ollama to evaluate correctness (if available)
    try:
        model = Ollama(model="mistral")
        evaluation_results = model.invoke(prompt)
        # model.invoke may return AIMessage or string
        eval_str = getattr(evaluation_results, "content", None) or str(evaluation_results)
        evaluation_results_str_cleaned = eval_str.strip().lower()
        print(f"[Ollama evaluator] Raw response: {evaluation_results_str_cleaned}")

        if re.search(r"\btrue\b", evaluation_results_str_cleaned):
            print("\033[92m" + "Evaluator: true" + "\033[0m")
            return True
        if re.search(r"\bfalse\b", evaluation_results_str_cleaned):
            print("\033[91m" + "Evaluator: false" + "\033[0m")
            return False

        print("[Ollama evaluator] Unexpected reply; falling back to deterministic evaluator.")
        return _fallback_evaluate(expected_response, response_text)

    except Exception as e:
        print("\n[⚠️  Ollama evaluator failed — using deterministic fallback evaluator]")
        print(f"Error: {e}")
        return _fallback_evaluate(expected_response, response_text)


def run_all():
    tests = [
        ("test_monopoly_rules", test_monopoly_rules),
        ("test_ticket_to_ride_rules", test_ticket_to_ride_rules),
        ("test_monopoly_rules2", test_monopoly_rules2),
        ("test_monopoly_rules3", test_monopoly_rules3),
        
    ]
    all_ok = True
    for name, fn in tests:
        print(f"\n=== Running {name} ===")
        try:
            ok = fn()
            print(f"Result: {ok}")
            if not ok:
                all_ok = False
        except Exception as exc:
            all_ok = False
            print(f"Exception while running {name}: {exc}")
            traceback.print_exc()

    if not all_ok:
        print("\nSOME TESTS FAILED")
        sys.exit(1)
    else:
        print("\nALL TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    run_all()
