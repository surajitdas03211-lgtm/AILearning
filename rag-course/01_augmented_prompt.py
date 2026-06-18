"""Module 1 · Exercise 01 — Augmented prompts (the 'G' before the 'R').

The DL.AI course Module 1 teaches "LLM Calls and Crafting Simple Augmented
Prompts" before it teaches retrieval. Do the same here: take a KNOWN fleet
document, inject it into the prompt as grounding, and get an answer that comes
from the document — not from the model's training memory.

No retrieval yet. You hardcode which doc to use. In Module 2 you'll *retrieve*
the right doc automatically; today you isolate the generation half so the only
variable is your prompt.

Corpus:  ../data/knowledge_base/  (real Siphyy fleet docs, copied from siphyy-core)
Prompt rules (from ../learning.md): Persona -> Context -> Instruction.

Run:    python 01_augmented_prompt.py
Grade:  paste your finished version back for review against the usual rubric.
"""

from __future__ import annotations

from pathlib import Path

KB_DIR = Path(__file__).resolve().parents[1] / "data" / "knowledge_base"
DEFAULT_MODEL = "gpt-5"


def load_doc(name: str) -> str:
    """Load a knowledge-base doc by filename. (Scaffolding — not the exercise.)

    >>> load_doc("brake_failure_emergency.md").startswith("# Brake Failure")
    True
    """
    return (KB_DIR / name).read_text(encoding="utf-8")


def build_augmented_prompt(question: str, context: str) -> str:
    """Build a grounded prompt: persona + injected context + strict instruction.

    The whole point of RAG's *augmentation* step: the model must answer FROM the
    context and admit when the context doesn't contain the answer — otherwise it
    falls back on training memory and you get confident, wrong fleet advice.

    Example shape (yours can differ):
        You are a fleet operations assistant. Answer ONLY from the context below.
        If the answer is not in the context, say "Not covered in the procedure."

        Context:
        <the document>

        Question: <the question>

    TODO: return a single prompt string that
      1. sets a fleet-assistant persona,
      2. injects `context`,
      3. instructs the model to use ONLY the context and to say so when the
         answer is absent.

    Reminder: grounding is an *instruction*, not a hope. If you don't forbid
    outside knowledge, you don't have RAG — you have a chatbot sitting next to a
    document.
    """
    return f"""\
You are a fleet operations assistant helping dispatchers and drivers make safe,
procedure-grounded decisions.

Context:
{context}

Instruction:
Answer the question using ONLY the context above.
If the context does not contain the answer, say "Not covered in the procedure."
Be concise and include the specific first action when the procedure gives one.

Question:
{question}
"""


def answer_from_doc(question: str, doc_name: str) -> str:
    """Load a doc, build the augmented prompt, call the LLM, return the answer.

    TODO:
      1. context = load_doc(doc_name)
      2. prompt  = build_augmented_prompt(question, context)
      3. call an LLM with `prompt` and return its text.

    Reminder: reuse the same client pattern as siphyy-core's
    fleet_knowledge_server.py, which uses `from openai import OpenAI`. Keep the
    call thin; the learning is the prompt, not the SDK. If the API key is
    missing, fail loudly — never swallow the error.
    """
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")

    context = load_doc(doc_name)
    prompt = build_augmented_prompt(question, context)
    client = OpenAI()
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.choices[0].message.content
    if answer is None:
        raise RuntimeError("OpenAI returned an empty answer.")
    return answer.strip()


if __name__ == "__main__":
    # A dispatcher's question whose answer lives in exactly one doc.
    question = (
        "If the brake pedal goes to the floor with no resistance, "
        "what is the first thing the driver should do?"
    )
    # A correctly grounded answer should be ~ "pump the pedal rapidly 4-6 times".
    print(answer_from_doc(question, "brake_failure_emergency.md"))
