# AILearning

Small AI learning repo for practicing retrieval-augmented generation.

## Current exercise

`rag-course/01_augmented_prompt.py` demonstrates the generation step of RAG:

1. Load one known fleet procedure document.
2. Inject that document into an augmented prompt.
3. Ask the model to answer only from the provided context.

The included document is:

`data/knowledge_base/brake_failure_emergency.md`

## Run

```bash
pip install openai python-dotenv
export OPENAI_API_KEY="your-key"
python rag-course/01_augmented_prompt.py
```
