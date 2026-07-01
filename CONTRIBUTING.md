# Contributing

Thanks for your interest in the RAG Compliance Engine.

## Development setup

```bash
cd src
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg   # Presidio model, ~560 MB, one-time
pytest -q
```

## Ground rules

- **Tests pass before you push.** `pytest -q` must be green. New behavior needs a test.
- **Don't weaken the security guarantee.** Access control is enforced in the data layer
  (vector store / lexical filter), never in the prompt. Any change to retrieval must keep
  both arms filtering to the caller's groups and tenant before fusion.
- **Fail closed.** If an answer can't be audited or output-redacted, the request should
  fail (500) rather than return an unverified result.
- **Keep the provider seams clean.** New backends implement the `VectorStore` /
  `LLMProvider` interfaces; the pipeline (`retrieve`, `ingest`, `answer_query`) stays
  backend-agnostic.
- Match the surrounding style; keep changes focused and diffs small.

## Pull requests

1. Branch from `main`.
2. Make the change with tests.
3. Run `pytest -q` and note the result in the PR description.
4. Open the PR against `main` with a short description of what and why.
