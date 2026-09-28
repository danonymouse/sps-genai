# sps-genai — Applied Generative AI

FastAPI service with two models:

- **Bigram text generation** (Module 3 class activity) — `POST /generate`
- **Word embeddings with spaCy** (Assignment 1) — `POST /embedding`
  Returns the 300-dimensional `en_core_web_lg` vector for a word.

## Run with Docker

```bash
docker build -t sps-genai .
docker run -p 8000:80 sps-genai
```

Then open http://127.0.0.1:8000/docs

## Run locally (without Docker)

```bash
uv sync
uv run fastapi dev app/main.py
```

## Example requests

```bash
# Word embedding
curl -X POST http://127.0.0.1:8000/embedding \
  -H "Content-Type: application/json" \
  -d '{"word": "apple"}'

# Text generation
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"start_word": "the", "length": 10}'
```

`/embedding` returns `404` if the word is not in the model's vocabulary,
and `422` if the word is empty.