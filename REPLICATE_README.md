# Text Embeddings with GEMMA

Convert any text into a 768‑dimensional vector. Simple, fast, and compact by default.

## What you get
- 768‑dimensional embeddings (128/256/512/768 via Matryoshka) (Google GEMMA)
- Fast inference (~50ms on GPU)
- Base64 output by default (compact). Use `array` for floats.
- Auto‑truncation for long text (no errors)

## Basic usage
Paste your text in the UI and run.

## Advanced options
- Task: `retrieval_query`, `retrieval_document`, `classification`, etc.
- Output format: `base64` (default) or `array`

## Code examples

Python:
```python
import replicate, base64, numpy as np

# Returns a base64 string by default
b64 = replicate.run("zsxkib/embedding-gemma-300m", input={"text": "Hello world"})

# Decode to float32 vector (len=768)
vec = np.frombuffer(base64.b64decode(b64), dtype=np.float32)
```

JavaScript:
```javascript
const b64 = await replicate.run("zsxkib/embedding-gemma-300m", { input: { text: "Hello world" } });
const embedding = new Float32Array(Buffer.from(b64, 'base64').buffer); // length 768
```

— Built by [@zsakib_](https://twitter.com/zsakib_) • [GitHub](https://github.com/zsxkib)
