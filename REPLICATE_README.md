---
name: embeddinggemma-300m
description: Turn text into embeddings with Google's EmbeddingGemma-300M model 🔍
github_url: https://github.com/zsxkib/cog-google-embeddinggemma-300m
weights_url: https://huggingface.co/google/embeddinggemma-300m
paper_url: https://arxiv.org/abs/2407.11687
license_url: https://ai.google.dev/gemma/terms
---

# EmbeddingGemma-300M

[![Replicate](https://replicate.com/zsxkib/cog-google-embeddinggemma-300m/badge)](https://replicate.com/zsxkib/cog-google-embeddinggemma-300m)

Turn any text into a 768-dimensional vector that captures its meaning. Google's EmbeddingGemma-300M understands what you're trying to do and adapts accordingly.

## What makes this model useful

**Smart task handling**: Unlike generic embedding models, this one knows the difference between search queries and documents, questions and answers. It uses different internal prompts depending on your task.

**Efficient by default**: Returns compact base64 format that's 75% smaller than JSON arrays - perfect for storage and APIs.

**Auto-truncation**: Long text gets automatically truncated to fit model limits with clear logging.

**Multilingual**: Trained on 100+ languages, so it works well beyond English.

**Fast and reliable**: Runs on GPU with automatic weight caching from CDN.

## Perfect for

- **Semantic search**: Find documents by meaning, not just keywords
- **Recommendation systems**: Match similar content based on understanding
- **Content classification**: Sort documents, emails, or support tickets  
- **Duplicate detection**: Find similar content in large collections
- **Code search**: Search codebases by what functions do, not variable names

## How to use it

### Basic text embedding (base64 default)

```python
output = replicate.run(
    "zsxkib/cog-google-embeddinggemma-300m",
    input={"text": "Machine learning helps computers understand data"}
)
# Returns compact base64 string
```

### Array format for direct use

```python
output = replicate.run(
    "zsxkib/cog-google-embeddinggemma-300m",
    input={
        "text": "Machine learning helps computers understand data",
        "output_format": "array"
    }
)
# Returns [0.1, -0.3, 0.7, ...]
```

### Search setup (query + documents)

```python
# Process your documents
doc_embedding = replicate.run(
    "zsxkib/cog-google-embeddinggemma-300m",
    input={
        "text": "Python is a programming language used for web development, data science, and automation",
        "task": "retrieval_document"
    }
)

# Process search queries
query_embedding = replicate.run(
    "zsxkib/cog-google-embeddinggemma-300m", 
    input={
        "text": "What programming languages are good for data science?",
        "task": "retrieval_query"
    }
)
```

## Input parameters

- **text** (string): Your input text (auto-truncated to 2048 tokens)
- **task** (string): What you're using the embedding for - `retrieval_document`, `retrieval_query`, `classification`, `clustering`, `semantic_similarity`, `question_answering`, `fact_verification`, or `code_retrieval`
- **output_format** (string): `base64` for compact storage (default) or `array` for direct use
- **normalize** (boolean): Whether to normalize to unit length (default: true)

## Output formats

**Base64 (default)**:
```json
{
  "embedding": "pVK9vQAAgDwAAIA8...",
  "output_format": "base64",
  "shape": [768],
  "processing_time_ms": 45.2
}
```

**Array**:
```json
{
  "embedding": [0.1, -0.3, 0.7, ...],
  "output_format": "array", 
  "shape": [768],
  "processing_time_ms": 45.2
}
```

## Working with base64 embeddings

Base64 is more efficient but needs decoding:

**Python**:
```python
import base64
import numpy as np

# Decode base64 to numpy array
embedding_bytes = base64.b64decode(output["embedding"])
embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
print(embedding.shape)  # (768,)
```

**JavaScript**:
```javascript
function decodeBase64Embedding(base64String) {
    const bytes = Uint8Array.from(atob(base64String), c => c.charCodeAt(0));
    return new Float32Array(bytes.buffer);
}
```

## Tips

**Match your tasks**: Use `retrieval_query` for search terms, `retrieval_document` for the content you're searching through. The model gives better results when it knows what you're doing.

**Choose your format**: Use `base64` for storage/APIs (75% smaller), `array` for direct computation.

**Auto-truncation**: Text over 2048 tokens gets truncated automatically - check the logs.

**Consistent processing**: Use the same task type and format for texts you'll compare later.

## Technical details

- 300 million parameter model from Google
- Based on the Gemma architecture 
- Trained with task-aware prompting
- Supports 100+ languages
- 2048 token limit per text
- Sub-second processing on GPU

Learn more in Google's paper: [EmbeddingGemma: Improved Text Embeddings](https://arxiv.org/abs/2407.11687)

## License

This model follows Google's Gemma license terms, which allows commercial use with attribution. See the full [Gemma Terms of Use](https://ai.google.dev/gemma/terms).

---

⭐ Star the repo on [GitHub](https://github.com/zsxkib/cog-google-embeddinggemma-300m)!  
🐦 Follow [@zsakib_](https://twitter.com/zsakib_) on X  
💻 Check out more projects [@zsxkib](https://github.com/zsxkib) on GitHub
