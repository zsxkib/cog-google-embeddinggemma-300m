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

**Multiple output formats**: Get embeddings as JSON arrays or compact base64 strings.

**Multilingual**: Trained on 100+ languages, so it works well beyond English.

**Fast and reliable**: Runs on GPU with automatic weight caching from CDN.

## Perfect for

- **Semantic search**: Find documents by meaning, not just keywords
- **Recommendation systems**: Match similar content based on understanding
- **Content classification**: Sort documents, emails, or support tickets  
- **Duplicate detection**: Find similar content in large collections
- **Code search**: Search codebases by what functions do, not variable names

## How to use it

### Basic text embedding

```python
output = replicate.run(
    "zsxkib/cog-google-embeddinggemma-300m",
    input={"text": "Machine learning helps computers understand data"}
)
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

### Compact storage format

```python
output = replicate.run(
    "zsxkib/cog-google-embeddinggemma-300m",
    input={
        "text": "Your text here",
        "output_format": "base64"
    }
)
```

## Input parameters

- **text** (string): Your input text (up to 2048 tokens)
- **task** (string): What you're using the embedding for - `retrieval_document`, `retrieval_query`, `classification`, `clustering`, `semantic_similarity`, `question_answering`, `fact_verification`, or `code_retrieval`
- **output_format** (string): `array` for JSON or `base64` for compact binary
- **normalize** (boolean): Whether to normalize to unit length (default: true)

## What you get

```json
{
  "embedding": [0.1, -0.3, 0.7, ...],
  "shape": [768],
  "task": "retrieval_document",
  "output_format": "array", 
  "normalized": true,
  "processing_time_ms": 342.5,
  "model_info": {
    "name": "google/embedding-gemma-300m",
    "dimensions": 768,
    "max_sequence_length": 2048,
    "device": "cuda:0"
  }
}
```

## Tips

**Match your tasks**: Use `retrieval_query` for search terms, `retrieval_document` for the content you're searching through. The model gives better results when it knows what you're doing.

**Batch when possible**: Process multiple texts in separate calls but reuse the same model instance for efficiency.

**Right-size dimensions**: All embeddings are 768 dimensions - perfect for most similarity tasks without being too large.

**Consistent processing**: Use the same task type for texts you'll compare later.

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
