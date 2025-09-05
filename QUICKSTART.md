# 🚀 Quick Start Guide

Get embeddings from your text in seconds with the EmbeddingGemma-300M model.

## Python

```python
import replicate
import base64
import numpy as np

# Get embedding (base64 by default - more efficient)
output = replicate.run(
    "zsxkib/embedding-gemma-300m",
    input={"text": "Your text here"}
)

# Decode base64 to numpy array
embedding_bytes = base64.b64decode(output['embedding'])
embedding = np.frombuffer(embedding_bytes, dtype=np.float32)

print(f"Embedding shape: {embedding.shape}")  # (768,)
print(f"Task: {output['task']}")               # retrieval_document
```

## JavaScript/Node.js

```javascript
const output = await replicate.run(
    "zsxkib/embedding-gemma-300m",
    { input: { text: "Your text here" } }
);

// Decode base64 to Float32Array
const buffer = Buffer.from(output.embedding, 'base64');
const embedding = new Float32Array(buffer.buffer);

console.log(`Embedding length: ${embedding.length}`);  // 768
console.log(`Task: ${output.task}`);                    // retrieval_document
```

## cURL

```bash
curl -s -X POST \
  -H "Authorization: Bearer $REPLICATE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"input": {"text": "Your text here"}}' \
  https://api.replicate.com/v1/models/zsxkib/embedding-gemma-300m/predictions
```

## Advanced Usage

### Different Tasks
```python
# For search queries
output = replicate.run(
    "zsxkib/embedding-gemma-300m",
    input={
        "text": "What is machine learning?",
        "task": "retrieval_query"
    }
)

# For classification
output = replicate.run(
    "zsxkib/embedding-gemma-300m",
    input={
        "text": "This movie is amazing!",
        "task": "classification"
    }
)
```

### Array Format (Less Efficient)
```python
# If you need arrays directly (larger response size)
output = replicate.run(
    "zsxkib/embedding-gemma-300m",
    input={
        "text": "Your text here",
        "output_format": "array"
    }
)

embedding = output['embedding']  # Already a list of floats
```

## Key Points

- **Default format is base64** for efficiency (4KB vs 12KB)
- **768-dimensional embeddings** from Google's GEMMA model
- **Multiple tasks supported**: retrieval_query, retrieval_document, classification
- **Auto-truncation** at 2048 tokens (no errors)
- **Fast inference** (~50ms on T4 GPU)

That's it! You're ready to generate embeddings! 🎉
