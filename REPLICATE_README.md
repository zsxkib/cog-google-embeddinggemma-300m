# Text Embeddings with GEMMA

Convert any text into 768-dimensional vectors perfect for search, recommendations, and AI applications.

## What you get

- **768-dimensional embeddings** from Google's GEMMA model
- **Fast inference** (~50ms on GPU)  
- **Smart format**: Base64 by default (3x smaller than arrays)
- **Auto-truncation** for long texts (no errors)

## Basic usage

Just paste your text and hit run. That's it.

## Advanced options

**Task types:**
- `retrieval_query` - For search queries  
- `retrieval_document` - For documents to search
- `classification` - For categorization tasks

**Output formats:**
- `base64` (default) - Compact, decode with base64 libraries
- `array` - Direct list of numbers (larger response)

## Code examples

**Python:**
```python
import replicate, base64, numpy as np

output = replicate.run("zsxkib/embedding-gemma-300m", 
    input={"text": "Hello world"})

# Decode base64 to numpy
embedding = np.frombuffer(base64.b64decode(output['embedding']), dtype=np.float32)
```

**JavaScript:**
```javascript
const output = await replicate.run("zsxkib/embedding-gemma-300m", 
    {input: {text: "Hello world"}});

// Decode base64 to Float32Array  
const embedding = new Float32Array(Buffer.from(output.embedding, 'base64').buffer);
```

Perfect for building semantic search, recommendation systems, or any app that needs to understand text similarity.

---
Made by [@zsakib_](https://twitter.com/zsakib_)
