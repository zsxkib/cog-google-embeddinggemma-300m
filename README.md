# EmbeddingGemma-300M

[![Replicate](https://replicate.com/zsxkib/embedding-gemma-300m/badge)](https://replicate.com/zsxkib/embedding-gemma-300m) 

Text embeddings using Google's GEMMA model. 768-dimensional vectors optimized for search, classification, and retrieval.

## Usage

```python
import replicate, base64, numpy as np

# Returns a base64 string by default
b64 = replicate.run(
    "zsxkib/embedding-gemma-300m",
    input={"text": "Your text here"}
)

# Decode base64 -> float32 vector (len=768)
embedding = np.frombuffer(base64.b64decode(b64), dtype=np.float32)
print(embedding.shape)  # (768,)
```

## Parameters

- `text`: Input text to embed
- `task`: Task type (`retrieval_query`, `retrieval_document`, `classification`)  
- `output_format`: Return format (`base64` or `array`)

## Output

Returns a base64 string by default (efficient). Set `output_format: "array"` to get a list of floats.

## Local Development

```bash
cog predict -i text="Hello world"
```

## Model Details

- **Architecture**: Google GEMMA embedding model (300M parameters)
- **Dimensions**: 768  
- **Max tokens**: 2048 (auto-truncated)
- **Tasks**: Retrieval queries/documents, classification

---
Built by [@zsakib_](https://twitter.com/zsakib_) • [GitHub](https://github.com/zsxkib)
