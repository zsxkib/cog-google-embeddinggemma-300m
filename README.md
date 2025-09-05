# EmbeddingGemma-300M

[![Replicate](https://replicate.com/zsxkib/embedding-gemma-300m/badge)](https://replicate.com/zsxkib/embedding-gemma-300m) 

Text embeddings using Google's GEMMA model. 768-dimensional vectors optimized for search, classification, and retrieval.

## Usage

```python
import replicate
import base64
import numpy as np

output = replicate.run(
    "zsxkib/embedding-gemma-300m",
    input={"text": "Your text here"}
)

# Decode base64 to numpy array
embedding = np.frombuffer(base64.b64decode(output['embedding']), dtype=np.float32)
print(embedding.shape)  # (768,)
```

## Parameters

- `text`: Input text to embed
- `task`: Task type (`retrieval_query`, `retrieval_document`, `classification`)  
- `output_format`: Return format (`base64` or `array`)

## Output

Returns base64-encoded embeddings by default for efficiency. Use `output_format: "array"` for direct lists.

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
