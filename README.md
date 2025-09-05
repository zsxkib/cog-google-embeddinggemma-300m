# EmbeddingGemma-300M for Replicate

[![Replicate](https://replicate.com/zsxkib/cog-google-embeddinggemma-300m/badge)](https://replicate.com/zsxkib/cog-google-embeddinggemma-300m)

Run Google's EmbeddingGemma-300M model to turn text into embeddings. The model weights download automatically from CDN, so you don't have to worry about setup.

## What it does

- **Turn text into numbers**: Google's EmbeddingGemma-300M model converts text into 768-dimensional vectors
- **Fast downloads**: Model weights cache automatically from CDN on first run
- **Multiple formats**: Get embeddings as JSON arrays or base64 strings
- **Task-aware**: Different prompts for search, Q&A, classification, and more
- **GPU fast**: Runs on CUDA for quick inference

## Quick Start

You need [Docker](https://docs.docker.com/get-docker/) and [Cog](https://github.com/replicate/cog) installed.

```bash
# Install Cog
sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)
sudo chmod +x /usr/local/bin/cog

# Clone and run
git clone https://github.com/zsxkib/cog-google-embeddinggemma-300m
cd cog-google-embeddinggemma-300m
cog predict -i text="Hello, world!"
```

The first run takes a few extra minutes to download the 2.4GB model weights. After that, it's cached and fast.

## How to use it

| Parameter | Type | Default | What it does |
|-----------|------|---------|-------------|
| `text` | string | required | Your input text (up to 2048 tokens) |
| `task` | string | `"retrieval_document"` | What you're using the embedding for |
| `output_format` | string | `"array"` | How you want the embedding returned |
| `normalize` | boolean | `true` | Whether to normalize to unit length |

### Task types

Pick the right task for better embeddings:

- `retrieval_query`: When you're searching for something
- `retrieval_document`: When you're processing documents to search through
- `question_answering`: For questions that need answers
- `fact_verification`: For checking if statements are true
- `classification`: For sorting text into categories
- `clustering`: For grouping similar texts
- `semantic_similarity`: For finding how similar texts are
- `code_retrieval`: For searching code

### Output formats

- `array`: Regular JSON list of numbers
- `base64`: Compressed binary format (smaller for storage)

## Examples

Basic embedding:
```bash
cog predict -i text="Machine learning helps computers learn from data"
```

For search (document and query):
```bash
# Process a document
cog predict -i text="Python is a programming language" -i task="retrieval_document"

# Process a search query  
cog predict -i text="What is Python?" -i task="retrieval_query"
```

Base64 format:
```bash
cog predict -i text="Hello world" -i output_format="base64"
```

## What you get back

```json
{
  "embedding": [0.1, -0.3, 0.7, ...],  // 768 numbers
  "shape": [768],
  "task": "retrieval_document", 
  "output_format": "array",
  "normalized": true,
  "processing_time_ms": 340.36,
  "model_info": {
    "name": "google/embedding-gemma-300m",
    "dimensions": 768,
    "max_sequence_length": 2048,
    "device": "cuda:0"
  }
}
```

## Speed and size

- Model: ~600MB compressed
- Embeddings: 768 dimensions
- Text limit: 2048 tokens (roughly 1500 words)
- Speed: 300-500ms per text on GPU
- First run: Extra 2-3 minutes for download

## How it works

Google's EmbeddingGemma-300M is a 300 million parameter model trained to create good embeddings. It understands context and can adapt its output based on what task you're doing.

The model downloads automatically using `pget` for parallel downloads and gets cached locally in HuggingFace format.

## Development

Build and test locally:

```bash
cog build
cog predict -i text="Your text here"
```

Push to Replicate:
```bash
cog push r8.im/your-username/your-model-name
```

## Files

- `predict.py`: Model loading and prediction code
- `cog.yaml`: Environment configuration  
- `requirements.txt`: Python packages needed

## License

Apache License 2.0

## Credits

- Built on Google's EmbeddingGemma-300M model
- Uses Replicate's Cog for deployment
- Model weights served from CDN for fast loading

---

⭐ Star the repo on [GitHub](https://github.com/zsxkib/cog-google-embeddinggemma-300m)!  
🐦 Follow [@zsakib_](https://twitter.com/zsakib_) on X  
💻 Check out more projects [@zsxkib](https://github.com/zsxkib) on GitHub
