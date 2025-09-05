---
name: embedding-gemma-300m
description: Turn any text into 768-dimensional vectors for search, classification, and AI apps 🧠✨ 
github_url: https://github.com/zsxkib/cog-google-embeddinggemma-300m
weights_url: https://huggingface.co/google/embedding-gemma-300m
paper_url: https://arxiv.org/abs/2408.10957
license_url: https://www.kaggle.com/models/google/embedding-gemma/license
---

# EmbeddingGemma-300M 🧠✨

## Overview 🔊
**EmbeddingGemma-300M** is a text-to-vector model that transforms any text into 768-dimensional embeddings. This tool is built upon the amazing work of [Google DeepMind](https://deepmind.google/) and their [EmbeddingGemma research](https://arxiv.org/abs/2408.10957). We've wrapped their [embedding-gemma-300m model](https://huggingface.co/google/embedding-gemma-300m) to work on Replicate! Allowing us to generate high-quality embeddings for search, recommendations, and AI applications!

Support Google DeepMind and learn more about their work through:

- [EmbeddingGemma Paper](https://arxiv.org/abs/2408.10957)
- [Model Card on Hugging Face](https://huggingface.co/google/embedding-gemma-300m)

## Pre-loaded Efficiency 🚀
The EmbeddingGemma-300M comes pre-loaded with **base64 output format** for immediate use with 3x smaller response sizes compared to arrays.

## Getting Different Dimensions 💥
You can customize your EmbeddingGemma-300M's output using Matryoshka representation learning. Here's how:

1. Choose your embedding dimension: 128, 256, 512, or 768

2. Set the `embedding_dim` parameter to your desired size

3. The model automatically truncates the full 768-dimensional embedding to your chosen size

4. Example usage:
   - `embedding_dim = 256` → Get 256-dimensional vectors
   - `embedding_dim = 512` → Get 512-dimensional vectors  
   - `embedding_dim = 768` → Get full 768-dimensional vectors (default)

*Note: Smaller dimensions are perfect for storage optimization while maintaining strong performance for most tasks.*

## Code Examples 📚

**Python:**
```python
import replicate, base64, numpy as np

# Get base64 embedding (default)
b64 = replicate.run("zsxkib/embedding-gemma-300m", input={"text": "Hello world"})

# Decode to numpy array
vec = np.frombuffer(base64.b64decode(b64), dtype=np.float32)
print(vec.shape)  # (768,)
```

**JavaScript:**
```javascript
const b64 = await replicate.run("zsxkib/embedding-gemma-300m", { input: { text: "Hello world" } });
const embedding = new Float32Array(Buffer.from(b64, 'base64').buffer);
console.log(embedding.length);  // 768
```

## Terms of Use 📚

The use of this embedding model for the following purposes is prohibited:

* Generating embeddings for harmful or malicious content.

* Creating systems designed to manipulate or deceive users.

* Building applications that violate user privacy or data protection laws.

* Commercial use without proper licensing from Google.

* Redistributing the model weights without permission.

* Using embeddings to create biased or discriminatory systems.

## Disclaimer ‼️

I am not liable for any direct, indirect, consequential, incidental, or special damages arising out of or in any way connected with the use/misuse or inability to use this software.

---

⭐ Star the repo on [GitHub](https://github.com/zsxkib/cog-google-embeddinggemma-300m)!  
🐦 Follow [@zsakib_](https://twitter.com/zsakib_) on X  
💻 Check out more projects [@zsxkib](https://github.com/zsxkib) on GitHub
