"""
EmbeddingGemma-300M predictor (clean + simple)

Design goals:
- No unnecessary metadata in outputs
- No try/except in predict() – let errors surface
- Return a single embedding: base64 string (default) or list of floats
"""

import os
import subprocess
import time
import warnings
from typing import Union, List

import base64
import numpy as np
import torch
from cog import BasePredictor, Input
from sentence_transformers import SentenceTransformer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_CACHE = "model_cache"
BASE_URL = "https://weights.replicate.delivery/default/embedding-gemma-300m/model_cache/"

# Set environment variables for model caching - needs to happen early
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE


def map_to_base64(ndarray2d):
    """Convert numpy arrays to base64 encoded strings"""
    return [base64.b64encode(x.astype(np.float32).tobytes()).decode("utf-8") for x in ndarray2d]


def map_to_list(ndarray2d):
    """Convert numpy arrays to regular Python lists"""
    return ndarray2d.tolist()


# Available output formats
FORMATS = [
    ("base64", map_to_base64),
    ("array", map_to_list),
]
FORMATS_MAP = dict(FORMATS)


def download_weights(url: str, dest: str) -> None:
    """Download model weights using pget with automatic .tar extraction"""
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Create model cache directory if it doesn't exist
        os.makedirs(MODEL_CACHE, exist_ok=True)

        model_files = [
            ".locks.tar",
            "models--google--embeddinggemma-300m.tar",
            "version.txt",
            "xet.tar",
        ]

        for model_file in model_files:
            url = BASE_URL + model_file
            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)

        # Load the model from cached location
        model_path = f"{MODEL_CACHE}/models--google--embeddinggemma-300m/snapshots/64614b0b8b64f0c6c1e52b07e4e9a4e8fe4d2da2"
        self.model = SentenceTransformer(
            model_path,
            trust_remote_code=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.model.eval()

        # Model constants
        self.MAX_TOKENS = getattr(self.model.tokenizer, "model_max_length", 2048)

    def predict(
        self,
        text: str = Input(
            description="Text to embed. Automatically truncated to 2048 tokens if longer.",
        ),
        task: str = Input(
            description="Task type - affects the internal prompt used for embedding",
            default="retrieval_document",
            choices=[
                "retrieval_query",     # "Represent this query for retrieving relevant documents"
                "retrieval_document",  # "Represent this document for retrieval" 
                "classification",      # "Represent this text for classification"
                "clustering",         # "Represent this text for clustering"
                "question_answering", # "Represent this question for finding answers"
                "fact_verification", # "Represent this statement for fact verification"
                "semantic_similarity", # "Represent this text for semantic similarity"
                "code_retrieval",     # "Represent this code for retrieval"
            ],
        ),
        embedding_dim: int = Input(
            description="Output embedding dimensions using Matryoshka truncation",
            default=768,
            choices=[128, 256, 512, 768],
        ),
        output_format: str = Input(
            description="Output format: base64 (efficient) or array (list of floats)",
            default="base64",
            choices=["base64", "array"],
        ),
        normalize: bool = Input(
            description="Whether to L2-normalize the embedding to unit length",
            default=True,
        ),
    ) -> Union[str, List[float]]:
        """Generate a single embedding for the given text.

        Returns:
            - base64 string when output_format == "base64" (default)
            - list[float] when output_format == "array"
        """

        if not text or not text.strip():
            raise ValueError("Text input cannot be empty")

        # Simple word-based truncation to avoid errors on long inputs
        words = text.split()
        if len(words) > self.MAX_TOKENS:
            text = " ".join(words[: self.MAX_TOKENS])

        # Task-specific prompt (minimal but effective)
        task_prompts = {
            "retrieval_query": f"Represent this query for retrieving relevant documents: {text}",
            "retrieval_document": f"Represent this document for retrieval: {text}",
            "question_answering": f"Represent this question for finding answers: {text}",
            "fact_verification": f"Represent this statement for fact verification: {text}",
            "classification": f"Represent this text for classification: {text}",
            "clustering": f"Represent this text for clustering: {text}",
            "semantic_similarity": f"Represent this text for semantic similarity: {text}",
            "code_retrieval": f"Represent this code for retrieval: {text}",
        }
        prompt = task_prompts.get(task, text)

        # Encode
        with torch.no_grad():
            embeddings = self.model.encode(
                [prompt],
                convert_to_tensor=True,
                device=self.model.device,
                show_progress_bar=False,
                normalize_embeddings=normalize,
            )

        # To numpy and potentially truncate using Matryoshka
        embeddings_np = embeddings.cpu().numpy()
        
        # Matryoshka truncation if requested
        if embedding_dim < 768:
            embeddings_np = embeddings_np[:, :embedding_dim]
        
        # Format output
        map_func = FORMATS_MAP[output_format]
        return map_func(embeddings_np)[0]
