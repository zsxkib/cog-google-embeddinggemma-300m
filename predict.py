"""
EmbeddingGemma-300M Model Predictor for Replicate
Optimized for fast loading with CDN-based model weights
"""

import os
import subprocess
import time
import warnings
from typing import Any, Dict, Union, List
import torch
import numpy as np
import base64
from sentence_transformers import SentenceTransformer
from cog import BasePredictor, Input

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_CACHE = "model_cache"
BASE_URL = "https://weights.replicate.delivery/default/embedding-gemma-300m/model_cache/"

# Set environment variables for model caching - needs to happen early
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE

# Output format mapping functions
def map_to_base64(ndarray2d):
    """Convert numpy arrays to base64 encoded strings"""
    return [base64.b64encode(x.astype(np.float32).tobytes()).decode('utf-8') for x in ndarray2d]

def map_to_list(ndarray2d):
    """Convert numpy arrays to regular Python lists"""
    return ndarray2d.tolist()

# Available output formats
FORMATS = [
    ("array", map_to_list),
    ("base64", map_to_base64),
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
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.eval()
        
        # Model constants
        self.MAX_TOKENS = getattr(self.model.tokenizer, 'model_max_length', 2048)

    def predict(
        self,
        text: str = Input(
            description="Input text to encode (up to 2048 tokens)",
            
        ),
        task: str = Input(
            description="Task type for encoding (affects the prompt used for embedding)",
            default="retrieval_document",
            choices=[
                "retrieval_query", "retrieval_document", "question_answering",
                "fact_verification", "classification", "clustering", 
                "semantic_similarity", "code_retrieval"
            ]
        ),
        output_format: str = Input(
            description="Format for the embedding output",
            default="array",
            choices=[k for k, _ in FORMATS]
        ),
        normalize: bool = Input(
            description="Whether to normalize embeddings to unit length",
            default=True
        )
    ) -> Union[Dict[str, Any], List[Union[List[float], str]]]:
        """Generate embeddings for the input text"""
        start_time = time.time()
        
        # Input validation
        if not text or not text.strip():
            raise ValueError("Text input cannot be empty")
            
        if len(text.split()) > self.MAX_TOKENS:
            raise ValueError(f"Text is too long. Maximum {self.MAX_TOKENS} tokens allowed")
        
        try:
            # Create task-specific prompt following embedding best practices
            task_prompts = {
                "retrieval_query": f"Represent this query for retrieving relevant documents: {text}",
                "retrieval_document": f"Represent this document for retrieval: {text}",
                "question_answering": f"Represent this question for finding answers: {text}",
                "fact_verification": f"Represent this statement for fact verification: {text}",
                "classification": f"Represent this text for classification: {text}",
                "clustering": f"Represent this text for clustering: {text}",
                "semantic_similarity": f"Represent this text for semantic similarity: {text}",
                "code_retrieval": f"Represent this code for retrieval: {text}"
            }
            
            prompt = task_prompts.get(task, text)
            
            # Generate embeddings
            with torch.no_grad():
                embeddings = self.model.encode(
                    [prompt],
                    convert_to_tensor=True,
                    device=self.model.device,
                    show_progress_bar=False,
                    normalize_embeddings=normalize
                )
            
            # Convert to numpy
            embeddings_np = embeddings.cpu().numpy()
            
            # Format output using the mapping functions
            map_func = FORMATS_MAP[output_format]
            formatted_embeddings = map_func(embeddings_np)
            
            processing_time = time.time() - start_time
            
            # Return single embedding for single input (not wrapped in array)
            result_embedding = formatted_embeddings[0]
            
            return {
                "embedding": result_embedding,
                "shape": list(embeddings_np[0].shape),
                "task": task,
                "output_format": output_format,
                "normalized": normalize,
                "processing_time_ms": round(processing_time * 1000, 2),
                "model_info": {
                    "name": "google/embedding-gemma-300m",
                    "dimensions": embeddings_np.shape[1],
                    "max_sequence_length": self.MAX_TOKENS,
                    "device": str(self.model.device)
                }
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                "error": f"Failed to generate embedding: {str(e)}",
                "task": task,
                "processing_time_ms": round(processing_time * 1000, 2)
            }
