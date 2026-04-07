#!/usr/bin/env python3
"""
Automatic Concept Vector Extraction

This module provides automatic extraction of steering vectors from concept names.
It uses an LLM to generate contrastive prompts, then extracts multi-layer vectors.
"""

import json
import asyncio
import httpx
import numpy as np
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid


@dataclass
class ConceptMetadata:
    """Metadata for a concept steering vector."""
    id: str
    name: str
    description: str
    positive_prompts: list[str]
    negative_prompts: list[str]
    layers: list[int]
    vector_norms: dict[int, float]
    recommended_strength: float
    created_at: str

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ConceptMetadata":
        return cls(**d)


@dataclass
class ExtractionProgress:
    """Progress information during extraction."""
    status: str  # "generating_prompts", "dumping_activations", "computing_vectors", "creating_gguf", "complete", "failed"
    progress: float  # 0.0 to 1.0
    current_step: str
    error: Optional[str] = None


class PromptGenerator:
    """Generate contrastive prompts for concept extraction using LLM."""

    SYSTEM_PROMPT = """You are helping create training data for activation steering in language models.

Given a concept, generate two sets of prompts:
1. POSITIVE prompts: Short sentences (1-2 sentences) that strongly embody or demonstrate the concept
2. NEGATIVE prompts: Neutral sentences that do NOT embody the concept (not opposites, just unrelated)

Rules:
- Each prompt should be 1-2 sentences
- Positive prompts should clearly and naturally demonstrate the concept
- Negative prompts should be neutral everyday sentences (weather, math, greetings, etc.)
- Vary the topics and styles within each set
- Output valid JSON only

Example for concept "formal speech":
{
  "positive": [
    "I would like to express my sincere gratitude for your assistance.",
    "It is with great pleasure that I announce the following decision.",
    "We respectfully request your consideration of this matter."
  ],
  "negative": [
    "The weather is nice today.",
    "I had pasta for lunch.",
    "The cat is sleeping on the couch."
  ]
}"""

    def __init__(self, llm_url: str = "http://localhost:8081"):
        """
        Initialize PromptGenerator.

        Args:
            llm_url: URL of the llama-server for prompt generation
        """
        self.llm_url = llm_url

    async def generate(
        self,
        concept: str,
        num_positive: int = 8,
        num_negative: int = 8,
        timeout: float = 120.0
    ) -> tuple[list[str], list[str]]:
        """
        Generate contrastive prompts for a concept.

        Args:
            concept: The concept name (e.g., "happiness", "formal speech", "Japanese language")
            num_positive: Number of positive prompts to generate
            num_negative: Number of negative prompts to generate
            timeout: Request timeout in seconds

        Returns:
            Tuple of (positive_prompts, negative_prompts)
        """
        user_prompt = f"""Generate {num_positive} positive prompts and {num_negative} negative prompts for the concept: "{concept}"

Output only valid JSON in this format:
{{"positive": [...], "negative": [...]}}"""

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{self.llm_url}/v1/chat/completions",
                json={
                    "messages": messages,
                    "max_tokens": 1000,
                    "temperature": 0.7,
                }
            )

            if response.status_code != 200:
                raise RuntimeError(f"LLM request failed: {response.status_code}")

            data = response.json()
            content = data["choices"][0]["message"]["content"]

            # Parse JSON from response
            try:
                # Try to extract JSON from the response
                json_start = content.find("{")
                json_end = content.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    result = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")

                positive = result.get("positive", [])
                negative = result.get("negative", [])

                if not positive or not negative:
                    raise ValueError("Missing positive or negative prompts")

                return positive[:num_positive], negative[:num_negative]

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                # If parsing fails, create fallback prompts
                print(f"Warning: Failed to parse LLM response, using fallback prompts: {e}")
                return self._fallback_prompts(concept, num_positive, num_negative)

    def _fallback_prompts(
        self,
        concept: str,
        num_positive: int,
        num_negative: int
    ) -> tuple[list[str], list[str]]:
        """Generate simple fallback prompts if LLM generation fails."""
        positive = [
            f"This is an example of {concept}.",
            f"I want to demonstrate {concept} in this sentence.",
            f"The following text embodies {concept}.",
            f"{concept} is shown in this example.",
            f"Here is a clear example of {concept}.",
            f"This sentence represents {concept}.",
            f"An illustration of {concept} follows.",
            f"This demonstrates the concept of {concept}.",
        ][:num_positive]

        negative = [
            "The weather today is pleasant.",
            "I had coffee this morning.",
            "The book is on the table.",
            "Two plus two equals four.",
            "The sky is blue.",
            "I went for a walk yesterday.",
            "The meeting starts at noon.",
            "Please pass the salt.",
        ][:num_negative]

        return positive, negative


class ConceptExtractor:
    """
    Extract multi-layer steering vectors from a concept name.

    This orchestrates the full extraction pipeline:
    1. Generate prompts using LLM
    2. Dump activations for all prompts
    3. Compute steering vectors per layer
    4. Validate vectors
    5. Create GGUF file
    """

    DEFAULT_LAYERS = [10, 12, 15, 18, 20, 22, 25]

    def __init__(
        self,
        model_path: str,
        llama_cpp_dir: str,
        llm_url: str = "http://localhost:8081",
        vectors_dir: str = "./vectors",
        n_gpu_layers: int = 99
    ):
        """
        Initialize ConceptExtractor.

        Args:
            model_path: Path to Bonsai-8B.gguf
            llama_cpp_dir: Path to llama.cpp build directory
            llm_url: URL of llama-server for prompt generation
            vectors_dir: Directory to save extracted vectors
            n_gpu_layers: GPU layers for inference
        """
        self.model_path = Path(model_path)
        self.llama_cpp_dir = Path(llama_cpp_dir)
        self.llm_url = llm_url
        self.vectors_dir = Path(vectors_dir)
        self.n_gpu_layers = n_gpu_layers

        self.vectors_dir.mkdir(parents=True, exist_ok=True)

        # Import BonsaiModel
        from bonsai_steering import BonsaiModel
        self.model = BonsaiModel(
            model_path=str(self.model_path),
            llama_cpp_dir=str(self.llama_cpp_dir),
            n_gpu_layers=n_gpu_layers
        )

        self.prompt_generator = PromptGenerator(llm_url=llm_url)

    async def extract(
        self,
        concept_name: str,
        layers: Optional[list[int]] = None,
        num_positive: int = 8,
        num_negative: int = 8,
        progress_callback: Optional[Callable[[ExtractionProgress], None]] = None
    ) -> tuple[dict[int, np.ndarray], ConceptMetadata]:
        """
        Extract steering vectors for a concept.

        Args:
            concept_name: Name of the concept to extract
            layers: Layers to extract vectors from (default: [10, 12, 15, 18, 20, 22, 25])
            num_positive: Number of positive prompts
            num_negative: Number of negative prompts
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (vectors_dict, metadata)
        """
        if layers is None:
            layers = self.DEFAULT_LAYERS

        def report_progress(status: str, progress: float, step: str):
            if progress_callback:
                progress_callback(ExtractionProgress(
                    status=status,
                    progress=progress,
                    current_step=step
                ))

        try:
            # Step 1: Generate prompts
            report_progress("generating_prompts", 0.1, "Generating contrastive prompts...")
            positive_prompts, negative_prompts = await self.prompt_generator.generate(
                concept_name, num_positive, num_negative
            )

            # Step 2: Dump activations
            report_progress("dumping_activations", 0.2, "Extracting activations from positive prompts...")

            # Create temp directory for activations
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                positive_acts = {layer: [] for layer in layers}
                negative_acts = {layer: [] for layer in layers}

                # Extract positive activations
                total_prompts = len(positive_prompts) + len(negative_prompts)
                for i, prompt in enumerate(positive_prompts):
                    progress = 0.2 + 0.3 * (i / total_prompts)
                    report_progress(
                        "dumping_activations",
                        progress,
                        f"Processing positive prompt {i+1}/{len(positive_prompts)}..."
                    )

                    for layer in layers:
                        act = self.model.get_last_token_activation(
                            prompt, layer, output_dir=tmpdir
                        )
                        positive_acts[layer].append(act)

                # Extract negative activations
                report_progress("dumping_activations", 0.5, "Extracting activations from negative prompts...")
                for i, prompt in enumerate(negative_prompts):
                    progress = 0.5 + 0.3 * (i / total_prompts)
                    report_progress(
                        "dumping_activations",
                        progress,
                        f"Processing negative prompt {i+1}/{len(negative_prompts)}..."
                    )

                    for layer in layers:
                        act = self.model.get_last_token_activation(
                            prompt, layer, output_dir=tmpdir
                        )
                        negative_acts[layer].append(act)

                # Step 3: Compute vectors
                report_progress("computing_vectors", 0.8, "Computing steering vectors...")
                vectors = {}
                vector_norms = {}

                for layer in layers:
                    pos_mean = np.mean(positive_acts[layer], axis=0)
                    neg_mean = np.mean(negative_acts[layer], axis=0)
                    steering_vec = pos_mean - neg_mean
                    vectors[layer] = steering_vec.astype(np.float32)
                    vector_norms[layer] = float(np.linalg.norm(steering_vec))

                # Step 4: Create metadata
                concept_id = f"{concept_name.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}"
                metadata = ConceptMetadata(
                    id=concept_id,
                    name=concept_name,
                    description=f"Steering vector for '{concept_name}' extracted from {len(positive_prompts)} positive and {len(negative_prompts)} negative prompts.",
                    positive_prompts=positive_prompts,
                    negative_prompts=negative_prompts,
                    layers=layers,
                    vector_norms=vector_norms,
                    recommended_strength=0.2,
                    created_at=datetime.now().isoformat()
                )

                report_progress("complete", 1.0, "Extraction complete!")
                return vectors, metadata

        except Exception as e:
            if progress_callback:
                progress_callback(ExtractionProgress(
                    status="failed",
                    progress=0.0,
                    current_step="Extraction failed",
                    error=str(e)
                ))
            raise

    def save_vectors(
        self,
        vectors: dict[int, np.ndarray],
        metadata: ConceptMetadata,
        save_numpy: bool = True
    ) -> list[Path]:
        """
        Save extracted vectors as .npy files.

        Args:
            vectors: Dictionary of layer -> vector
            metadata: Concept metadata
            save_numpy: Whether to save individual .npy files

        Returns:
            List of saved file paths
        """
        saved_paths = []
        concept_slug = metadata.id

        if save_numpy:
            for layer, vec in vectors.items():
                npy_path = self.vectors_dir / f"{concept_slug}_layer{layer}.npy"
                np.save(npy_path, vec)
                saved_paths.append(npy_path)

        # Save metadata
        meta_path = self.vectors_dir / f"{concept_slug}_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        saved_paths.append(meta_path)

        return saved_paths

    def create_gguf(
        self,
        vectors: dict[int, np.ndarray],
        metadata: ConceptMetadata
    ) -> Path:
        """
        Create GGUF control vector file.

        Args:
            vectors: Dictionary of layer -> vector
            metadata: Concept metadata

        Returns:
            Path to created GGUF file
        """
        # Import GGUF creation function
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from convert_to_gguf import create_control_vector_gguf

        output_path = self.vectors_dir / f"{metadata.id}.gguf"
        create_control_vector_gguf(vectors, output_path)

        return output_path


async def demo():
    """Demo showing automatic concept extraction."""
    import sys

    # Configuration
    base_dir = Path(__file__).parent.parent
    llama_cpp_dir = base_dir / "llama.cpp" / "build"
    model_path = base_dir / "llama.cpp" / "models" / "Bonsai-8B.gguf"
    vectors_dir = base_dir / "vectors"

    print("=" * 60)
    print("Automatic Concept Extraction Demo")
    print("=" * 60)

    def progress_callback(progress: ExtractionProgress):
        print(f"[{progress.progress*100:.0f}%] {progress.current_step}")

    extractor = ConceptExtractor(
        model_path=str(model_path),
        llama_cpp_dir=str(llama_cpp_dir),
        llm_url="http://localhost:8081",
        vectors_dir=str(vectors_dir)
    )

    # Extract "happiness" concept
    concept = "happiness"
    print(f"\nExtracting concept: {concept}")
    print("-" * 40)

    vectors, metadata = await extractor.extract(
        concept_name=concept,
        layers=[10, 15, 20, 25],
        num_positive=5,
        num_negative=5,
        progress_callback=progress_callback
    )

    print(f"\nExtraction Results:")
    print(f"  Concept: {metadata.name}")
    print(f"  Layers: {metadata.layers}")
    print(f"  Vector norms: {metadata.vector_norms}")
    print(f"  Positive prompts: {len(metadata.positive_prompts)}")
    print(f"  Negative prompts: {len(metadata.negative_prompts)}")

    # Save vectors
    print("\nSaving vectors...")
    saved = extractor.save_vectors(vectors, metadata)
    for p in saved:
        print(f"  Saved: {p}")

    # Create GGUF
    print("\nCreating GGUF...")
    gguf_path = extractor.create_gguf(vectors, metadata)
    print(f"  GGUF: {gguf_path}")

    print("\n" + "=" * 60)
    print("Demo complete!")


if __name__ == "__main__":
    asyncio.run(demo())
