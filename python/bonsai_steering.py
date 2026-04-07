#!/usr/bin/env python3
"""
Bonsai-8B Activation Steering Python Wrapper

This module provides a Python interface for:
- Dumping intermediate layer activations from Bonsai-8B (1-bit GGUF)
- Computing steering vectors from contrastive prompt pairs
- Applying steering vectors during text generation
"""

import subprocess
import json
import numpy as np
from pathlib import Path
from typing import Optional, Union
import tempfile
import os


class BonsaiModel:
    """Bonsai-8B activation dumping and steering wrapper."""

    def __init__(
        self,
        model_path: str,
        llama_cpp_dir: str,
        n_gpu_layers: int = 99
    ):
        """
        Initialize BonsaiModel.

        Args:
            model_path: Path to Bonsai-8B.gguf
            llama_cpp_dir: Path to llama.cpp build directory
            n_gpu_layers: Number of layers to offload to GPU (default: 99 = all)
        """
        self.model_path = Path(model_path)
        self.llama_cpp_dir = Path(llama_cpp_dir)
        self.n_gpu_layers = n_gpu_layers

        # Verify paths
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.dump_cli = self.llama_cpp_dir / "bin" / "llama-activation-dump"
        self.steer_cli = self.llama_cpp_dir / "bin" / "llama-activation-steering"

        if not self.dump_cli.exists():
            raise FileNotFoundError(f"Activation dump CLI not found: {self.dump_cli}")
        if not self.steer_cli.exists():
            raise FileNotFoundError(f"Steering CLI not found: {self.steer_cli}")

    def dump_activations(
        self,
        prompt: str,
        layers: list[int],
        output_dir: str = "./activations",
        dump_format: str = "numpy"
    ) -> dict[int, dict[int, np.ndarray]]:
        """
        Dump activations for specified layers.

        Args:
            prompt: Input prompt
            layers: List of layer numbers to dump (e.g., [10, 15, 20, 25])
            output_dir: Directory to save activation files
            dump_format: Output format ("numpy" or "raw")

        Returns:
            Dictionary mapping layer -> token_pos -> activation array
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        layer_str = ",".join(str(l) for l in layers)

        cmd = [
            str(self.dump_cli),
            "-m", str(self.model_path),
            "-p", prompt,
            "--dump-activations", str(output_path),
            "--dump-layers", layer_str,
            "--dump-format", dump_format,
            "-ngl", str(self.n_gpu_layers)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Activation dump failed: {result.stderr}")

        # Load results
        activations = {}
        for layer in layers:
            activations[layer] = {}
            for f in output_path.glob(f"layer_{layer}_token_*.npy"):
                token_pos = int(f.stem.split("_")[-1])
                activations[layer][token_pos] = np.load(str(f))

        return activations

    def get_last_token_activation(
        self,
        prompt: str,
        layer: int,
        output_dir: str = "./activations"
    ) -> np.ndarray:
        """
        Get activation at the last token position for a specific layer.

        Args:
            prompt: Input prompt
            layer: Layer number

        Returns:
            Activation array of shape (hidden_dim,)
        """
        activations = self.dump_activations(prompt, [layer], output_dir)
        token_acts = activations[layer]
        last_pos = max(token_acts.keys())
        return token_acts[last_pos]

    def generate_with_steering(
        self,
        prompt: str,
        steer_config: dict,
        n_tokens: int = 128,
        temperature: float = 0.5
    ) -> str:
        """
        Generate text with steering vectors applied.

        Args:
            prompt: Input prompt
            steer_config: Steering configuration dict with format:
                {
                    "interventions": [
                        {"layer": 25, "vector": "path/to/vector.npy", "strength": 1.0},
                        ...
                    ],
                    "token_position": "last"
                }
            n_tokens: Number of tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        # Write config to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(steer_config, f)
            config_path = f.name

        try:
            cmd = [
                str(self.steer_cli),
                "-m", str(self.model_path),
                "-p", prompt,
                "-n", str(n_tokens),
                "--temp", str(temperature),
                "--steer-config", config_path,
                "-ngl", str(self.n_gpu_layers)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Steering generation failed: {result.stderr}")

            # Extract generated text from output
            output = result.stdout
            # Find the generation section
            if "Generation:" in output:
                gen_start = output.find("Generation:") + len("Generation:\n")
                gen_text = output[gen_start:].strip()
                # Remove trailing performance info
                if "llama_perf" in gen_text:
                    gen_text = gen_text[:gen_text.find("llama_perf")].strip()
                return gen_text
            return output
        finally:
            os.unlink(config_path)

    def generate_with_single_steering(
        self,
        prompt: str,
        vector_path: str,
        layer: int,
        strength: float = 1.0,
        n_tokens: int = 128,
        temperature: float = 0.5
    ) -> str:
        """
        Generate text with a single steering vector.

        Args:
            prompt: Input prompt
            vector_path: Path to steering vector .npy file
            layer: Layer to apply steering
            strength: Steering strength
            n_tokens: Number of tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        cmd = [
            str(self.steer_cli),
            "-m", str(self.model_path),
            "-p", prompt,
            "-n", str(n_tokens),
            "--temp", str(temperature),
            "--steer-vector", str(vector_path),
            "--steer-layer", str(layer),
            "--steer-strength", str(strength),
            "-ngl", str(self.n_gpu_layers)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Steering generation failed: {result.stderr}")

        output = result.stdout
        if "Generation:" in output:
            gen_start = output.find("Generation:") + len("Generation:\n")
            gen_text = output[gen_start:].strip()
            if "llama_perf" in gen_text:
                gen_text = gen_text[:gen_text.find("llama_perf")].strip()
            return gen_text
        return output

    def compute_steering_vector(
        self,
        positive_prompts: list[str],
        negative_prompts: list[str],
        layer: int,
        output_dir: str = "./activations"
    ) -> np.ndarray:
        """
        Compute steering vector from contrastive prompt pairs.

        The steering vector is computed as:
            mean(positive_activations) - mean(negative_activations)

        Args:
            positive_prompts: List of prompts representing the target concept
            negative_prompts: List of neutral/baseline prompts
            layer: Layer to extract activations from

        Returns:
            Steering vector of shape (hidden_dim,)
        """
        pos_acts = []
        for p in positive_prompts:
            act = self.get_last_token_activation(p, layer, output_dir)
            pos_acts.append(act)

        neg_acts = []
        for p in negative_prompts:
            act = self.get_last_token_activation(p, layer, output_dir)
            neg_acts.append(act)

        pos_mean = np.mean(pos_acts, axis=0)
        neg_mean = np.mean(neg_acts, axis=0)

        steering_vector = pos_mean - neg_mean
        return steering_vector

    def save_steering_vector(
        self,
        vector: np.ndarray,
        path: str
    ) -> None:
        """Save steering vector to .npy file."""
        np.save(path, vector.astype(np.float32))

    def load_steering_vector(
        self,
        path: str
    ) -> np.ndarray:
        """Load steering vector from .npy file."""
        return np.load(path)


def demo():
    """Demo showing basic usage."""
    import sys

    # Configuration
    llama_cpp_dir = Path(__file__).parent.parent / "llama.cpp" / "build"
    model_path = Path(__file__).parent.parent / "llama.cpp" / "models" / "Bonsai-8B.gguf"

    print("=" * 60)
    print("Bonsai-8B Activation Steering Demo")
    print("=" * 60)

    # Initialize model
    model = BonsaiModel(
        model_path=str(model_path),
        llama_cpp_dir=str(llama_cpp_dir)
    )

    # 1. Dump activations
    print("\n1. Dumping activations for 'The capital of France is'...")
    activations = model.dump_activations(
        prompt="The capital of France is",
        layers=[10, 15, 20, 25],
        output_dir="./demo_activations"
    )
    print(f"   Layers dumped: {list(activations.keys())}")
    for layer, token_acts in activations.items():
        print(f"   Layer {layer}: {len(token_acts)} tokens")

    # 2. Save a steering vector (using layer 25 last token activation)
    print("\n2. Saving steering vector...")
    sv = activations[25][max(activations[25].keys())]
    sv_path = "./demo_activations/france_steering_vector.npy"
    model.save_steering_vector(sv, sv_path)
    print(f"   Saved to: {sv_path}")
    print(f"   Shape: {sv.shape}, Norm: {np.linalg.norm(sv):.2f}")

    # 3. Generate with steering
    print("\n3. Generating with steering...")
    prompt = "What is 2+2?"

    print(f"\n   Prompt: {prompt}")
    print("\n   Strength 0.0 (baseline):")
    text = model.generate_with_single_steering(
        prompt=prompt,
        vector_path=sv_path,
        layer=25,
        strength=0.0,
        n_tokens=32,
        temperature=0.0
    )
    print(f"   {text}")

    print("\n   Strength 1.0:")
    text = model.generate_with_single_steering(
        prompt=prompt,
        vector_path=sv_path,
        layer=25,
        strength=1.0,
        n_tokens=32,
        temperature=0.0
    )
    print(f"   {text}")

    print("\n" + "=" * 60)
    print("Demo complete!")


if __name__ == "__main__":
    demo()
