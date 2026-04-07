#!/usr/bin/env python3
"""
Activation Steering Experiment for Bonsai-8B

This script demonstrates steering experiments similar to
pphouse/llm_feature_vec Experiment 11 (Cat Topic Steering).
"""

import numpy as np
from pathlib import Path
from bonsai_steering import BonsaiModel
import json


def cat_topic_steering_experiment():
    """
    Cat Topic Steering Experiment

    Compute a "cat" steering vector from contrastive prompts and
    apply it to change model outputs toward cat-related topics.
    """
    print("=" * 60)
    print("Cat Topic Steering Experiment")
    print("=" * 60)

    # Configuration
    llama_cpp_dir = Path(__file__).parent.parent / "llama.cpp" / "build"
    model_path = Path(__file__).parent.parent / "llama.cpp" / "models" / "Bonsai-8B.gguf"
    vectors_dir = Path("./vectors")
    vectors_dir.mkdir(exist_ok=True)

    # Initialize model
    model = BonsaiModel(
        model_path=str(model_path),
        llama_cpp_dir=str(llama_cpp_dir)
    )

    # Contrastive prompts for "cat" concept
    positive_prompts = [
        "Cats are wonderful pets that bring joy to millions.",
        "My cat loves to chase mice around the house.",
        "The kitten played with a ball of yarn all day.",
        "Feline companions have been domesticated for thousands of years.",
        "The cat purred contentedly on my lap.",
        "Whiskers twitched as the cat stalked its prey.",
        "Meowing softly, the cat asked for food.",
        "The tabby cat lounged in the sunny window.",
    ]

    negative_prompts = [
        "The weather today is quite pleasant.",
        "Mathematics is an important subject in school.",
        "The history of ancient Rome is fascinating.",
        "Technology has transformed our daily lives.",
        "The economy shows signs of improvement.",
        "Scientific research advances human knowledge.",
        "Music brings people together across cultures.",
        "The mountain view was breathtaking.",
    ]

    # Compute steering vectors for multiple layers
    target_layers = [10, 15, 20, 25]
    steering_vectors = {}

    print("\n1. Computing steering vectors...")
    for layer in target_layers:
        print(f"   Layer {layer}...", end="", flush=True)
        sv = model.compute_steering_vector(
            positive_prompts=positive_prompts,
            negative_prompts=negative_prompts,
            layer=layer,
            output_dir="./cat_activations"
        )
        steering_vectors[layer] = sv
        sv_path = vectors_dir / f"cat_layer{layer}.npy"
        model.save_steering_vector(sv, str(sv_path))
        print(f" norm={np.linalg.norm(sv):.2f}")

    # Test prompts (neutral)
    test_prompts = [
        "What is 2+2?",
        "Tell me about your day.",
        "The weather is nice today.",
    ]

    # Test with different strengths
    strengths = [0.0, 0.3, 0.5, 1.0]

    print("\n2. Testing steering at layer 25...")
    sv_path = vectors_dir / "cat_layer25.npy"

    for prompt in test_prompts:
        print(f"\n   Prompt: '{prompt}'")
        for strength in strengths:
            output = model.generate_with_single_steering(
                prompt=prompt,
                vector_path=str(sv_path),
                layer=25,
                strength=strength,
                n_tokens=40,
                temperature=0.5
            )
            # Clean up output
            output = output.replace(prompt, "").strip()
            output = output[:100] + "..." if len(output) > 100 else output
            print(f"      Strength {strength:.1f}: {output}")

    # Save experiment results
    results = {
        "experiment": "cat_topic_steering",
        "model": "Bonsai-8B (1-bit Q1_0_g128)",
        "positive_prompts": positive_prompts,
        "negative_prompts": negative_prompts,
        "target_layers": target_layers,
        "steering_vector_norms": {
            str(layer): float(np.linalg.norm(sv))
            for layer, sv in steering_vectors.items()
        },
        "test_prompts": test_prompts,
        "strengths_tested": strengths,
    }

    results_path = Path("./results")
    results_path.mkdir(exist_ok=True)
    with open(results_path / "cat_steering_experiment.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("Experiment complete! Results saved to ./results/")


def japanese_language_steering_experiment():
    """
    Japanese Language Steering Experiment

    Attempt to steer the model toward Japanese language output.
    """
    print("=" * 60)
    print("Japanese Language Steering Experiment")
    print("=" * 60)

    # Configuration
    llama_cpp_dir = Path(__file__).parent.parent / "llama.cpp" / "build"
    model_path = Path(__file__).parent.parent / "llama.cpp" / "models" / "Bonsai-8B.gguf"
    vectors_dir = Path("./vectors")
    vectors_dir.mkdir(exist_ok=True)

    # Initialize model
    model = BonsaiModel(
        model_path=str(model_path),
        llama_cpp_dir=str(llama_cpp_dir)
    )

    # Contrastive prompts for Japanese language
    positive_prompts = [
        "I will respond in Japanese. こんにちは",
        "Japanese is a beautiful language. 日本語は美しい",
        "Let me explain this in Japanese. 説明します",
        "Here is the answer in Japanese: はい",
        "I speak Japanese fluently. 日本語を話します",
    ]

    negative_prompts = [
        "I will respond in English.",
        "English is the common language.",
        "Let me explain this clearly.",
        "Here is the answer:",
        "I speak clearly and concisely.",
    ]

    # Compute steering vector for layer 20
    print("\n1. Computing Japanese steering vector (layer 20)...")
    sv = model.compute_steering_vector(
        positive_prompts=positive_prompts,
        negative_prompts=negative_prompts,
        layer=20,
        output_dir="./japanese_activations"
    )
    sv_path = vectors_dir / "japanese_layer20.npy"
    model.save_steering_vector(sv, str(sv_path))
    print(f"   Norm: {np.linalg.norm(sv):.2f}")

    # Test
    test_prompt = "What is the capital of Japan?"
    strengths = [0.0, 0.5, 1.0, 2.0]

    print(f"\n2. Testing with prompt: '{test_prompt}'")
    for strength in strengths:
        output = model.generate_with_single_steering(
            prompt=test_prompt,
            vector_path=str(sv_path),
            layer=20,
            strength=strength,
            n_tokens=40,
            temperature=0.5
        )
        output = output.replace(test_prompt, "").strip()
        output = output[:100] + "..." if len(output) > 100 else output
        print(f"   Strength {strength:.1f}: {output}")

    print("\n" + "=" * 60)


def main():
    import sys

    if len(sys.argv) > 1:
        experiment = sys.argv[1]
        if experiment == "cat":
            cat_topic_steering_experiment()
        elif experiment == "japanese":
            japanese_language_steering_experiment()
        else:
            print(f"Unknown experiment: {experiment}")
            print("Available: cat, japanese")
    else:
        # Run cat experiment by default
        cat_topic_steering_experiment()


if __name__ == "__main__":
    main()
