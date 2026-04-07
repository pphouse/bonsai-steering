#!/usr/bin/env python3
"""
Quantitative Evaluation for Concept Steering

Metrics:
1. Steering Score: How much the response changes toward the target concept
2. Coherence: Whether responses remain coherent with steering
3. Controllability: Linear response to strength changes
4. Concept Specificity: Does steering affect only the target concept?
"""

import asyncio
import json
import httpx
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Callable
from datetime import datetime


@dataclass
class EvaluationResult:
    """Results from steering evaluation."""
    concept_name: str
    gguf_file: str

    # Steering effectiveness
    steering_score: float  # 0-1, how much response changes
    direction_alignment: float  # -1 to 1, alignment with concept direction

    # Coherence
    baseline_length: float  # average response length without steering
    steered_length: float  # average response length with steering
    length_ratio: float  # steered/baseline

    # Controllability
    strength_correlation: float  # correlation between strength and effect
    recommended_strength: float  # optimal strength based on evaluation

    # Test details
    test_prompts: list[str]
    strengths_tested: list[float]
    responses: dict  # {strength: [responses]}

    created_at: str

    def to_dict(self):
        return asdict(self)

    def summary(self) -> str:
        return f"""
Evaluation Results for "{self.concept_name}"
{'='*50}
Steering Score:      {self.steering_score:.3f} (0=no effect, 1=strong effect)
Direction Alignment: {self.direction_alignment:.3f} (-1 to 1)
Length Ratio:        {self.length_ratio:.2f}x (steered/baseline)
Controllability:     {self.strength_correlation:.3f} (strength-effect correlation)
Recommended Strength: {self.recommended_strength:.2f}
{'='*50}
"""


class SteeringEvaluator:
    """Evaluate steering vector effectiveness."""

    DEFAULT_TEST_PROMPTS = [
        "Hello, how are you today?",
        "Tell me about yourself.",
        "What is your favorite thing?",
        "Can you help me with something?",
        "What do you think about the weather?",
    ]

    DEFAULT_STRENGTHS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

    def __init__(
        self,
        webui_url: str = "http://localhost:8080",
        timeout: float = 60.0
    ):
        self.webui_url = webui_url
        self.timeout = timeout

    async def evaluate(
        self,
        concept_name: str,
        gguf_file: str,
        layer_start: int = 10,
        layer_end: int = 25,
        test_prompts: Optional[list[str]] = None,
        strengths: Optional[list[float]] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> EvaluationResult:
        """
        Evaluate a steering vector's effectiveness.

        Args:
            concept_name: Name of the concept being evaluated
            gguf_file: GGUF file containing the steering vectors
            layer_start: Start layer for steering
            layer_end: End layer for steering
            test_prompts: Prompts to test with
            strengths: Strength values to test
            progress_callback: Callback for progress updates (message, progress 0-1)

        Returns:
            EvaluationResult with all metrics
        """
        if test_prompts is None:
            test_prompts = self.DEFAULT_TEST_PROMPTS
        if strengths is None:
            strengths = self.DEFAULT_STRENGTHS

        def report(msg: str, progress: float):
            if progress_callback:
                progress_callback(msg, progress)

        responses = {}
        total_tests = len(strengths) * len(test_prompts)
        current = 0

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for strength in strengths:
                report(f"Testing strength {strength}...", current / total_tests)

                # Apply steering
                await client.post(
                    f"{self.webui_url}/api/steering/apply",
                    json={
                        "gguf_file": gguf_file if strength > 0 else None,
                        "strength": strength,
                        "layer_start": layer_start,
                        "layer_end": layer_end
                    }
                )

                # Wait for server restart
                await asyncio.sleep(2)

                responses[strength] = []
                for prompt in test_prompts:
                    current += 1
                    report(f"Testing strength {strength}, prompt {current % len(test_prompts) + 1}...",
                           current / total_tests)

                    try:
                        resp = await client.post(
                            f"{self.webui_url}/api/chat",
                            json={
                                "message": prompt,
                                "max_tokens": 100,
                                "temperature": 0.7
                            }
                        )
                        data = resp.json()
                        responses[strength].append(data.get("response", ""))
                    except Exception as e:
                        responses[strength].append(f"[Error: {e}]")

        # Calculate metrics
        report("Calculating metrics...", 0.95)

        # Steering score: how different are steered responses from baseline?
        baseline_responses = responses.get(0.0, [])
        max_strength_responses = responses.get(max(strengths), [])

        steering_score = self._calculate_difference_score(
            baseline_responses, max_strength_responses
        )

        # Length analysis
        baseline_lengths = [len(r) for r in baseline_responses]
        steered_lengths = [len(r) for r in max_strength_responses]
        baseline_length = np.mean(baseline_lengths) if baseline_lengths else 0
        steered_length = np.mean(steered_lengths) if steered_lengths else 0
        length_ratio = steered_length / baseline_length if baseline_length > 0 else 1.0

        # Controllability: correlation between strength and response change
        strength_effects = []
        for strength in strengths:
            if strength == 0.0:
                continue
            effect = self._calculate_difference_score(
                baseline_responses, responses.get(strength, [])
            )
            strength_effects.append((strength, effect))

        if len(strength_effects) >= 2:
            x = [s for s, _ in strength_effects]
            y = [e for _, e in strength_effects]
            strength_correlation = float(np.corrcoef(x, y)[0, 1])

            # Find recommended strength (where effect plateaus)
            recommended_strength = self._find_optimal_strength(strength_effects)
        else:
            strength_correlation = 0.0
            recommended_strength = 0.2

        # Direction alignment (simplified - using length change as proxy)
        direction_alignment = min(1.0, max(-1.0, (length_ratio - 1.0)))

        report("Evaluation complete!", 1.0)

        return EvaluationResult(
            concept_name=concept_name,
            gguf_file=gguf_file,
            steering_score=steering_score,
            direction_alignment=direction_alignment,
            baseline_length=baseline_length,
            steered_length=steered_length,
            length_ratio=length_ratio,
            strength_correlation=strength_correlation if not np.isnan(strength_correlation) else 0.0,
            recommended_strength=recommended_strength,
            test_prompts=test_prompts,
            strengths_tested=strengths,
            responses=responses,
            created_at=datetime.now().isoformat()
        )

    def _calculate_difference_score(
        self,
        baseline: list[str],
        steered: list[str]
    ) -> float:
        """Calculate how different steered responses are from baseline."""
        if not baseline or not steered:
            return 0.0

        scores = []
        for b, s in zip(baseline, steered):
            if not b or not s:
                continue

            # Character-level Jaccard distance
            b_chars = set(b.lower())
            s_chars = set(s.lower())

            if not b_chars and not s_chars:
                scores.append(0.0)
                continue

            intersection = len(b_chars & s_chars)
            union = len(b_chars | s_chars)
            jaccard = intersection / union if union > 0 else 0

            # Convert similarity to difference
            scores.append(1.0 - jaccard)

        return np.mean(scores) if scores else 0.0

    def _find_optimal_strength(
        self,
        strength_effects: list[tuple[float, float]]
    ) -> float:
        """Find the strength where effect is good but not excessive."""
        if not strength_effects:
            return 0.2

        # Sort by strength
        sorted_effects = sorted(strength_effects, key=lambda x: x[0])

        # Find knee point - where additional strength gives diminishing returns
        for i in range(1, len(sorted_effects) - 1):
            prev_strength, prev_effect = sorted_effects[i-1]
            curr_strength, curr_effect = sorted_effects[i]
            next_strength, next_effect = sorted_effects[i+1]

            # Calculate slopes
            slope1 = (curr_effect - prev_effect) / (curr_strength - prev_strength + 0.001)
            slope2 = (next_effect - curr_effect) / (next_strength - curr_strength + 0.001)

            # If slope decreases significantly, we found the knee
            if slope1 > 0 and slope2 < slope1 * 0.5:
                return curr_strength

        # Default to moderate strength
        return 0.2


async def evaluate_concept(
    concept_name: str,
    gguf_file: str,
    webui_url: str = "http://localhost:8080"
) -> EvaluationResult:
    """Convenience function to evaluate a single concept."""
    evaluator = SteeringEvaluator(webui_url=webui_url)

    def progress(msg, p):
        print(f"[{p*100:.0f}%] {msg}")

    result = await evaluator.evaluate(
        concept_name=concept_name,
        gguf_file=gguf_file,
        progress_callback=progress
    )

    return result


async def demo():
    """Demo evaluation."""
    print("=" * 60)
    print("Steering Evaluation Demo")
    print("=" * 60)

    # Check if there are any concepts to evaluate
    async with httpx.AsyncClient() as client:
        resp = await client.get("http://localhost:8080/api/concepts")
        concepts = resp.json().get("concepts", [])

    if not concepts:
        print("No concepts found. Please extract a concept first.")
        return

    # Evaluate first concept
    concept = concepts[0]
    print(f"\nEvaluating: {concept['name']}")
    print("-" * 40)

    result = await evaluate_concept(
        concept_name=concept['name'],
        gguf_file=concept['gguf_file']
    )

    print(result.summary())

    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    output_path = results_dir / f"eval_{concept['id']}.json"
    with open(output_path, 'w') as f:
        json.dump(result.to_dict(), f, indent=2, default=str)

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(demo())
