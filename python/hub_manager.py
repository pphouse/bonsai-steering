"""
Hugging Face Hub integration for sharing steering concepts.
Allows users to upload/download concept vectors and evaluations.
"""

import os
import json
import tempfile
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from datetime import datetime

try:
    from huggingface_hub import HfApi, hf_hub_download, upload_file, list_repo_files
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


# Default Hub repository for community concepts
DEFAULT_REPO_ID = "pphouse/bonsai-steering-concepts"


@dataclass
class ConceptCard:
    """Metadata card for a shared concept."""
    id: str
    name: str
    description: str
    author: str
    model: str  # e.g., "Bonsai-8B"
    layers: list[int]
    recommended_strength: float
    evaluation: Optional[dict] = None
    tags: list[str] = None
    created_at: str = None
    downloads: int = 0
    likes: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        if d['tags'] is None:
            d['tags'] = []
        if d['created_at'] is None:
            d['created_at'] = datetime.now().isoformat()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'ConceptCard':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class HubManager:
    """Manages interaction with Hugging Face Hub for concept sharing."""

    def __init__(
        self,
        repo_id: str = DEFAULT_REPO_ID,
        local_cache_dir: str = None,
        token: str = None
    ):
        if not HF_AVAILABLE:
            raise ImportError("huggingface_hub not installed. Run: pip install huggingface_hub")

        self.repo_id = repo_id
        self.api = HfApi(token=token)
        self.token = token

        # Local cache for downloaded concepts
        if local_cache_dir:
            self.cache_dir = Path(local_cache_dir)
        else:
            self.cache_dir = Path.home() / ".cache" / "bonsai-steering"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def list_concepts(self, tag: str = None) -> list[ConceptCard]:
        """List all available concepts from the Hub."""
        try:
            files = list_repo_files(self.repo_id)

            concepts = []
            for f in files:
                if f.endswith("_card.json"):
                    # Download and parse concept card
                    local_path = hf_hub_download(
                        repo_id=self.repo_id,
                        filename=f,
                        cache_dir=str(self.cache_dir)
                    )

                    with open(local_path) as fp:
                        data = json.load(fp)
                        card = ConceptCard.from_dict(data)

                        # Filter by tag if specified
                        if tag is None or (card.tags and tag in card.tags):
                            concepts.append(card)

            return sorted(concepts, key=lambda x: x.downloads, reverse=True)

        except Exception as e:
            print(f"Error listing concepts: {e}")
            return []

    def download_concept(self, concept_id: str) -> tuple[Path, ConceptCard]:
        """Download a concept GGUF and its metadata."""
        # Download card
        card_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=f"concepts/{concept_id}_card.json",
            cache_dir=str(self.cache_dir)
        )

        with open(card_path) as f:
            card = ConceptCard.from_dict(json.load(f))

        # Download GGUF
        gguf_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=f"concepts/{concept_id}.gguf",
            cache_dir=str(self.cache_dir)
        )

        return Path(gguf_path), card

    def upload_concept(
        self,
        concept_id: str,
        gguf_path: Path,
        metadata: dict,
        evaluation: dict = None,
        description: str = "",
        author: str = "anonymous",
        tags: list[str] = None
    ) -> str:
        """Upload a concept to the Hub."""

        # Create concept card
        card = ConceptCard(
            id=concept_id,
            name=metadata.get("name", concept_id),
            description=description or f"Steering concept: {metadata.get('name', concept_id)}",
            author=author,
            model="Bonsai-8B",
            layers=metadata.get("layers", []),
            recommended_strength=metadata.get("recommended_strength", 0.2),
            evaluation=evaluation,
            tags=tags or [],
            created_at=datetime.now().isoformat()
        )

        # Upload GGUF file
        gguf_url = upload_file(
            path_or_fileobj=str(gguf_path),
            path_in_repo=f"concepts/{concept_id}.gguf",
            repo_id=self.repo_id,
            token=self.token
        )

        # Upload concept card
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(card.to_dict(), f, indent=2)
            card_temp_path = f.name

        try:
            card_url = upload_file(
                path_or_fileobj=card_temp_path,
                path_in_repo=f"concepts/{concept_id}_card.json",
                repo_id=self.repo_id,
                token=self.token
            )
        finally:
            os.unlink(card_temp_path)

        return f"https://huggingface.co/{self.repo_id}/blob/main/concepts/{concept_id}.gguf"

    def search_concepts(self, query: str) -> list[ConceptCard]:
        """Search concepts by name or tags."""
        all_concepts = self.list_concepts()
        query_lower = query.lower()

        results = []
        for card in all_concepts:
            # Match in name
            if query_lower in card.name.lower():
                results.append(card)
                continue

            # Match in description
            if query_lower in card.description.lower():
                results.append(card)
                continue

            # Match in tags
            if card.tags and any(query_lower in tag.lower() for tag in card.tags):
                results.append(card)

        return results

    def get_concept_details(self, concept_id: str) -> Optional[ConceptCard]:
        """Get detailed information about a concept."""
        try:
            card_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=f"concepts/{concept_id}_card.json",
                cache_dir=str(self.cache_dir)
            )

            with open(card_path) as f:
                return ConceptCard.from_dict(json.load(f))
        except Exception as e:
            print(f"Error getting concept details: {e}")
            return None


# Convenience functions for quick access
def list_community_concepts(tag: str = None) -> list[ConceptCard]:
    """List concepts from the community Hub."""
    manager = HubManager()
    return manager.list_concepts(tag)


def download_community_concept(concept_id: str, vectors_dir: Path) -> tuple[Path, ConceptCard]:
    """Download a concept to the local vectors directory."""
    manager = HubManager()
    gguf_path, card = manager.download_concept(concept_id)

    # Copy to vectors directory
    local_path = vectors_dir / f"{concept_id}.gguf"
    if not local_path.exists():
        import shutil
        shutil.copy(gguf_path, local_path)

    return local_path, card


def share_concept(
    concept_id: str,
    gguf_path: Path,
    metadata: dict,
    evaluation: dict = None,
    description: str = "",
    author: str = "anonymous",
    tags: list[str] = None,
    token: str = None
) -> str:
    """Share a concept to the community Hub."""
    manager = HubManager(token=token)
    return manager.upload_concept(
        concept_id=concept_id,
        gguf_path=gguf_path,
        metadata=metadata,
        evaluation=evaluation,
        description=description,
        author=author,
        tags=tags
    )


if __name__ == "__main__":
    # Test listing concepts
    print("Testing Hub integration...")

    if HF_AVAILABLE:
        try:
            concepts = list_community_concepts()
            print(f"Found {len(concepts)} concepts")
            for c in concepts[:5]:
                print(f"  - {c.name}: {c.description[:50]}...")
        except Exception as e:
            print(f"Hub not accessible: {e}")
    else:
        print("huggingface_hub not installed")
