#!/usr/bin/env python3
"""
Bonsai Activation Steering Web UI - Using llama-server backend
Steering control is the main feature, system prompts are supplementary
Includes automatic concept extraction
"""

import os
import sys
import json
import httpx
import asyncio
import subprocess
import signal
from pathlib import Path
from typing import Optional
import uuid

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

# Add python directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

# Configuration
BASE_DIR = Path(__file__).parent.parent
VECTORS_DIR = BASE_DIR / "vectors"
LLAMA_SERVER_BIN = Path("/Users/naoto/prismml-llama.cpp/build/bin/llama-server")
MODEL_PATH = BASE_DIR / "llama.cpp/models/Bonsai-8B.gguf"
LLAMA_SERVER_PORT = 8081
LLAMA_SERVER_URL = f"http://localhost:{LLAMA_SERVER_PORT}"

app = FastAPI(title="Bonsai Steering Chat")

# Global server process and current config
llama_process: Optional[subprocess.Popen] = None
current_steering_config = {
    "gguf_file": None,
    "strength": 0.0,
    "layer_start": 10,
    "layer_end": 25
}


class SteeringServerConfig(BaseModel):
    gguf_file: Optional[str] = None  # None means no steering
    strength: float = 0.2
    layer_start: int = 10
    layer_end: int = 25


class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 200
    temperature: float = 0.7
    system_prompt: str = ""  # Optional supplementary system prompt


class ExtractionRequest(BaseModel):
    concept_name: str
    num_positive: int = 8
    num_negative: int = 8
    layers: list[int] = [10, 12, 15, 18, 20, 22, 25]


class EvaluationRequest(BaseModel):
    concept_name: str
    gguf_file: str
    layer_start: int = 10
    layer_end: int = 25
    test_prompts: Optional[list[str]] = None
    strengths: Optional[list[float]] = None


# Job storage
extraction_jobs: dict[str, dict] = {}
evaluation_jobs: dict[str, dict] = {}


def start_llama_server(config: SteeringServerConfig):
    """Start llama-server with specified control vector configuration"""
    global llama_process, current_steering_config

    # Kill existing process
    if llama_process is not None:
        try:
            llama_process.terminate()
            llama_process.wait(timeout=5)
        except:
            llama_process.kill()
        llama_process = None

    # Build command
    cmd = [
        str(LLAMA_SERVER_BIN),
        "-m", str(MODEL_PATH),
        "-ngl", "99",
        "--port", str(LLAMA_SERVER_PORT),
        "--chat-template", "chatml",  # Qwen3/Bonsai-8B uses ChatML format
    ]

    # Add control vector if specified
    if config.gguf_file:
        gguf_path = VECTORS_DIR / config.gguf_file
        if gguf_path.exists():
            cmd.extend([
                "--control-vector-scaled", f"{gguf_path}:{config.strength}",
                "--control-vector-layer-range", str(config.layer_start), str(config.layer_end)
            ])

    print(f"Starting llama-server: {' '.join(cmd)}")

    # Start process
    llama_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(BASE_DIR)
    )

    # Update current config
    current_steering_config = {
        "gguf_file": config.gguf_file,
        "strength": config.strength if config.gguf_file else 0.0,
        "layer_start": config.layer_start,
        "layer_end": config.layer_end
    }

    return llama_process


async def wait_for_server(timeout: float = 60.0):
    """Wait for llama-server to be ready"""
    start_time = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - start_time < timeout:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"{LLAMA_SERVER_URL}/health")
                if response.status_code == 200:
                    return True
        except:
            pass
        await asyncio.sleep(0.5)
    return False


@app.get("/api/steering/vectors")
async def list_steering_vectors():
    """List available GGUF control vector files"""
    vectors = []

    if VECTORS_DIR.exists():
        for f in VECTORS_DIR.glob("*.gguf"):
            vectors.append({
                "filename": f.name,
                "name": f.stem,
                "size_kb": round(f.stat().st_size / 1024, 1)
            })

    return {"vectors": sorted(vectors, key=lambda x: x["name"])}


@app.get("/api/steering/status")
async def get_steering_status():
    """Get current steering configuration"""
    server_running = False
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{LLAMA_SERVER_URL}/health")
            server_running = response.status_code == 200
    except:
        pass

    return {
        "server_running": server_running,
        "config": current_steering_config
    }


@app.post("/api/steering/apply")
async def apply_steering(config: SteeringServerConfig):
    """Apply new steering configuration (restarts llama-server)"""
    try:
        start_llama_server(config)

        # Wait for server to be ready
        ready = await wait_for_server(timeout=60.0)

        if not ready:
            return {"success": False, "error": "Server failed to start within timeout"}

        return {
            "success": True,
            "config": current_steering_config
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/presets")
async def list_presets():
    """List steering presets - GGUF-based control vectors as main feature"""
    presets = [
        {
            "name": "No Steering",
            "description": "Baseline model without control vectors",
            "steering": {"gguf_file": None, "strength": 0.0},
            "system_prompt": ""
        },
        {
            "name": "Japanese (Weak)",
            "description": "Gentle Japanese steering - strength 0.1",
            "steering": {"gguf_file": "japanese_v2.gguf", "strength": 0.1, "layer_start": 10, "layer_end": 25},
            "system_prompt": ""
        },
        {
            "name": "Japanese (Medium)",
            "description": "Medium Japanese steering - strength 0.2",
            "steering": {"gguf_file": "japanese_v2.gguf", "strength": 0.2, "layer_start": 10, "layer_end": 25},
            "system_prompt": ""
        },
        {
            "name": "Japanese (Strong)",
            "description": "Strong Japanese steering - strength 0.3",
            "steering": {"gguf_file": "japanese_v2.gguf", "strength": 0.3, "layer_start": 10, "layer_end": 25},
            "system_prompt": ""
        },
        {
            "name": "Cat Suppressed",
            "description": "Suppress cat concept - negative steering",
            "steering": {"gguf_file": "cat.gguf", "strength": -0.5, "layer_start": 10, "layer_end": 30},
            "system_prompt": ""
        },
        {
            "name": "Cat Enhanced",
            "description": "Enhance cat concept - positive steering",
            "steering": {"gguf_file": "cat.gguf", "strength": 0.5, "layer_start": 10, "layer_end": 30},
            "system_prompt": ""
        },
    ]
    return {"presets": presets}


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream a response using llama-server"""

    messages = []

    # Add system prompt if provided (supplementary)
    if request.system_prompt:
        messages.append({"role": "system", "content": request.system_prompt})

    messages.append({"role": "user", "content": request.message})

    async def generate():
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{LLAMA_SERVER_URL}/v1/chat/completions",
                    json={
                        "messages": messages,
                        "max_tokens": request.max_tokens,
                        "temperature": request.temperature,
                        "stream": True
                    }
                ) as response:
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                yield f"data: {json.dumps({'done': True})}\n\n"
                                break
                            try:
                                chunk = json.loads(data)
                                if "choices" in chunk and len(chunk["choices"]) > 0:
                                    delta = chunk["choices"][0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        yield f"data: {json.dumps({'token': content})}\n\n"
                            except json.JSONDecodeError:
                                pass
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Non-streaming chat endpoint"""

    messages = []

    if request.system_prompt:
        messages.append({"role": "system", "content": request.system_prompt})

    messages.append({"role": "user", "content": request.message})

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{LLAMA_SERVER_URL}/v1/chat/completions",
                json={
                    "messages": messages,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                }
            )

            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail="llama-server error")

            data = response.json()
            content = data["choices"][0]["message"]["content"]

            return {
                "response": content,
                "prompt": request.message,
                "steering": current_steering_config
            }

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Request timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    """Check if llama-server is available"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{LLAMA_SERVER_URL}/health")
            return {
                "status": "ok",
                "llama_server": response.json(),
                "steering": current_steering_config
            }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main UI"""
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return html_path.read_text()
    return "<h1>Bonsai Steering Chat</h1><p>index.html not found</p>"


# ============ Concept Extraction Endpoints ============

@app.post("/api/extract/start")
async def start_extraction(request: ExtractionRequest, background_tasks: BackgroundTasks):
    """Start concept extraction job"""
    job_id = str(uuid.uuid4())[:8]

    extraction_jobs[job_id] = {
        "status": "pending",
        "progress": 0.0,
        "current_step": "Initializing...",
        "concept_name": request.concept_name,
        "result": None,
        "error": None
    }

    # Run extraction in background
    background_tasks.add_task(
        run_extraction,
        job_id,
        request.concept_name,
        request.num_positive,
        request.num_negative,
        request.layers
    )

    return {"job_id": job_id, "status": "started"}


@app.get("/api/extract/{job_id}")
async def get_extraction_status(job_id: str):
    """Get extraction job status"""
    if job_id not in extraction_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return extraction_jobs[job_id]


@app.post("/api/extract/{job_id}/cancel")
async def cancel_extraction(job_id: str):
    """Cancel extraction job"""
    if job_id not in extraction_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    extraction_jobs[job_id]["status"] = "cancelled"
    return {"status": "cancelled"}


async def run_extraction(
    job_id: str,
    concept_name: str,
    num_positive: int,
    num_negative: int,
    layers: list[int]
):
    """Run concept extraction in background"""
    try:
        from concept_extractor import ConceptExtractor, ExtractionProgress

        def update_progress(progress: ExtractionProgress):
            extraction_jobs[job_id].update({
                "status": progress.status,
                "progress": progress.progress,
                "current_step": progress.current_step,
                "error": progress.error
            })

        extractor = ConceptExtractor(
            model_path=str(MODEL_PATH),
            llama_cpp_dir=str(BASE_DIR / "llama.cpp" / "build"),
            llm_url=LLAMA_SERVER_URL,
            vectors_dir=str(VECTORS_DIR)
        )

        vectors, metadata = await extractor.extract(
            concept_name=concept_name,
            layers=layers,
            num_positive=num_positive,
            num_negative=num_negative,
            progress_callback=update_progress
        )

        # Save vectors and create GGUF
        extractor.save_vectors(vectors, metadata)
        gguf_path = extractor.create_gguf(vectors, metadata)

        extraction_jobs[job_id].update({
            "status": "complete",
            "progress": 1.0,
            "current_step": "Complete!",
            "result": {
                "concept_id": metadata.id,
                "name": metadata.name,
                "gguf_file": gguf_path.name,
                "layers": metadata.layers,
                "vector_norms": metadata.vector_norms,
                "recommended_strength": metadata.recommended_strength,
                "positive_prompts": metadata.positive_prompts,
                "negative_prompts": metadata.negative_prompts
            }
        })

    except Exception as e:
        import traceback
        extraction_jobs[job_id].update({
            "status": "failed",
            "progress": 0.0,
            "current_step": "Extraction failed",
            "error": str(e)
        })
        traceback.print_exc()


@app.get("/api/concepts")
async def list_concepts():
    """List all extracted concepts with their metadata"""
    concepts = []

    if VECTORS_DIR.exists():
        for meta_file in VECTORS_DIR.glob("*_metadata.json"):
            try:
                with open(meta_file) as f:
                    metadata = json.load(f)

                # Check if GGUF exists
                gguf_path = VECTORS_DIR / f"{metadata['id']}.gguf"
                metadata["has_gguf"] = gguf_path.exists()
                metadata["gguf_file"] = f"{metadata['id']}.gguf" if gguf_path.exists() else None

                # Check for evaluation results
                eval_path = BASE_DIR / "results" / f"eval_{metadata['id']}.json"
                if eval_path.exists():
                    with open(eval_path) as f:
                        metadata["evaluation"] = json.load(f)

                concepts.append(metadata)
            except Exception as e:
                print(f"Error loading {meta_file}: {e}")

    return {"concepts": sorted(concepts, key=lambda x: x.get("created_at", ""), reverse=True)}


# ============ Evaluation Endpoints ============

@app.post("/api/evaluate/start")
async def start_evaluation(request: EvaluationRequest, background_tasks: BackgroundTasks):
    """Start concept evaluation job"""
    job_id = str(uuid.uuid4())[:8]

    evaluation_jobs[job_id] = {
        "status": "pending",
        "progress": 0.0,
        "current_step": "Initializing...",
        "concept_name": request.concept_name,
        "result": None,
        "error": None
    }

    background_tasks.add_task(
        run_evaluation,
        job_id,
        request.concept_name,
        request.gguf_file,
        request.layer_start,
        request.layer_end,
        request.test_prompts,
        request.strengths
    )

    return {"job_id": job_id, "status": "started"}


@app.get("/api/evaluate/{job_id}")
async def get_evaluation_status(job_id: str):
    """Get evaluation job status"""
    if job_id not in evaluation_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return evaluation_jobs[job_id]


async def run_evaluation(
    job_id: str,
    concept_name: str,
    gguf_file: str,
    layer_start: int,
    layer_end: int,
    test_prompts: Optional[list[str]],
    strengths: Optional[list[float]]
):
    """Run evaluation in background"""
    try:
        from evaluator import SteeringEvaluator

        def update_progress(msg: str, progress: float):
            evaluation_jobs[job_id].update({
                "progress": progress,
                "current_step": msg
            })

        evaluator = SteeringEvaluator(webui_url="http://localhost:8080")

        result = await evaluator.evaluate(
            concept_name=concept_name,
            gguf_file=gguf_file,
            layer_start=layer_start,
            layer_end=layer_end,
            test_prompts=test_prompts,
            strengths=strengths,
            progress_callback=update_progress
        )

        # Save evaluation results
        results_dir = BASE_DIR / "results"
        results_dir.mkdir(exist_ok=True)

        # Find concept ID from gguf_file name
        concept_id = gguf_file.replace(".gguf", "")
        eval_path = results_dir / f"eval_{concept_id}.json"

        with open(eval_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

        evaluation_jobs[job_id].update({
            "status": "complete",
            "progress": 1.0,
            "current_step": "Evaluation complete!",
            "result": {
                "steering_score": result.steering_score,
                "direction_alignment": result.direction_alignment,
                "length_ratio": result.length_ratio,
                "strength_correlation": result.strength_correlation,
                "recommended_strength": result.recommended_strength
            }
        })

    except Exception as e:
        import traceback
        evaluation_jobs[job_id].update({
            "status": "failed",
            "progress": 0.0,
            "current_step": "Evaluation failed",
            "error": str(e)
        })
        traceback.print_exc()


# ============ Hub Sharing Endpoints ============

class ShareRequest(BaseModel):
    concept_id: str
    description: str = ""
    author: str = "anonymous"
    tags: list[str] = []
    hf_token: Optional[str] = None


@app.get("/api/hub/concepts")
async def list_hub_concepts(tag: str = None):
    """List concepts from the community Hub."""
    try:
        from hub_manager import list_community_concepts
        concepts = list_community_concepts(tag)
        return {
            "concepts": [c.to_dict() for c in concepts],
            "count": len(concepts)
        }
    except ImportError:
        return {"error": "huggingface_hub not installed", "concepts": []}
    except Exception as e:
        return {"error": str(e), "concepts": []}


@app.get("/api/hub/search")
async def search_hub_concepts(q: str):
    """Search concepts in the Hub."""
    try:
        from hub_manager import HubManager
        manager = HubManager()
        concepts = manager.search_concepts(q)
        return {
            "concepts": [c.to_dict() for c in concepts],
            "count": len(concepts),
            "query": q
        }
    except ImportError:
        return {"error": "huggingface_hub not installed", "concepts": []}
    except Exception as e:
        return {"error": str(e), "concepts": []}


@app.post("/api/hub/download/{concept_id}")
async def download_hub_concept(concept_id: str):
    """Download a concept from the Hub to local vectors directory."""
    try:
        from hub_manager import download_community_concept
        local_path, card = download_community_concept(concept_id, VECTORS_DIR)

        # Also save metadata locally
        meta_path = VECTORS_DIR / f"{concept_id}_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(card.to_dict(), f, indent=2)

        return {
            "success": True,
            "concept_id": concept_id,
            "name": card.name,
            "gguf_file": f"{concept_id}.gguf",
            "local_path": str(local_path)
        }
    except ImportError:
        raise HTTPException(status_code=500, detail="huggingface_hub not installed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/hub/share")
async def share_concept_to_hub(request: ShareRequest):
    """Share a local concept to the community Hub."""
    try:
        from hub_manager import share_concept

        # Get local metadata
        meta_path = VECTORS_DIR / f"{request.concept_id}_metadata.json"
        if not meta_path.exists():
            raise HTTPException(status_code=404, detail="Concept metadata not found")

        with open(meta_path) as f:
            metadata = json.load(f)

        # Get evaluation if exists
        eval_path = BASE_DIR / "results" / f"eval_{request.concept_id}.json"
        evaluation = None
        if eval_path.exists():
            with open(eval_path) as f:
                evaluation = json.load(f)

        # Get GGUF path
        gguf_path = VECTORS_DIR / f"{request.concept_id}.gguf"
        if not gguf_path.exists():
            raise HTTPException(status_code=404, detail="Concept GGUF not found")

        # Share to Hub
        url = share_concept(
            concept_id=request.concept_id,
            gguf_path=gguf_path,
            metadata=metadata,
            evaluation=evaluation,
            description=request.description,
            author=request.author,
            tags=request.tags,
            token=request.hf_token
        )

        return {
            "success": True,
            "url": url,
            "concept_id": request.concept_id
        }

    except ImportError:
        raise HTTPException(status_code=500, detail="huggingface_hub not installed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("shutdown")
def shutdown_event():
    """Clean up llama-server on shutdown"""
    global llama_process
    if llama_process is not None:
        llama_process.terminate()
        try:
            llama_process.wait(timeout=5)
        except:
            llama_process.kill()


@app.on_event("startup")
async def startup_event():
    """Auto-start llama-server with no steering on startup"""
    print("Auto-starting llama-server...")
    start_llama_server(SteeringServerConfig(gguf_file=None, strength=0.0))
    ready = await wait_for_server(timeout=60.0)
    if ready:
        print("llama-server ready!")
    else:
        print("Warning: llama-server failed to start")


if __name__ == "__main__":
    print(f"Vectors: {VECTORS_DIR}")
    print(f"llama-server binary: {LLAMA_SERVER_BIN}")
    print(f"Model: {MODEL_PATH}")
    print(f"Starting server at http://localhost:8080")
    uvicorn.run(app, host="0.0.0.0", port=8080)
