# Bonsai Steering Platform

**Automatic concept vector extraction and steering for Bonsai-8B (1-bit LLM)**

1-bit量子化モデル（Bonsai-8B Q1_0_g128）上でActivation Steeringを実現するプラットフォーム。
概念名を入力するだけで、自動的にステアリングベクトルを抽出し、モデルの振る舞いを制御できます。

## Features

- **Automatic Concept Extraction**: 概念名入力 → LLMがプロンプト生成 → 多層ベクトル自動抽出
- **Real-time Steering**: 推論時にリアルタイムでステアリング適用、強度調整可能
- **WebUI**: 概念抽出、ステアリング制御、チャットのためのインタラクティブUI
- **GGUF Support**: llama.cpp互換のcontrol vector形式

## Quick Start

### Prerequisites

- Python 3.9+
- [PrismML llama.cpp](https://github.com/PrismML/llama.cpp) with control vector support
- Bonsai-8B model (Q1_0_g128 GGUF format)

### Installation

```bash
# Clone the repository
git clone https://github.com/naotokui/bonsai-steering.git
cd bonsai-steering

# Install dependencies
pip install fastapi uvicorn httpx numpy pydantic

# Start the WebUI
python webui/server.py
```

### Usage

1. http://localhost:8080 をブラウザで開く
2. **概念抽出**: 概念名（例: "happiness", "formal speech"）を入力して「Extract」
3. **ステアリング適用**: 抽出した概念またはプリセットを選択
4. **チャット**: ステアリングされたモデルと対話

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    WebUI (FastAPI)                      │
│  - Chat interface with streaming                        │
│  - Concept extraction UI                                │
│  - Steering controls                                    │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────┐
│                  Concept Extractor                      │
│  - PromptGenerator (LLM-based)                         │
│  - Activation dumping (llama-activation-dump)          │
│  - Vector computation (mean difference)                │
│  - GGUF conversion                                      │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────┐
│                   llama-server                          │
│  - Control vector support (--control-vector-scaled)    │
│  - Chat completions API                                 │
│  - Streaming                                            │
└─────────────────────────────────────────────────────────┘
```

## How It Works

### Concept Vector Extraction

1. **Prompt Generation**: LLMがpositive prompts（概念を体現）とnegative prompts（中立ベースライン）を生成
2. **Activation Dumping**: 指定レイヤーで隠れ状態を抽出
3. **Vector Computation**: `steering_vector = mean(positive) - mean(negative)`
4. **Multi-layer**: 複数レイヤー（例: 10, 12, 15, 18, 20, 22, 25）でベクトル抽出

### Steering Application

推論時に残差ストリームにcontrol vectorを加算:
```
hidden_state = hidden_state + strength * steering_vector
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Non-streaming chat |
| `/api/chat/stream` | POST | Streaming chat (SSE) |
| `/api/steering/apply` | POST | Apply steering configuration |
| `/api/steering/status` | GET | Get current steering |
| `/api/extract/start` | POST | Start concept extraction |
| `/api/extract/{job_id}` | GET | Get extraction progress |
| `/api/concepts` | GET | List extracted concepts |

## Project Structure

```
bonsai-steering/
├── webui/
│   ├── server.py           # FastAPI server
│   └── index.html          # Frontend UI
├── python/
│   ├── bonsai_steering.py  # Core steering wrapper
│   └── concept_extractor.py # Automatic extraction
├── convert_to_gguf.py      # GGUF conversion utility
└── vectors/                # Extracted vectors (metadata only in git)
```

## CLI Usage

### Activation Dump

```bash
./llama.cpp/build/bin/llama-activation-dump \
  -m llama.cpp/models/Bonsai-8B.gguf \
  -p "The capital of France is" \
  --dump-activations ./activations/ \
  --dump-layers "10,15,20,25" \
  -ngl 99
```

### Python API

```python
from python.concept_extractor import ConceptExtractor

extractor = ConceptExtractor(
    model_path="llama.cpp/models/Bonsai-8B.gguf",
    llama_cpp_dir="llama.cpp/build",
    llm_url="http://localhost:8081"
)

# Extract concept vectors automatically
vectors, metadata = await extractor.extract(
    concept_name="happiness",
    layers=[10, 15, 20, 25],
    num_positive=8,
    num_negative=8
)

# Save and create GGUF
extractor.save_vectors(vectors, metadata)
gguf_path = extractor.create_gguf(vectors, metadata)
```

## Model Information

- **Model**: Bonsai-8B (prism-ml/Bonsai-8B-gguf)
- **Quantization**: Q1_0_g128 (1-bit weights, FP32 activations)
- **File Size**: ~1.1GB
- **Architecture**: Qwen3-based
- **Hidden Dim**: 4096
- **Layers**: 36

## Roadmap

- [x] Core steering functionality
- [x] WebUI with real-time streaming
- [x] Automatic concept extraction
- [ ] Quantitative evaluation metrics
- [ ] Public concept database (Hugging Face Hub)
- [ ] Concept map visualization
- [ ] pip package distribution

## License

MIT License

## Acknowledgments

- [PrismML](https://prismml.io/) for Bonsai-8B and llama.cpp fork with control vector support
- Inspired by [representation engineering](https://arxiv.org/abs/2310.01405) research
