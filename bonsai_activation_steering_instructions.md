# Claude Code 指示書: Bonsai-8B (1-bit GGUF) 活性化ステアリング基盤構築

## プロジェクト概要

llama.cpp を改造して、Bonsai-8B (Q1_0_g128) の中間層活性化を**ダンプ・介入**できるようにする。
最終目標は、`pphouse/llm_feature_vec` で行っている TransformerLens ベースの Activation Steering を、1-bit GGUF モデル上で再現すること。

**背景**: TransformerLens は HuggingFace Transformers 形式のモデルにしか対応しておらず、llama.cpp の GGUF 推論パスにはフック機構がない。本プロジェクトでは llama.cpp のフォワードパスに介入点を追加し、Python から制御できるようにする。

---

## Phase 0: 環境構築

### タスク 0-1: PrismML fork の llama.cpp をクローン＆ビルド

```bash
git clone https://github.com/PrismML-Eng/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON && cmake --build build -j
```

- PrismML fork には Q1_0_g128 カーネルが含まれている（upstream にはない）
- CUDA が使えない場合は `-DGGML_CUDA=OFF` で CPU ビルド、Mac なら Metal がデフォルト
- ビルド成功を `./build/bin/llama-cli --help` で確認

### タスク 0-2: Bonsai-8B モデルのダウンロード

```bash
huggingface-cli download prism-ml/Bonsai-8B-gguf Bonsai-8B.gguf --local-dir ./models/
```

- ファイルサイズ: 約 1.16 GB
- SHA256: `ead25897bc034fa52569d0c6d054ce38216f95db09900c8add8f6bbfb370cff1`

### タスク 0-3: 動作確認

```bash
./build/bin/llama-cli \
  -m models/Bonsai-8B.gguf \
  -p "What is 2+2?" \
  -n 64 --temp 0.5 -ngl 99
```

正常にテキストが生成されることを確認。

---

## Phase 1: 活性化ダンプ機能の追加

### 目的

フォワードパス中の各 Transformer 層の residual stream（残差ストリーム）を、推論を止めずにファイルに書き出せるようにする。

### タスク 1-1: llama.cpp のフォワードパス構造を理解する

以下のファイルを読んで、フォワードパスの流れを把握する:

1. `src/llama.cpp` — モデルのロードとグラフ構築
2. `src/llama-context.cpp` — 推論コンテキスト管理
3. `ggml/src/ggml.c` (または `ggml-cuda/`) — テンソル演算

**確認すべきポイント**:
- 各 Transformer 層の attention + FFN 後の残差接続がどこにあるか
- テンソル名の命名規則（例: `blk.0.attn_norm`, `blk.0.ffn_out` など）
- GGUF Q1_0_g128 の逆量子化が推論グラフのどの時点で起きるか
  - 重要: 重みは 1-bit だが、**活性化（中間テンソル）は FP16/FP32** のはず。確認すること

### タスク 1-2: 活性化テンソルに名前でアクセスする仕組みを調査

llama.cpp の計算グラフ（`ggml_cgraph`）内で、各層の残差ストリームのテンソルを名前で検索できるか確認する。

```c
// 例: ggml_graph_get_tensor() のような関数があるか
struct ggml_tensor * t = ggml_graph_get_tensor(graph, "blk.15.ffn_out");
```

もしない場合は、テンソル名→ポインタのマッピングを自前で構築する必要がある。

### タスク 1-3: 活性化ダンプ用のコールバック機構を実装

`llama-cli` または新規ツール `llama-activation-dump` に以下の機能を追加:

```
コマンドライン引数:
  --dump-activations <出力ディレクトリ>
  --dump-layers <レイヤー番号のカンマ区切り>  (例: "10,15,20,25")
  --dump-format <numpy|safetensors|raw>
```

**実装方針**:

1. `ggml_backend_sched_set_eval_callback()` を使うか、計算グラフ実行後にテンソルデータを読み出す
2. 指定レイヤーの残差ストリームテンソルを取得
3. FP32 に変換してファイルに書き出す

**出力フォーマット** (numpy `.npy` 推奨):

```
output_dir/
  layer_10_token_0.npy   # shape: (hidden_dim,)
  layer_10_token_1.npy
  layer_15_token_0.npy
  ...
  metadata.json          # プロンプト、トークン列、モデル情報
```

### タスク 1-4: ダンプ機能の検証

```bash
./build/bin/llama-activation-dump \
  -m models/Bonsai-8B.gguf \
  -p "The capital of France is" \
  --dump-activations ./activations/ \
  --dump-layers "10,15,20,25" \
  --dump-format numpy
```

Python で読み込んで検証:

```python
import numpy as np
act = np.load("activations/layer_20_token_4.npy")
print(f"Shape: {act.shape}")   # 期待: (hidden_dim,) 例: (4096,)
print(f"Dtype: {act.dtype}")   # 期待: float32
print(f"Norm: {np.linalg.norm(act):.2f}")
print(f"Mean: {act.mean():.4f}, Std: {act.std():.4f}")
```

**検証項目**:
- shape が hidden_dim と一致するか
- 値が NaN/Inf でないか
- 異なるプロンプトで異なる値が出るか（決定論的ではない場合）
- 同じプロンプト・同じ seed で再現性があるか

---

## Phase 2: 活性化介入（Steering）機能の追加

### 目的

推論中の特定レイヤーの残差ストリームに、外部から読み込んだベクトルを加算できるようにする。

### タスク 2-1: ステアリングベクトルの注入機構を実装

```
コマンドライン引数:
  --steer-vector <ベクトルファイル.npy>
  --steer-layer <レイヤー番号>        (例: 25)
  --steer-strength <スカラー値>      (例: 1.5)
  --steer-token-pos <位置>           ("last" | "all" | 整数)
```

**実装方針**:

1. 起動時に .npy ファイルからステアリングベクトル `v` を読み込む
2. `ggml_backend_sched_set_eval_callback()` 内、または計算グラフにカスタムオペレーションを挿入
3. 指定レイヤーの残差ストリームテンソル `h` に対して: `h[token_pos] += strength * v`
4. 変更後の `h` でフォワードパスを継続

**注意点**:
- CUDA の場合、テンソルが GPU メモリ上にある。CPU にコピーして加算→戻す、またはカスタム CUDA カーネルで直接加算
- 自己回帰生成の各ステップで介入が必要（生成トークンごとに毎回）
- 多層ステアリング対応: 複数の `--steer-vector` / `--steer-layer` ペアを受け付ける

### タスク 2-2: 多層ステアリング対応

pphouse/llm_feature_vec の Experiment 15 では複数層（10, 15, 20, 25）に同時介入している。これを再現するため:

```
--steer-config <JSONファイル>
```

JSON フォーマット:

```json
{
  "interventions": [
    {"layer": 10, "vector": "vectors/japanese_layer10.npy", "strength": 0.5},
    {"layer": 15, "vector": "vectors/japanese_layer15.npy", "strength": 0.5},
    {"layer": 20, "vector": "vectors/japanese_layer20.npy", "strength": 0.5},
    {"layer": 25, "vector": "vectors/japanese_layer25.npy", "strength": 0.5}
  ],
  "token_position": "last"
}
```

### タスク 2-3: 介入機能の検証

1. Phase 1 で取得した活性化をそのままステアリングベクトルとして使い、出力が変化することを確認
2. ゼロベクトルを注入して出力が変化しないことを確認
3. 強度を 0.0 → 0.5 → 1.0 → 2.0 と変化させて出力の変化を観察

---

## Phase 3: Python バインディング

### 目的

C++ で実装した機能を Python から呼び出せるようにし、既存の実験コードとの統合を容易にする。

### タスク 3-1: Python ラッパーの作成

`llama-cpp-python` のフォークを作るか、subprocess ベースの簡易ラッパーを作成:

**Option A: subprocess ベース（推奨、実装が軽い）**

```python
# bonsai_steering.py

import subprocess
import json
import numpy as np
from pathlib import Path
from typing import Optional

class BonsaiModel:
    """Bonsai-8B の活性化ダンプ・ステアリングを行うラッパー"""

    def __init__(self, model_path: str, llama_cli_path: str):
        self.model_path = model_path
        self.llama_cli_path = llama_cli_path

    def dump_activations(
        self,
        prompt: str,
        layers: list[int],
        output_dir: str = "./activations"
    ) -> dict[int, np.ndarray]:
        """指定レイヤーの活性化をダンプして返す"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        layer_str = ",".join(str(l) for l in layers)

        cmd = [
            self.llama_cli_path,
            "-m", self.model_path,
            "-p", prompt,
            "-n", "1",
            "--dump-activations", output_dir,
            "--dump-layers", layer_str,
            "--dump-format", "numpy",
            "-ngl", "99"
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        # 結果を読み込む
        activations = {}
        for layer in layers:
            path = Path(output_dir) / f"layer_{layer}_last.npy"
            if path.exists():
                activations[layer] = np.load(str(path))
        return activations

    def generate_with_steering(
        self,
        prompt: str,
        steer_config: dict,
        n_tokens: int = 128
    ) -> str:
        """ステアリングベクトルを適用して生成"""
        config_path = "/tmp/steer_config.json"
        with open(config_path, "w") as f:
            json.dump(steer_config, f)

        cmd = [
            self.llama_cli_path,
            "-m", self.model_path,
            "-p", prompt,
            "-n", str(n_tokens),
            "--steer-config", config_path,
            "-ngl", "99"
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return result.stdout

    def compute_steering_vector(
        self,
        positive_prompts: list[str],
        negative_prompts: list[str],
        layer: int
    ) -> np.ndarray:
        """対照的なプロンプトペアからステアリングベクトルを計算"""
        pos_acts = []
        for p in positive_prompts:
            acts = self.dump_activations(p, [layer])
            if layer in acts:
                pos_acts.append(acts[layer])

        neg_acts = []
        for p in negative_prompts:
            acts = self.dump_activations(p, [layer])
            if layer in acts:
                neg_acts.append(acts[layer])

        pos_mean = np.mean(pos_acts, axis=0)
        neg_mean = np.mean(neg_acts, axis=0)

        steering_vector = pos_mean - neg_mean
        return steering_vector
```

**Option B: ctypes / pybind11 ベース（高性能だが工数大）**

- `llama.h` の既存 C API を拡張してコールバックを登録
- pybind11 で Python モジュールとしてビルド
- 毎トークン生成ごとに Python コールバックが走る

→ Phase 3 では Option A を先に実装し、性能が問題になったら Option B に移行。

### タスク 3-2: 既存実験の再現テスト

pphouse/llm_feature_vec の Experiment 11 (Cat Topic Steering) 相当を Bonsai-8B で再現:

```python
model = BonsaiModel(
    model_path="models/Bonsai-8B.gguf",
    llama_cli_path="./build/bin/llama-activation-dump"
)

# 猫関連プロンプト vs 中立プロンプト
positive = [
    "Cats are wonderful pets that bring joy to millions.",
    "My cat loves to chase mice around the house.",
    "The kitten played with a ball of yarn all day.",
    # ... 15例
]
negative = [
    "The weather today is quite pleasant.",
    "Mathematics is an important subject in school.",
    "The history of ancient Rome is fascinating.",
    # ... 15例
]

# Layer 25 でステアリングベクトルを計算
sv = model.compute_steering_vector(positive, negative, layer=25)
np.save("vectors/cat_layer25.npy", sv)
print(f"Steering vector norm: {np.linalg.norm(sv):.2f}")

# ステアリング適用
config = {
    "interventions": [
        {"layer": 25, "vector": "vectors/cat_layer25.npy", "strength": 1.0}
    ],
    "token_position": "last"
}

for strength in [0.0, 0.5, 1.0, 1.5, 2.0]:
    config["interventions"][0]["strength"] = strength
    output = model.generate_with_steering("What is 2+2?", config)
    print(f"\n=== Strength {strength} ===")
    print(output)
```

---

## Phase 4: 1-bit 特有の研究実験

### タスク 4-1: 1-bit vs FP16 の活性化空間比較

Bonsai-8B (1-bit) と unpacked 版 (FP16) で同じプロンプトの活性化を比較:

```python
# 比較メトリクス
# 1. コサイン類似度
# 2. ベクトルノルムの分布
# 3. 主成分分析（PCA）での可視化
# 4. ステアリングベクトルの類似度
```

**検証すべき仮説**:
- 1-bit の重みから生成される活性化は、FP16 と比べてどの程度の情報を保持しているか
- ステアリングベクトルの方向は 1-bit と FP16 で類似しているか
- 1-bit モデルでのステアリング最適強度は FP16 と異なるか

### タスク 4-2: ステアリング耐性実験

以下の概念でステアリングを試行し、1-bit モデルの特性を調べる:

| 概念カテゴリ | テスト概念 | 期待 (FP16参考) |
|---|---|---|
| 具体的コンテンツ | cat, food, space | 成功しやすい |
| 言語 | Japanese, French | 部分的成功 |
| 抽象的行動 | formal, emotional | 困難 |
| 固有名詞 | brand names | 困難 |

**記録すべきデータ**:
- 各強度での出力テキスト
- 崩壊（collapse）が起きる強度閾値
- ベクトルノルム
- ステアリング前後のパープレキシティ変化

### タスク 4-3: 結果レポートの作成

Experiment 結果を STEERING_EXPERIMENTS_SUMMARY.md と同じフォーマットで記録。

---

## ディレクトリ構成

```
bonsai-activation-steering/
├── llama.cpp/                     # PrismML fork (改造版)
│   ├── src/
│   │   ├── llama.cpp              # 改造: 活性化ダンプ・介入コード
│   │   └── llama-activation-dump.cpp  # 新規: 専用CLI
│   └── build/
├── python/
│   ├── bonsai_steering.py         # Python ラッパー
│   ├── compute_vectors.py         # ステアリングベクトル計算
│   ├── run_experiment.py          # 実験実行スクリプト
│   └── visualize.py               # 結果可視化
├── vectors/                       # 計算済みステアリングベクトル (.npy)
├── activations/                   # ダンプした活性化データ
├── results/                       # 実験結果
│   ├── experiment_cat.md
│   ├── experiment_japanese.md
│   └── 1bit_vs_fp16_comparison.md
├── models/
│   └── Bonsai-8B.gguf
└── README.md
```

---

## 実装の優先順位

| 優先度 | Phase | タスク | 推定工数 | ブロッカー |
|---|---|---|---|---|
| **P0** | 0 | 環境構築・動作確認 | 1時間 | なし |
| **P0** | 1-1,1-2 | フォワードパス構造の理解 | 2-4時間 | C++コードの複雑さ |
| **P1** | 1-3 | 活性化ダンプ実装 | 4-8時間 | ggml API の理解 |
| **P1** | 1-4 | ダンプ検証 | 1時間 | Phase 1-3 |
| **P1** | 2-1 | ステアリング注入実装 | 4-8時間 | Phase 1完了 |
| **P2** | 2-2 | 多層ステアリング | 2-4時間 | Phase 2-1 |
| **P2** | 3-1 | Python ラッパー | 2-4時間 | Phase 2完了 |
| **P3** | 3-2 | 既存実験の再現 | 2-4時間 | Phase 3-1 |
| **P3** | 4 | 1-bit 特有の研究 | 4-8時間 | Phase 3完了 |

**合計推定工数: 22-42 時間**

---

## Claude Code への追加指示

### コーディングスタイル

- llama.cpp の既存コードスタイルに合わせる（C スタイルの命名、`ggml_` プレフィックス等）
- 新規ファイルはなるべく追加、既存ファイルへの変更は最小限
- 変更箇所には `// ACTIVATION_STEERING:` コメントで目印をつける

### デバッグ方針

- まず `fprintf(stderr, ...)` でテンソル名・shape・値を出力して確認
- CUDA テンソルは `ggml_backend_tensor_get()` で CPU に読み出してから確認
- セグフォルトが出たら `compute graph` のテンソル一覧をダンプして、名前とサイズを確認

### 重要な注意事項

1. **活性化は FP16/FP32**: 重みが 1-bit でも、行列積の結果（活性化）は浮動小数点。ダンプ対象は活性化であって重みではない
2. **Q1_0_g128 の逆量子化**: PrismML fork のカスタムカーネルが逆量子化を担当している。このカーネルの出力テンソルが活性化の起点
3. **メモリ制約**: Bonsai-8B は 1.16 GB しかないので VRAM は余裕がある。活性化のダンプ用バッファも十分確保可能
4. **KV キャッシュ**: 自己回帰生成時、KV キャッシュの介入は不要。残差ストリームへの介入のみでステアリングは機能する（TransformerLens と同じ）

### 参考になるコード

- `examples/embedding/embedding.cpp` — 埋め込みベクトルの取り出し方の参考
- `examples/perplexity/perplexity.cpp` — トークンごとの内部状態アクセスの参考
- `ggml/include/ggml.h` — テンソル API のリファレンス
- `src/llama.cpp` 内の `llm_build_*` 関数群 — 計算グラフ構築の詳細

---

## 成功基準

### Phase 1 完了条件
- [ ] 指定レイヤーの活性化を .npy ファイルとして書き出せる
- [ ] Python で読み込んで shape, dtype, 値の範囲が妥当
- [ ] 異なるプロンプトで異なる活性化が得られる

### Phase 2 完了条件
- [ ] 外部の .npy ベクトルを指定レイヤーに加算できる
- [ ] ゼロベクトルで出力が変化しない
- [ ] 非ゼロベクトルで出力が変化する
- [ ] 強度パラメータが正しく機能する

### Phase 3 完了条件
- [ ] Python からワンライナーで活性化ダンプが可能
- [ ] Python からワンライナーでステアリング生成が可能
- [ ] 対照的プロンプトからステアリングベクトルを自動計算できる

### Phase 4 完了条件
- [ ] Cat Topic Steering が Bonsai-8B で動作するか判定
- [ ] 1-bit と FP16 の活性化空間の比較レポート完成
- [ ] 1-bit モデル特有のステアリング特性を文書化
