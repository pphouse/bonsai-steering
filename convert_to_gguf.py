#!/usr/bin/env python3
"""
Convert numpy steering vectors to GGUF format for use with llama-server
"""

import numpy as np
import struct
import sys
from pathlib import Path

# GGUF constants
GGUF_MAGIC = 0x46554747  # "GGUF"
GGUF_VERSION = 3

# GGUF value types
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_STRING = 8

# GGML types
GGML_TYPE_F32 = 0


def write_string(f, s):
    """Write a GGUF string (length + bytes)"""
    encoded = s.encode('utf-8')
    f.write(struct.pack('<Q', len(encoded)))
    f.write(encoded)


def write_gguf_header(f, n_tensors, n_kv):
    """Write GGUF file header"""
    f.write(struct.pack('<I', GGUF_MAGIC))
    f.write(struct.pack('<I', GGUF_VERSION))
    f.write(struct.pack('<Q', n_tensors))
    f.write(struct.pack('<Q', n_kv))


def write_kv_string(f, key, value):
    """Write a string key-value pair"""
    write_string(f, key)
    f.write(struct.pack('<I', GGUF_TYPE_STRING))
    write_string(f, value)


def write_kv_int32(f, key, value):
    """Write an int32 key-value pair"""
    write_string(f, key)
    f.write(struct.pack('<I', GGUF_TYPE_INT32))
    f.write(struct.pack('<i', value))


def create_control_vector_gguf(vectors_dict, output_path, arch="qwen3", model_hint="Bonsai-8B"):
    """
    Create a GGUF control vector file from numpy vectors

    vectors_dict: {layer_num: numpy_array}
    """

    n_tensors = len(vectors_dict)
    n_kv = 3  # architecture, model_hint, layer_count

    # Prepare tensor data
    tensor_infos = []
    tensor_data = []
    offset = 0

    for layer, vec in sorted(vectors_dict.items()):
        vec = vec.astype(np.float32)
        name = f"direction.{layer}"

        # Tensor info: name, n_dims, dims, type, offset
        tensor_infos.append({
            'name': name,
            'n_dims': 1,
            'dims': [vec.shape[0]],
            'type': GGML_TYPE_F32,
            'offset': offset
        })

        tensor_data.append(vec.tobytes())
        offset += len(tensor_data[-1])

        # Align to 32 bytes
        padding = (32 - (offset % 32)) % 32
        if padding:
            tensor_data.append(b'\x00' * padding)
            offset += padding

    with open(output_path, 'wb') as f:
        # Header
        write_gguf_header(f, n_tensors, n_kv)

        # Key-value pairs
        write_kv_string(f, "general.architecture", arch)
        write_kv_string(f, f"{arch}.model_hint", model_hint)
        write_kv_int32(f, f"{arch}.layer_count", n_tensors)

        # Tensor infos
        for info in tensor_infos:
            write_string(f, info['name'])
            f.write(struct.pack('<I', info['n_dims']))
            for dim in info['dims']:
                f.write(struct.pack('<Q', dim))
            f.write(struct.pack('<I', info['type']))
            f.write(struct.pack('<Q', info['offset']))

        # Alignment padding before tensor data
        current_pos = f.tell()
        alignment = 32
        padding = (alignment - (current_pos % alignment)) % alignment
        f.write(b'\x00' * padding)

        # Tensor data
        for data in tensor_data:
            f.write(data)

    print(f"Created {output_path}")
    print(f"  Tensors: {n_tensors}")
    print(f"  Layers: {sorted(vectors_dict.keys())}")


def main():
    vectors_dir = Path("/Users/naoto/bonsai8b_stearing/vectors")
    output_dir = vectors_dir

    # Japanese v2 vectors
    jp_vectors = {}
    for layer in [10, 12, 15, 18, 20, 22, 25]:
        vec_path = vectors_dir / f"japanese_v2_layer{layer}.npy"
        if vec_path.exists():
            jp_vectors[layer] = np.load(vec_path)
            print(f"Loaded layer {layer}: shape={jp_vectors[layer].shape}, norm={np.linalg.norm(jp_vectors[layer]):.2f}")

    if jp_vectors:
        output_path = output_dir / "japanese_v2.gguf"
        create_control_vector_gguf(jp_vectors, output_path)

    # Cat vectors
    cat_vectors = {}
    for layer in [10, 15, 20, 25, 30]:
        vec_path = vectors_dir / f"cat_layer{layer}.npy"
        if vec_path.exists():
            cat_vectors[layer] = np.load(vec_path)
            print(f"Loaded cat layer {layer}: norm={np.linalg.norm(cat_vectors[layer]):.2f}")

    if cat_vectors:
        output_path = output_dir / "cat.gguf"
        create_control_vector_gguf(cat_vectors, output_path)


if __name__ == "__main__":
    main()
