# `ndarray_transformer`

> [!WARNING]
> This package is **experimental** and serves as an end-to-end showcase application stress-testing `package:ndarray` on real transformer inference.

Pure-Dart BERT / `sentence-transformers/all-MiniLM-L6-v2` sentence-embedding and semantic-search engine built on `package:ndarray`. Loads standard Hugging Face `.safetensors` checkpoints directly into `NDArray<Float32>` native memory—no Python runtime, ONNX Runtime, or TFLite dependency required.

## What It Includes

- **[`SafetensorsFile`](lib/src/safetensors.dart)**: Spec-compliant `.safetensors` reader and writer with bounds/overlap validation, lazy on-disk slice loading via `RandomAccessFile`, and bulk native copies via `NDArray.fromBytes` / `NDArray.toBytes` (auto-casting `F16`/`BF16`/`F64` to `Float32` on demand).
- **[`WordPieceTokenizer`](lib/src/wordpiece_tokenizer.dart)**: Pure-Dart BERT Basic + WordPiece subword tokenizer loading Hugging Face `tokenizer.json` or `vocab.txt`, including CJK spacing, punctuation splitting, Latin-1 + combining-diacritic accent folding, sentence pairs, and batch padding with attention masks.
- **[`BertEncoder`](lib/src/bert_model.dart)**: Full `Float32` BERT encoder (embeddings, LayerNorm, batched multi-head self-attention via `matmul`, exact `erf`-based GELU, attention-masked mean pooling, and L2 normalization) with per-layer `NDArray.scope` disposal so peak native memory stays $O(1)$ in layer depth.
- **[`SemanticSearchIndex`](lib/src/semantic_search.dart)**: Cosine-similarity ranking over L2-normalized `NDArray<Float32>` matrices using `matmul` and `argsort`.

## Quick Start

### 1. Offline Synthetic Demo (Zero Downloads)

Run the end-to-end `.safetensors` encode/load → WordPiece → BERT forward pass → semantic search CLI completely offline:

```bash
dart run pkgs/ndarray_transformer/bin/embed.dart --demo-synthetic --benchmark
```

### 2. Real `all-MiniLM-L6-v2` Weights (Hugging Face)

Download `sentence-transformers/all-MiniLM-L6-v2` (~87 MiB) once and run real semantic search:

```bash
dart run pkgs/ndarray_transformer/bin/embed.dart \
  --download \
  --query "vector similarity search with neural embeddings" \
  --benchmark
```

Or compile to a standalone native AOT bundle for peak throughput:

```bash
cd pkgs/ndarray_transformer
dart build cli --target bin/embed.dart --output build/embed_cli
./build/embed_cli/bundle/bin/embed --model-dir ~/.cache/scientific_dart/all-MiniLM-L6-v2 --benchmark
```

## `package:ndarray` Ergonomics & Primitive Findings

Building a real `Float32` neural network on `ndarray 0.4.0-wip` confirmed what works well (`NDArray.fromBytes` / `fromTypedData`, `NDArray.scope`, native `erf` on `Float32`, batched 4D `matmul` on transposed views, `take`, `argsort`) and surfaced concrete ergonomic/performance gaps worth addressing in `ndarray`:

1. **`mean`, `variance`, and `std` unconditionally promote `Float32` to `Float64`**: Because `mean` uses `DoublePrecisionOf<R>` and `variance`/`std` return `NDArray<Float64>`, implementing pure-`Float32` LayerNorm had to express `mean` and `variance` manually via `sum(x, axis: -1, keepDims: true) * (1.0 / H)` (since `sum` preserves `Float32` via `AccumulatorOf`). Adding `Float32`-preserving `mean`/`variance`/`std` (or `meanAs`/`varianceAs`) would eliminate boilerplate across ML workloads.
2. **No `softmax` / `logSumExp` / `layerNorm` primitives**: Softmax currently composes 4 separate allocations/passes (`max`, `-`, `exp`, `sum`, `/`) per attention layer. A fused native `logSumExp` / `softmax` (or `layerNorm`) op would cut intermediate allocations and memory bandwidth in half.
3. **No `NDArray.take` accepting `NDArray<Int64>` indices directly with multi-dimensional output shape**: Embedding lookup (`[V, H]` indexed by `[B, S]` token IDs) currently flattens token IDs to a Dart `List<int>`, calls `wordEmbeddings.take(flatIds, axis: 0)`, and reshapes to `[B, S, H]`. Supporting `take(NDArray<IntegerDType> indices, {int axis = 0})` preserving `indices.shape` would match NumPy/PyTorch embedding gathers directly.
4. **No `topK` / `argpartition` descending helper**: Retrieving top-`k` cosine similarity hits currently runs a full `argsort` ($O(N \log N)$) and reads from the tail instead of an $O(N)$ descending `topK` selection.
5. **No `safetensors` I/O in `package:ndarray`**: `NDArray.fromBytes` made writing a spec-compliant `.safetensors` loader straightforward (~280 LOC) with zero intermediate Dart list copies.
6. **Bug caught & fixed — `_withWrappedScalar` allocation-tracker leak**: Running `bert_encoder_test.dart` with `-DTRACK_NDARRAY_ALLOCATIONS=true` surfaced that `NDArray._withWrappedScalar` created a `ScratchArena`-backed scalar wrapper via `NDArray.fromPointer` inside `ResourceScope.unmanaged` without calling `otherArr.dispose()` before `ScratchArena.reset`, leaking every scalar operator wrapper into `ResourceScope._trackedAllocations` with a dangling arena pointer. Fixed in `pkgs/ndarray/lib/src/ndarray.dart` by disposing `otherArr` in `finally`.

## Running Tests

```bash
dart test pkgs/ndarray_transformer
```

`test/bert_encoder_test.dart` verifies exact numerical parity (max absolute error `< 1e-5`) of both token hidden states and pooled embeddings against an independent Python/NumPy BERT reference implementation (when `python3` with `numpy` is available), and checks zero native memory leaks via `NDArray.checkNoLeaks()`.

## License

This project is licensed under the **[Apache License, Version 2.0](LICENSE)**.

## Disclaimer

This is not an official Google product.
