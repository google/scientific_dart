# gpuarray Feature Roadmap

This document outlines the architectural milestones and feature roadmap for evolving `gpuarray` into a fully-fledged GPU tensor and scientific array computing ecosystem for Dart, targeting parity with **CuPy** (scientific computing) and **PyTorch** (deep learning).

---

## 🗺️ Architectural Tracks

```
                        ┌──────────────────────────────────────────────────────────┐
                        │                   gpuarray Architecture                  │
                        └──────────────────────────────────────────────────────────┘
                                    │                                    │
                                    ▼                                    ▼
                     ┌──────────────────────────────┐     ┌──────────────────────────────┐
                     │      Track A: CuPy Parity    │     │   Track B: PyTorch Parity    │
                     │    (Scientific Computing)    │     │       (Deep Learning)        │
                     └──────────────────────────────┘     └──────────────────────────────┘
                                    │                                    │
     ┌──────────────────────────────┴────────────────────────────────────┴──────────────────────────────┐
     │                                    7 Strategic Milestones                                        │
     ├──────────────────────────────────────────────────────────────────────────────────────────────────┤
     │  Phase 1: Advanced Indexing & Tensor Geometry                                                    │
     │  Phase 2: Hardware Acceleration & JIT Compute Shaders (WebGPU / wgpu / Vulkan / Metal)           │
     │  Phase 3: Extended Data Types & Half Precision (Float16, BFloat16, Complex)                      │
     │  Phase 4: GPU Linear Algebra (Decompositions, Solvers, Einsum)                                   │
     │  Phase 5: GPU Signal Processing & Parallel RNG (FFT, Philox, Box-Muller)                        │
     │  Phase 6: Automatic Differentiation & Computational Graph (Reverse-Mode Autodiff)                │
     │  Phase 7: Neural Network Primitives & Optimizers (Conv2D, LayerNorm, FlashAttention, AdamW)     │
     └──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Phase 1: Advanced Indexing & Tensor Geometry (CuPy Baseline) ✅

Focuses on comprehensive multi-axis tensor manipulation, views, and selection routines matching NumPy/CuPy.

- [x] **Multi-Axis Strided Slicing & Views**:
  - Full support for arbitrary negative strides, slice steps (`a[1:10:2, ::-1]`), and zero-copy subviews.
- [x] **Ellipsis (`...`) Indexing**:
  - Dynamically expanding unspecified middle dimensions in multi-dimensional slices.
- [x] **Boolean Masking & Conditional Selection**:
  - `a[mask]` extraction and `a[mask] = values` masked assignment.
  - `where(condition, x, y)`, `select(condlist, choicelist)`, `extract(condition, arr)`.
- [x] **Fancy (Integer Array) Indexing**:
  - Coordinate index arrays (`a[[0, 2], [1, 3]]`).
  - `take_along_axis(arr, indices, axis)` and `put_along_axis(arr, indices, values, axis)`.
  - `nonzero()`, `flatnonzero()`, and `argwhere()`.
- [x] **Tensor Assembly & Rearrangement**:
  - Concatenation & stacking: `concatenate`, `stack`, `vstack`, `hstack`, `dstack`, `column_stack`.
  - Splitting: `split`, `array_split`, `hsplit`, `vsplit`, `dsplit`.
  - Tiling & repetition: `tile`, `repeat`.
  - Padding & rolling: `pad`, `roll`, `flip`, `rot90`.
  - Diagonal & triangular operations: `diag`, `diagonal`, `trace`, `triu`, `tril`.
  - Axis permutation: `moveaxis`, `swapaxes`, `expand_dims`, `broadcast_to`.

---

## ⚡ Phase 2: Hardware Acceleration & JIT Compute Shaders ✅

Replaces software vector dispatchers with direct GPU hardware compute pipelines.

- [x] **WebGPU / `GpuBackend` Driver Architecture**:
  - Modular `GpuBackend` abstract interface with `CpuVectorBackend` and `WebGpuBackend` driver abstraction.
  - Standardized cross-platform compute backend supporting Linux, macOS, iOS, Windows, and Browser WebGPU.
- [x] **WGSL Compute Shaders & JIT Kernel Fusion Engine**:
  - Dynamic generation of WGSL compute shaders for elementwise operations, workgroup shared-memory tree-reductions, tiled block GEMM ($16 \times 16$), 2D convolution, and matrix transposition.
  - `gpuarray.jit` and `gpuarray.wgsl` kernel fusion engine fusing multi-op chains (e.g. $y = \text{silu}(a \cdot x + b)$) into a single dispatch pass.
- [x] **VRAM Caching Memory Pool (`GpuMemoryPool`)**:
  - Sub-allocating power-of-two size-bucketed pool allocator eliminating driver allocation overhead ($O(1)$ allocation and recycling).
- [x] **Modern Transformer & LLM Building Blocks (`gpuarray.nn`)**:
  - `ScaledDotProductAttention` with causal triangular masking and Autograd backward differentiation.
  - `MultiheadAttention`, `RMSNorm`, `RotaryEmbedding` (`RoPE`), `SwiGLU` / `GeGLU`, and full Pre-LN/Post-LN `TransformerEncoderLayer` / `TransformerDecoderLayer`.

---

## 🔢 Phase 3: Extended Data Types & Half Precision ✅

Expands numerical precision options for high-throughput AI workloads and scientific simulations.

- [x] **Half-Precision Floating Point**:
  - `DType.float16` (IEEE 754 half precision) and `DType.bfloat16` (Brain Floating Point).
  - Native 16-bit math acceleration on tensor cores and modern GPUs.
- [x] **Complex Numbers**:
  - `DType.complex64` ($2 \times \text{float32}$) and `DType.complex128` ($2 \times \text{float64}$).
  - Complex arithmetic, conjugate (`conj()`), magnitude (`abs()`), angle (`angle()`), and complex matrix multiplication.
- [x] **Expanded Integer Types**:
  - `int8`, `uint16`, `uint32`, `uint64`.

---

## 🧮 Phase 4: GPU Linear Algebra & Decompositions (`gpuarray.linalg`) ✅

Hardware-accelerated numerical linear algebra running entirely in VRAM.

- [x] **Matrix Decompositions**:
  - Singular Value Decomposition: `svd(a, {bool fullMatrices = true})`.
  - QR Decomposition: `qr(a, {String mode = 'reduced'})`.
  - Cholesky Decomposition: `cholesky(a, {bool upper = false})`.
  - Eigenvalue & Eigendecomposition: `eig(a)`, `eigh(a)`, `eigvals(a)`, `eigvalsh(a)`.
  - LU Decomposition: `lu(a)`, `lu_factor(a)`, `lu_solve(lu, p, b)`.
- [x] **Matrix Solvers & Inverses**:
  - Linear system solver: `solve(a, b)`.
  - Inverses: `inv(a)` and pseudo-inverse `pinv(a, {double rcond = 1e-15})`.
  - Determinants: `det(a)` and `slogdet(a)`.
  - Matrix norms & condition numbers: `norm(a, {dynamic ord, dynamic axis})`, `cond(a)`.
  - Matrix functions: `matrix_power(a, n)`, `matrix_rank(a)`.
- [x] **Tensor Contractions & Einstein Summation**:
  - `einsum(String subscripts, List<GpuArray> operands)`.
  - `tensordot(a, b, {dynamic axes = 2})`.
  - `kron(a, b)` (Kronecker product), `inner(a, b)`, `outer(a, b)`, `cross(a, b)`, `multi_dot(arrays)`.

---

## 🌊 Phase 5: GPU Signal Processing & Parallel RNG (`gpuarray.fft` & `gpuarray.random`) ✅

High-throughput frequency-domain operations and on-device random generation.

- [x] **Fast Fourier Transforms (`fft`)**:
  - 1D, 2D, and N-D complex FFTs: `fft`, `ifft`, `fft2`, `ifft2`.
  - Real-to-complex FFTs: `rfft`, `irfft`.
  - Frequency utilities: `fftfreq(n, {double d = 1.0})`, `rfftfreq(n, {double d = 1.0})`, `fftshift`, `ifftshift`.
- [x] **Parallel GPU Random Number Generation (`random`)**:
  - Counter-based parallel PRNGs (**Philox4x32-10**) executed on GPU/host.
  - Uniform sampling: `rand(*shape)`, `uniform(low, high, shape)`.
  - Normal distribution sampling: `randn(*shape)`, `normal(loc, scale, shape)` (Box-Muller transform).
  - Discrete & continuous distributions: `randint`, `exponential`.
  - Random permutations & sampling: `choice`, `permutation`, `shuffle`.

---

## 🧠 Phase 6: Automatic Differentiation Engine (PyTorch Parity) ✅

Adds dynamic computational graph tracking and reverse-mode autodiff.

- [x] **Dynamic Computational Graph**:
  - Operation node recording during forward pass (`GradFn`, `AddBackward`, `SubBackward`, `MulBackward`, `DivBackward`, `MatmulBackward`, `SumBackward`, `MeanBackward`).
  - Tensor attributes: `requiresGrad = true`, `grad`, `gradFn`.
  - Graph control: `tensor.detach()`, `tensor.zeroGrad()`, `no_grad()`, `isLeaf`.
- [x] **Reverse-Mode Differentiation (`backward`)**:
  - `tensor.backward([GpuArray? gradient, bool retainGraph = false])`.
  - Automatic gradient accumulation (`tensor.grad += dL/dx`).
  - Unbroadcasting gradient reduction across batch and feature dimensions.

---

## 🤖 Phase 7: Neural Network Primitives & Optimizers (`gpuarray.nn`) ✅

High-performance deep learning layers and training routines.

- [x] **Modules & Sequential Containers**:
  - Base `Module` hierarchy with recursive parameter tracking, `train()`, `eval()`, `zeroGrad()`.
  - `Sequential` chaining pipeline.
- [x] **Neural Network Layers**:
  - `Linear(inFeatures, outFeatures, {hasBias = true})`.
  - `Conv2d(inChannels, outChannels, kernelSize, {stride, padding})`.
  - `LayerNorm(normalizedShape)`.
  - `Dropout(p)`.
  - `Embedding(numEmbeddings, embeddingDim)`.
- [x] **Activation & Loss Functions**:
  - Activations: `relu`, `gelu`, `silu`, `sigmoid`, `tanh`, `softmax`, `log_softmax`.
  - Loss functions: `mse_loss(input, target)`, `cross_entropy(logits, targets)`.
- [x] **GPU Optimizers**:
  - `SGD` with momentum and weight decay.
  - `Adam` (Adaptive Moment Estimation with bias correction).
  - `AdamW` (Decoupled weight decay Adam).

---

## 💾 Phase 8: Model Serialization & SafeTensors (`gpuarray.safetensors`) ✅

Enables zero-copy binary tensor saving and loading in standard AI formats.

- [x] **HuggingFace SafeTensors Format**:
  - `saveSafetensors(Map<String, GpuArray> tensors, {metadata})`.
  - `loadSafetensors(Uint8List bytes, {device})`.
  - `saveSafetensorsFile(filePath, tensors)` and `loadSafetensorsFile(filePath)`.
  - Full support across all 15 data types without precision loss.
