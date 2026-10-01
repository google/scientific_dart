// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import 'dart:typed_data';

import '../autograd/autograd_wgsl.dart';
import '../backend/wgsl/wgsl_types.dart';
import '../buffer.dart';
import '../dtype.dart';
import '../gpu_array.dart';

/// Dispatches the `Embedding.forward` row-gather kernel on the GPU, validating index bounds.
void dispatchEmbeddingForward({
  required GpuArray<DTypeTag> weight,
  required GpuArray<DTypeTag> indices,
  required GpuArray<DTypeTag> output,
  required int numEmbeddings,
  required int embeddingDim,
}) {
  final indexCount = indices.size;
  final totalElements = indexCount * embeddingDim;
  if (totalElements == 0) return;

  final contiguousWeight = weight.isContiguous ? weight : weight.copy();
  final contiguousIndices = indices.isContiguous ? indices : indices.copy();
  final statusBuffer = GpuBuffer.allocate(sizeInBytes: 8, device: weight.device)
    ..clear();

  try {
    final bindings = [
      storageBinding(
        0,
        'weight_buf',
        contiguousWeight.dtype,
        WgslBufferAccess.read,
      ),
      storageBinding(
        1,
        'indices_buf',
        contiguousIndices.dtype,
        WgslBufferAccess.read,
      ),
      storageBinding(2, 'out_buf', output.dtype, WgslBufferAccess.readWrite),
      const WgslBinding(
        group: 0,
        binding: 3,
        name: 'status_buf',
        access: WgslBufferAccess.readWrite,
        customTypeName: 'array<atomic<i32>>',
      ),
      const WgslBinding(
        group: 0,
        binding: 4,
        name: 'uniforms',
        isUniform: true,
        customTypeName: 'EmbeddingForwardUniforms',
      ),
    ];

    final loadIndexFn = wgslLoadIndex(
      contiguousIndices.dtype,
      'indices_buf',
      'load_index',
    );
    final loadWeightFn = wgslLoadFloat(
      contiguousWeight.dtype,
      'weight_buf',
      'load_weight',
    );
    final storeOutFn = wgslStoreFloat(output.dtype, 'out_buf', 'store_out');
    final copyElementStmt =
        (contiguousWeight.dtype == DType.float64 &&
            output.dtype == DType.float64)
        ? 'out_buf[thread_index] = weight_buf[weight_index];'
        : 'store_out(thread_index, load_weight(weight_index));';

    final code =
        '''
$wgslF64ConversionHelpers
struct EmbeddingForwardUniforms {
  total_elements: u32, num_embeddings: u32, embedding_dim: u32, weight_offset: u32,
  indices_offset: u32, pad0: u32, pad1: u32, pad2: u32,
}
${bindings.map((b) => b.toWgslDeclaration()).join('\n')}
$loadIndexFn
$loadWeightFn
$storeOutFn
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>) {
  let thread_index = global_id.x + global_id.y * (num_wg.x * 256u);
  if (thread_index >= uniforms.total_elements) { return; }
  let item_index = thread_index / uniforms.embedding_dim;
  let dim_index = thread_index % uniforms.embedding_dim;
  let token_index = load_index(uniforms.indices_offset + item_index);
  if (token_index < 0 || token_index >= i32(uniforms.num_embeddings)) {
    let prev = atomicCompareExchangeWeak(&status_buf[0], 0, 1);
    if (prev.exchanged) { atomicStore(&status_buf[1], token_index); }
    return;
  }
  let weight_index = uniforms.weight_offset + u32(token_index) * uniforms.embedding_dim + dim_index;
  $copyElementStmt
}
''';

    dispatch1DKernel(
      device: weight.device,
      name:
          'nn_embedding_forward_${contiguousWeight.dtype.name}_${contiguousIndices.dtype.name}',
      code: code,
      bindings: bindings,
      buffers: [
        contiguousWeight.buffer,
        contiguousIndices.buffer,
        output.buffer,
        statusBuffer,
      ],
      uniforms: [
        totalElements,
        numEmbeddings,
        embeddingDim,
        contiguousWeight.offsetElements,
        contiguousIndices.offsetElements,
        0,
        0,
        0,
      ],
      totalElements: totalElements,
    );

    final statusBytes = statusBuffer.readBytes();
    final statusView = ByteData.sublistView(statusBytes);
    if (statusView.getInt32(0, Endian.little) != 0) {
      final offendingIndex = statusView.getInt32(4, Endian.little);
      output.dispose();
      RangeError.checkValueInInterval(
        offendingIndex,
        0,
        numEmbeddings - 1,
        'indices',
      );
    }
  } finally {
    statusBuffer.dispose();
    if (!identical(contiguousWeight, weight)) {
      contiguousWeight.dispose();
    }
    if (!identical(contiguousIndices, indices)) {
      contiguousIndices.dispose();
    }
  }
}

/// Dispatches the `EmbeddingBackward` row scatter-add kernel on the GPU.
void dispatchEmbeddingBackward({
  required GpuArray<DTypeTag> gradOutput,
  required GpuArray<DTypeTag> indices,
  required GpuArray<DTypeTag> gradWeight,
  required int numEmbeddings,
  required int embeddingDim,
}) {
  final indexCount = indices.size;
  final totalElements = numEmbeddings * embeddingDim;
  if (totalElements == 0 || indexCount == 0) return;

  final contiguousGrad = gradOutput.isContiguous
      ? gradOutput
      : gradOutput.copy();
  final contiguousIndices = indices.isContiguous ? indices : indices.copy();

  try {
    final bindings = [
      storageBinding(
        0,
        'grad_out_buf',
        contiguousGrad.dtype,
        WgslBufferAccess.read,
      ),
      storageBinding(
        1,
        'indices_buf',
        contiguousIndices.dtype,
        WgslBufferAccess.read,
      ),
      storageBinding(
        2,
        'grad_weight_buf',
        gradWeight.dtype,
        WgslBufferAccess.readWrite,
      ),
      const WgslBinding(
        group: 0,
        binding: 3,
        name: 'uniforms',
        isUniform: true,
        customTypeName: 'EmbeddingBackwardUniforms',
      ),
    ];

    final loadGradFn = wgslLoadFloat(
      contiguousGrad.dtype,
      'grad_out_buf',
      'load_grad',
    );
    final loadIndexFn = wgslLoadIndex(
      contiguousIndices.dtype,
      'indices_buf',
      'load_index',
    );
    final storeWeightFn = wgslStoreFloat(
      gradWeight.dtype,
      'grad_weight_buf',
      'store_grad_weight',
    );

    final code =
        '''
$wgslF64ConversionHelpers
struct EmbeddingBackwardUniforms {
  total_elements: u32, index_count: u32, num_embeddings: u32, embedding_dim: u32,
  grad_offset: u32, indices_offset: u32, pad0: u32, pad1: u32,
}
${bindings.map((b) => b.toWgslDeclaration()).join('\n')}
$loadGradFn
$loadIndexFn
$storeWeightFn
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>) {
  let thread_index = global_id.x + global_id.y * (num_wg.x * 256u);
  if (thread_index >= uniforms.total_elements) { return; }
  let row = thread_index / uniforms.embedding_dim;
  let col = thread_index % uniforms.embedding_dim;
  var acc: f32 = 0.0;
  for (var i: u32 = 0u; i < uniforms.index_count; i = i + 1u) {
    if (load_index(uniforms.indices_offset + i) == i32(row)) {
      acc = acc + load_grad(uniforms.grad_offset + i * uniforms.embedding_dim + col);
    }
  }
  store_grad_weight(thread_index, acc);
}
''';

    dispatch1DKernel(
      device: gradWeight.device,
      name:
          'autograd_embedding_backward_${gradWeight.dtype.name}_${contiguousIndices.dtype.name}',
      code: code,
      bindings: bindings,
      buffers: [
        contiguousGrad.buffer,
        contiguousIndices.buffer,
        gradWeight.buffer,
      ],
      uniforms: [
        totalElements,
        indexCount,
        numEmbeddings,
        embeddingDim,
        contiguousGrad.offsetElements,
        contiguousIndices.offsetElements,
        0,
        0,
      ],
      totalElements: totalElements,
    );
  } finally {
    if (!identical(contiguousGrad, gradOutput)) {
      contiguousGrad.dispose();
    }
    if (!identical(contiguousIndices, indices)) {
      contiguousIndices.dispose();
    }
  }
}

/// Dispatches an upper-triangular causal attention mask kernel (`col > row ? -1e9 : 0.0`) on the GPU.
void dispatchCausalMask({
  required GpuArray<DTypeTag> causalMask,
  required int querySeqLength,
  required int keySeqLength,
}) {
  final totalElements = querySeqLength * keySeqLength;
  if (totalElements == 0) return;

  final bindings = [
    storageBinding(0, 'mask_buf', causalMask.dtype, WgslBufferAccess.readWrite),
    const WgslBinding(
      group: 0,
      binding: 1,
      name: 'uniforms',
      isUniform: true,
      customTypeName: 'CausalMaskUniforms',
    ),
  ];
  final storeMaskFn = wgslStoreFloat(
    causalMask.dtype,
    'mask_buf',
    'store_mask',
  );

  final code =
      '''
$wgslF64ConversionHelpers
struct CausalMaskUniforms {
  total_elements: u32, query_len: u32, key_len: u32, pad0: u32,
}
${bindings.map((b) => b.toWgslDeclaration()).join('\n')}
$storeMaskFn
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>) {
  let thread_index = global_id.x + global_id.y * (num_wg.x * 256u);
  if (thread_index >= uniforms.total_elements) { return; }
  let row = thread_index / uniforms.key_len;
  let col = thread_index % uniforms.key_len;
  store_mask(thread_index, select(0.0, -1e9, col > row));
}
''';

  dispatch1DKernel(
    device: causalMask.device,
    name: 'nn_causal_mask_${causalMask.dtype.name}',
    code: code,
    bindings: bindings,
    buffers: [causalMask.buffer],
    uniforms: [totalElements, querySeqLength, keySeqLength, 0],
    totalElements: totalElements,
  );
}

/// Dispatches a boolean-to-additive attention mask conversion kernel (`keep ? 0.0 : -1e9`) on the GPU.
void dispatchBooleanAdditiveMask({
  required GpuArray<DTypeTag> boolMask,
  required GpuArray<DTypeTag> additiveMask,
}) {
  final totalElements = additiveMask.size;
  if (totalElements == 0) return;

  final contiguousBool = boolMask.isContiguous ? boolMask : boolMask.copy();

  try {
    final bindings = [
      storageBinding(
        0,
        'bool_buf',
        contiguousBool.dtype,
        WgslBufferAccess.read,
      ),
      storageBinding(
        1,
        'mask_buf',
        additiveMask.dtype,
        WgslBufferAccess.readWrite,
      ),
      const WgslBinding(
        group: 0,
        binding: 2,
        name: 'uniforms',
        isUniform: true,
        customTypeName: 'BoolMaskUniforms',
      ),
    ];

    final loadBoolFn = wgslLoadIndex(
      contiguousBool.dtype,
      'bool_buf',
      'load_bool',
    );
    final storeMaskFn = wgslStoreFloat(
      additiveMask.dtype,
      'mask_buf',
      'store_mask',
    );

    final code =
        '''
$wgslF64ConversionHelpers
struct BoolMaskUniforms {
  total_elements: u32, bool_offset: u32, pad0: u32, pad1: u32,
}
${bindings.map((b) => b.toWgslDeclaration()).join('\n')}
$loadBoolFn
$storeMaskFn
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>) {
  let thread_index = global_id.x + global_id.y * (num_wg.x * 256u);
  if (thread_index >= uniforms.total_elements) { return; }
  let keep = load_bool(uniforms.bool_offset + thread_index) != 0;
  store_mask(thread_index, select(-1e9, 0.0, keep));
}
''';

    dispatch1DKernel(
      device: additiveMask.device,
      name: 'nn_bool_additive_mask_${additiveMask.dtype.name}',
      code: code,
      bindings: bindings,
      buffers: [contiguousBool.buffer, additiveMask.buffer],
      uniforms: [totalElements, contiguousBool.offsetElements, 0, 0],
      totalElements: totalElements,
    );
  } finally {
    if (!identical(contiguousBool, boolMask)) {
      contiguousBool.dispose();
    }
  }
}

/// Dispatches the `RotaryEmbedding` cosine and sine table precomputation kernel on the GPU.
void dispatchRotaryEmbeddingCache({
  required GpuArray<DTypeTag> cosCached,
  required GpuArray<DTypeTag> sinCached,
  required int maxSequenceLength,
  required int dim,
  required double base,
}) {
  final halfDim = dim ~/ 2;
  final totalPairs = maxSequenceLength * halfDim;
  if (totalPairs == 0) return;

  final bindings = [
    storageBinding(0, 'cos_buf', cosCached.dtype, WgslBufferAccess.readWrite),
    storageBinding(1, 'sin_buf', sinCached.dtype, WgslBufferAccess.readWrite),
    const WgslBinding(
      group: 0,
      binding: 2,
      name: 'uniforms',
      isUniform: true,
      customTypeName: 'RopeCacheUniforms',
    ),
  ];
  final storeCosFn = wgslStoreFloat(cosCached.dtype, 'cos_buf', 'store_cos');
  final storeSinFn = wgslStoreFloat(sinCached.dtype, 'sin_buf', 'store_sin');

  final code =
      '''
$wgslF64ConversionHelpers
struct RopeCacheUniforms {
  total_pairs: u32, dim: u32, half_dim: u32, base_bits: u32,
}
${bindings.map((b) => b.toWgslDeclaration()).join('\n')}
$storeCosFn
$storeSinFn
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>) {
  let thread_index = global_id.x + global_id.y * (num_wg.x * 256u);
  if (thread_index >= uniforms.total_pairs) { return; }
  let position = thread_index / uniforms.half_dim;
  let freq_index = thread_index % uniforms.half_dim;
  let base_val = bitcast<f32>(uniforms.base_bits);
  let inv_freq = 1.0 / pow(base_val, (2.0 * f32(freq_index)) / f32(uniforms.dim));
  let theta = f32(position) * inv_freq;
  let cos_val = cos(theta);
  let sin_val = sin(theta);
  let row_base = position * uniforms.dim;
  store_cos(row_base + freq_index, cos_val);
  store_cos(row_base + uniforms.half_dim + freq_index, cos_val);
  store_sin(row_base + freq_index, sin_val);
  store_sin(row_base + uniforms.half_dim + freq_index, sin_val);
}
''';

  dispatch1DKernel(
    device: cosCached.device,
    name: 'nn_rope_cache_${cosCached.dtype.name}',
    code: code,
    bindings: bindings,
    buffers: [cosCached.buffer, sinCached.buffer],
    uniforms: [totalPairs, dim, halfDim, float32ToBits(base)],
    totalElements: totalPairs,
  );
}
