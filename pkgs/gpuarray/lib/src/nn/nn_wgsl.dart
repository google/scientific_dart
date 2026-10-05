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

String _activationForwardExpr(String op) => switch (op) {
  'relu' => 'let y = max(x, 0.0);',
  'sigmoid' => 'let y = 1.0 / (1.0 + exp(-x));',
  'tanh' => 'let y = tanh(clamp(x, -15.0, 15.0));',
  'gelu' =>
    '''
  let k0 = 0.7978845608028654;
  let k1 = 0.044715;
  let inner = clamp(k0 * (x + k1 * x * x * x), -15.0, 15.0);
  let y = 0.5 * x * (1.0 + tanh(inner));''',
  'silu' || 'swish' => 'let y = x / (1.0 + exp(-x));',
  'leaky_relu' =>
    '''
  let slope = bitcast<f32>(uniforms.param0_bits);
  let y = select(slope * x, x, x > 0.0);''',
  'elu' =>
    '''
  let alpha = bitcast<f32>(uniforms.param0_bits);
  let y = select(alpha * (exp(x) - 1.0), x, x > 0.0);''',
  'softplus' =>
    '''
  let beta = bitcast<f32>(uniforms.param0_bits);
  let thresh = bitcast<f32>(uniforms.param1_bits);
  let bx = beta * x;
  let y = select(log(1.0 + exp(bx)) / beta, x, bx > thresh);''',
  _ => throw ArgumentError.value(op, 'op', 'Must be a supported activation.'),
};

String _activationBackwardExpr(String op) => switch (op) {
  'relu' => 'let grad = select(0.0, go, s_val > 0.0);',
  'sigmoid' =>
    '''
  let sig = 1.0 / (1.0 + exp(-s_val));
  let grad = go * sig * (1.0 - sig);''',
  'tanh' =>
    '''
  let t_val = tanh(clamp(s_val, -15.0, 15.0));
  let grad = go * (1.0 - t_val * t_val);''',
  'gelu' =>
    '''
  let k0 = 0.7978845608028654;
  let k1 = 0.044715;
  let x2 = s_val * s_val;
  let x3 = x2 * s_val;
  let inner = clamp(k0 * (s_val + k1 * x3), -15.0, 15.0);
  let t_val = tanh(inner);
  let sech2 = 1.0 - t_val * t_val;
  let d_inner = k0 * (1.0 + 3.0 * k1 * x2);
  let d_gelu = 0.5 * (1.0 + t_val) + 0.5 * s_val * sech2 * d_inner;
  let grad = go * d_gelu;''',
  'silu' || 'swish' =>
    '''
  let sig = 1.0 / (1.0 + exp(-s_val));
  let d_silu = sig * (1.0 + s_val * (1.0 - sig));
  let grad = go * d_silu;''',
  'leaky_relu' =>
    '''
  let slope = bitcast<f32>(uniforms.param0_bits);
  let grad = go * select(slope, 1.0, s_val > 0.0);''',
  'elu' =>
    '''
  let alpha = bitcast<f32>(uniforms.param0_bits);
  let grad = go * select(alpha * exp(s_val), 1.0, s_val > 0.0);''',
  'softplus' =>
    '''
  let beta = bitcast<f32>(uniforms.param0_bits);
  let thresh = bitcast<f32>(uniforms.param1_bits);
  let bx = beta * s_val;
  let d_sp = select(1.0 / (1.0 + exp(-bx)), 1.0, bx > thresh);
  let grad = go * d_sp;''',
  _ => throw ArgumentError.value(op, 'op', 'Must be a supported activation.'),
};

/// Dispatches a single-pass 1D elementwise activation forward kernel on the GPU.
void dispatchUnaryActivationForward({
  required GpuArray<DTypeTag> input,
  required GpuArray<DTypeTag> output,
  required String op,
  double param0 = 0.0,
  double param1 = 0.0,
}) {
  final totalElements = output.size;
  if (totalElements == 0) return;

  final contiguousInput = input.isContiguous ? input : input.copy();
  try {
    final exprWgsl = _activationForwardExpr(op);
    final isInPlace =
        identical(contiguousInput.buffer, output.buffer) &&
        contiguousInput.dtype == output.dtype;

    if (isInPlace) {
      final bindings = [
        storageBinding(0, 'io_buf', output.dtype, WgslBufferAccess.readWrite),
        const WgslBinding(
          group: 0,
          binding: 1,
          name: 'uniforms',
          isUniform: true,
          customTypeName: 'UnaryActivationUniforms',
        ),
      ];
      final loadFn = wgslLoadFloat(output.dtype, 'io_buf', 'load_io');
      final storeFn = wgslStoreFloat(output.dtype, 'io_buf', 'store_io');
      final code =
          '''
$wgslF64ConversionHelpers
struct UnaryActivationUniforms {
  total_elements: u32, in_offset: u32, out_offset: u32, param0_bits: u32,
  param1_bits: u32, pad0: u32, pad1: u32, pad2: u32,
}
${bindings.map((b) => b.toWgslDeclaration()).join('\n')}
$loadFn
$storeFn
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>) {
  let thread_index = global_id.x + global_id.y * (num_wg.x * 256u);
  if (thread_index >= uniforms.total_elements) { return; }
  let x = load_io(uniforms.in_offset + thread_index);
  $exprWgsl
  store_io(uniforms.out_offset + thread_index, y);
}
''';
      dispatch1DKernel(
        device: output.device,
        name: 'nn_${op}_inplace_${output.dtype.name}',
        code: code,
        bindings: bindings,
        buffers: [output.buffer],
        uniforms: [
          totalElements,
          contiguousInput.offsetElements,
          output.offsetElements,
          float32ToBits(param0),
          float32ToBits(param1),
          0,
          0,
          0,
        ],
        totalElements: totalElements,
      );
      return;
    }

    final bindings = [
      storageBinding(0, 'in_buf', contiguousInput.dtype, WgslBufferAccess.read),
      storageBinding(1, 'out_buf', output.dtype, WgslBufferAccess.readWrite),
      const WgslBinding(
        group: 0,
        binding: 2,
        name: 'uniforms',
        isUniform: true,
        customTypeName: 'UnaryActivationUniforms',
      ),
    ];

    final loadInFn = wgslLoadFloat(contiguousInput.dtype, 'in_buf', 'load_in');
    final storeOutFn = wgslStoreFloat(output.dtype, 'out_buf', 'store_out');

    final code =
        '''
$wgslF64ConversionHelpers
struct UnaryActivationUniforms {
  total_elements: u32, in_offset: u32, out_offset: u32, param0_bits: u32,
  param1_bits: u32, pad0: u32, pad1: u32, pad2: u32,
}
${bindings.map((b) => b.toWgslDeclaration()).join('\n')}
$loadInFn
$storeOutFn
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>) {
  let thread_index = global_id.x + global_id.y * (num_wg.x * 256u);
  if (thread_index >= uniforms.total_elements) { return; }
  let x = load_in(uniforms.in_offset + thread_index);
  $exprWgsl
  store_out(uniforms.out_offset + thread_index, y);
}
''';

    dispatch1DKernel(
      device: output.device,
      name: 'nn_${op}_forward_${output.dtype.name}',
      code: code,
      bindings: bindings,
      buffers: [contiguousInput.buffer, output.buffer],
      uniforms: [
        totalElements,
        contiguousInput.offsetElements,
        output.offsetElements,
        float32ToBits(param0),
        float32ToBits(param1),
        0,
        0,
        0,
      ],
      totalElements: totalElements,
    );
  } finally {
    if (!identical(contiguousInput, input)) {
      contiguousInput.dispose();
    }
  }
}

/// Dispatches a single-pass 1D elementwise activation backward kernel on the GPU.
void dispatchUnaryActivationBackward({
  required GpuArray<DTypeTag> gradOutput,
  required GpuArray<DTypeTag> savedTensor,
  required GpuArray<DTypeTag> gradInput,
  required String op,
  double param0 = 0.0,
  double param1 = 0.0,
}) {
  final totalElements = gradInput.size;
  if (totalElements == 0) return;

  final contiguousGradOut = gradOutput.isContiguous
      ? gradOutput
      : gradOutput.copy();
  final contiguousSaved = savedTensor.isContiguous
      ? savedTensor
      : savedTensor.copy();

  try {
    final exprWgsl = _activationBackwardExpr(op);
    final bindings = [
      storageBinding(
        0,
        'grad_out_buf',
        contiguousGradOut.dtype,
        WgslBufferAccess.read,
      ),
      storageBinding(
        1,
        'saved_buf',
        contiguousSaved.dtype,
        WgslBufferAccess.read,
      ),
      storageBinding(
        2,
        'grad_in_buf',
        gradInput.dtype,
        WgslBufferAccess.readWrite,
      ),
      const WgslBinding(
        group: 0,
        binding: 3,
        name: 'uniforms',
        isUniform: true,
        customTypeName: 'UnaryActivationBackwardUniforms',
      ),
    ];

    final loadGradOutFn = wgslLoadFloat(
      contiguousGradOut.dtype,
      'grad_out_buf',
      'load_grad_out',
    );
    final loadSavedFn = wgslLoadFloat(
      contiguousSaved.dtype,
      'saved_buf',
      'load_saved',
    );
    final storeGradInFn = wgslStoreFloat(
      gradInput.dtype,
      'grad_in_buf',
      'store_grad_in',
    );

    final code =
        '''
$wgslF64ConversionHelpers
struct UnaryActivationBackwardUniforms {
  total_elements: u32, grad_out_offset: u32, saved_offset: u32, grad_in_offset: u32,
  param0_bits: u32, param1_bits: u32, pad0: u32, pad1: u32,
}
${bindings.map((b) => b.toWgslDeclaration()).join('\n')}
$loadGradOutFn
$loadSavedFn
$storeGradInFn
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>) {
  let thread_index = global_id.x + global_id.y * (num_wg.x * 256u);
  if (thread_index >= uniforms.total_elements) { return; }
  let go = load_grad_out(uniforms.grad_out_offset + thread_index);
  let s_val = load_saved(uniforms.saved_offset + thread_index);
  $exprWgsl
  store_grad_in(uniforms.grad_in_offset + thread_index, grad);
}
''';

    dispatch1DKernel(
      device: gradInput.device,
      name: 'autograd_${op}_backward_${gradInput.dtype.name}',
      code: code,
      bindings: bindings,
      buffers: [
        contiguousGradOut.buffer,
        contiguousSaved.buffer,
        gradInput.buffer,
      ],
      uniforms: [
        totalElements,
        contiguousGradOut.offsetElements,
        contiguousSaved.offsetElements,
        gradInput.offsetElements,
        float32ToBits(param0),
        float32ToBits(param1),
        0,
        0,
      ],
      totalElements: totalElements,
    );
  } finally {
    if (!identical(contiguousGradOut, gradOutput)) {
      contiguousGradOut.dispose();
    }
    if (!identical(contiguousSaved, savedTensor)) {
      contiguousSaved.dispose();
    }
  }
}

/// Dispatches an in-place fused `SGD.step` update kernel on the GPU.
void dispatchSgdStep({
  required GpuArray<DTypeTag> parameter,
  required GpuArray<DTypeTag> grad,
  GpuArray<DTypeTag>? velocity,
  required double lr,
  required double momentum,
  required double weightDecay,
  required bool nesterov,
}) {
  final totalElements = parameter.size;
  if (totalElements == 0) return;

  final contiguousGrad = grad.isContiguous ? grad : grad.copy();
  try {
    if (velocity == null) {
      final bindings = [
        storageBinding(
          0,
          'param_buf',
          parameter.dtype,
          WgslBufferAccess.readWrite,
        ),
        storageBinding(
          1,
          'grad_buf',
          contiguousGrad.dtype,
          WgslBufferAccess.read,
        ),
        const WgslBinding(
          group: 0,
          binding: 2,
          name: 'uniforms',
          isUniform: true,
          customTypeName: 'SgdSimpleUniforms',
        ),
      ];
      final loadParamFn = wgslLoadFloat(
        parameter.dtype,
        'param_buf',
        'load_param',
      );
      final storeParamFn = wgslStoreFloat(
        parameter.dtype,
        'param_buf',
        'store_param',
      );
      final loadGradFn = wgslLoadFloat(
        contiguousGrad.dtype,
        'grad_buf',
        'load_grad',
      );

      final code =
          '''
$wgslF64ConversionHelpers
struct SgdSimpleUniforms {
  total_elements: u32, param_offset: u32, grad_offset: u32, lr_bits: u32,
  weight_decay_bits: u32, pad0: u32, pad1: u32, pad2: u32,
}
${bindings.map((b) => b.toWgslDeclaration()).join('\n')}
$loadParamFn
$storeParamFn
$loadGradFn
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>) {
  let thread_index = global_id.x + global_id.y * (num_wg.x * 256u);
  if (thread_index >= uniforms.total_elements) { return; }
  let p = load_param(uniforms.param_offset + thread_index);
  var g = load_grad(uniforms.grad_offset + thread_index);
  let wd = bitcast<f32>(uniforms.weight_decay_bits);
  if (wd != 0.0) {
    g = g + wd * p;
  }
  let lr_val = bitcast<f32>(uniforms.lr_bits);
  store_param(uniforms.param_offset + thread_index, p - lr_val * g);
}
''';

      dispatch1DKernel(
        device: parameter.device,
        name:
            'nn_sgd_simple_${parameter.dtype.name}_${contiguousGrad.dtype.name}',
        code: code,
        bindings: bindings,
        buffers: [parameter.buffer, contiguousGrad.buffer],
        uniforms: [
          totalElements,
          parameter.offsetElements,
          contiguousGrad.offsetElements,
          float32ToBits(lr),
          float32ToBits(weightDecay),
          0,
          0,
          0,
        ],
        totalElements: totalElements,
      );
    } else {
      final bindings = [
        storageBinding(
          0,
          'param_buf',
          parameter.dtype,
          WgslBufferAccess.readWrite,
        ),
        storageBinding(
          1,
          'grad_buf',
          contiguousGrad.dtype,
          WgslBufferAccess.read,
        ),
        storageBinding(
          2,
          'vel_buf',
          velocity.dtype,
          WgslBufferAccess.readWrite,
        ),
        const WgslBinding(
          group: 0,
          binding: 3,
          name: 'uniforms',
          isUniform: true,
          customTypeName: 'SgdMomentumUniforms',
        ),
      ];
      final loadParamFn = wgslLoadFloat(
        parameter.dtype,
        'param_buf',
        'load_param',
      );
      final storeParamFn = wgslStoreFloat(
        parameter.dtype,
        'param_buf',
        'store_param',
      );
      final loadGradFn = wgslLoadFloat(
        contiguousGrad.dtype,
        'grad_buf',
        'load_grad',
      );
      final loadVelFn = wgslLoadFloat(velocity.dtype, 'vel_buf', 'load_vel');
      final storeVelFn = wgslStoreFloat(velocity.dtype, 'vel_buf', 'store_vel');

      final code =
          '''
$wgslF64ConversionHelpers
struct SgdMomentumUniforms {
  total_elements: u32, param_offset: u32, grad_offset: u32, lr_bits: u32,
  momentum_bits: u32, weight_decay_bits: u32, nesterov: u32, pad0: u32,
}
${bindings.map((b) => b.toWgslDeclaration()).join('\n')}
$loadParamFn
$storeParamFn
$loadGradFn
$loadVelFn
$storeVelFn
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>) {
  let thread_index = global_id.x + global_id.y * (num_wg.x * 256u);
  if (thread_index >= uniforms.total_elements) { return; }
  let p = load_param(uniforms.param_offset + thread_index);
  var g = load_grad(uniforms.grad_offset + thread_index);
  let wd = bitcast<f32>(uniforms.weight_decay_bits);
  if (wd != 0.0) {
    g = g + wd * p;
  }
  let mom = bitcast<f32>(uniforms.momentum_bits);
  let v_prev = load_vel(thread_index);
  let v_next = mom * v_prev + g;
  store_vel(thread_index, v_next);
  let step_g = select(v_next, g + mom * v_next, uniforms.nesterov != 0u);
  let lr_val = bitcast<f32>(uniforms.lr_bits);
  store_param(uniforms.param_offset + thread_index, p - lr_val * step_g);
}
''';

      dispatch1DKernel(
        device: parameter.device,
        name:
            'nn_sgd_momentum_${parameter.dtype.name}_${contiguousGrad.dtype.name}',
        code: code,
        bindings: bindings,
        buffers: [parameter.buffer, contiguousGrad.buffer, velocity.buffer],
        uniforms: [
          totalElements,
          parameter.offsetElements,
          contiguousGrad.offsetElements,
          float32ToBits(lr),
          float32ToBits(momentum),
          float32ToBits(weightDecay),
          nesterov ? 1 : 0,
          0,
        ],
        totalElements: totalElements,
      );
    }
  } finally {
    if (!identical(contiguousGrad, grad)) {
      contiguousGrad.dispose();
    }
  }
}

/// Dispatches an in-place fused `Adam.step` / `AdamW.step` update kernel on the GPU.
void dispatchAdamStep({
  required GpuArray<DTypeTag> parameter,
  required GpuArray<DTypeTag> grad,
  required GpuArray<DTypeTag> firstMoment,
  required GpuArray<DTypeTag> secondMoment,
  required double lr,
  required double beta1,
  required double beta2,
  required double eps,
  required double weightDecay,
  required double invBiasCorrection1,
  required double invBiasCorrection2,
  required bool decoupledWeightDecay,
}) {
  final totalElements = parameter.size;
  if (totalElements == 0) return;

  final contiguousGrad = grad.isContiguous ? grad : grad.copy();
  try {
    final bindings = [
      storageBinding(
        0,
        'param_buf',
        parameter.dtype,
        WgslBufferAccess.readWrite,
      ),
      storageBinding(
        1,
        'grad_buf',
        contiguousGrad.dtype,
        WgslBufferAccess.read,
      ),
      storageBinding(2, 'm_buf', firstMoment.dtype, WgslBufferAccess.readWrite),
      storageBinding(
        3,
        'v_buf',
        secondMoment.dtype,
        WgslBufferAccess.readWrite,
      ),
      const WgslBinding(
        group: 0,
        binding: 4,
        name: 'uniforms',
        isUniform: true,
        customTypeName: 'AdamStepUniforms',
      ),
    ];

    final loadParamFn = wgslLoadFloat(
      parameter.dtype,
      'param_buf',
      'load_param',
    );
    final storeParamFn = wgslStoreFloat(
      parameter.dtype,
      'param_buf',
      'store_param',
    );
    final loadGradFn = wgslLoadFloat(
      contiguousGrad.dtype,
      'grad_buf',
      'load_grad',
    );
    final loadMFn = wgslLoadFloat(firstMoment.dtype, 'm_buf', 'load_m');
    final storeMFn = wgslStoreFloat(firstMoment.dtype, 'm_buf', 'store_m');
    final loadVFn = wgslLoadFloat(secondMoment.dtype, 'v_buf', 'load_v');
    final storeVFn = wgslStoreFloat(secondMoment.dtype, 'v_buf', 'store_v');

    final code =
        '''
$wgslF64ConversionHelpers
struct AdamStepUniforms {
  total_elements: u32, param_offset: u32, grad_offset: u32, lr_bits: u32,
  beta1_bits: u32, beta2_bits: u32, eps_bits: u32, weight_decay_bits: u32,
  inv_bc1_bits: u32, inv_bc2_bits: u32, decoupled_wd: u32, pad0: u32,
}
${bindings.map((b) => b.toWgslDeclaration()).join('\n')}
$loadParamFn
$storeParamFn
$loadGradFn
$loadMFn
$storeMFn
$loadVFn
$storeVFn
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>) {
  let thread_index = global_id.x + global_id.y * (num_wg.x * 256u);
  if (thread_index >= uniforms.total_elements) { return; }
  var p = load_param(uniforms.param_offset + thread_index);
  var g = load_grad(uniforms.grad_offset + thread_index);
  let lr_val = bitcast<f32>(uniforms.lr_bits);
  let wd = bitcast<f32>(uniforms.weight_decay_bits);
  if (wd != 0.0) {
    if (uniforms.decoupled_wd != 0u) {
      p = p * (1.0 - lr_val * wd);
    } else {
      g = g + wd * p;
    }
  }
  let b1 = bitcast<f32>(uniforms.beta1_bits);
  let b2 = bitcast<f32>(uniforms.beta2_bits);
  let m_prev = load_m(thread_index);
  let v_prev = load_v(thread_index);
  let m_next = b1 * m_prev + (1.0 - b1) * g;
  let v_next = b2 * v_prev + (1.0 - b2) * (g * g);
  store_m(thread_index, m_next);
  store_v(thread_index, v_next);
  let m_hat = m_next * bitcast<f32>(uniforms.inv_bc1_bits);
  let v_hat = v_next * bitcast<f32>(uniforms.inv_bc2_bits);
  let eps_val = bitcast<f32>(uniforms.eps_bits);
  let p_next = p - lr_val * (m_hat / (sqrt(v_hat) + eps_val));
  store_param(uniforms.param_offset + thread_index, p_next);
}
''';

    dispatch1DKernel(
      device: parameter.device,
      name: 'nn_adam_step_${parameter.dtype.name}_${contiguousGrad.dtype.name}',
      code: code,
      bindings: bindings,
      buffers: [
        parameter.buffer,
        contiguousGrad.buffer,
        firstMoment.buffer,
        secondMoment.buffer,
      ],
      uniforms: [
        totalElements,
        parameter.offsetElements,
        contiguousGrad.offsetElements,
        float32ToBits(lr),
        float32ToBits(beta1),
        float32ToBits(beta2),
        float32ToBits(eps),
        float32ToBits(weightDecay),
        float32ToBits(invBiasCorrection1),
        float32ToBits(invBiasCorrection2),
        decoupledWeightDecay ? 1 : 0,
        0,
      ],
      totalElements: totalElements,
    );
  } finally {
    if (!identical(contiguousGrad, grad)) {
      contiguousGrad.dispose();
    }
  }
}
