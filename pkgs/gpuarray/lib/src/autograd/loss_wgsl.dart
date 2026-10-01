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

import '../backend/wgsl/wgsl_types.dart';
import '../buffer.dart';
import '../gpu_array.dart';
import 'autograd_wgsl.dart';

/// Dispatches the `crossEntropy` forward per-sample negative log-probability gather kernel on the GPU.
void dispatchCrossEntropyForward({
  required GpuArray<DTypeTag> logProbabilities,
  required GpuArray<DTypeTag> targets,
  required GpuArray<DTypeTag> sampleLosses,
  required int numSamples,
  required int numClasses,
}) {
  if (numSamples == 0) return;

  final contiguousLogProbs = logProbabilities.isContiguous
      ? logProbabilities
      : logProbabilities.copy();
  final contiguousTargets = targets.isContiguous ? targets : targets.copy();
  final statusBuffer = GpuBuffer.allocate(
    sizeInBytes: 8,
    device: logProbabilities.device,
  )..clear();

  try {
    final bindings = [
      storageBinding(
        0,
        'log_probs_buf',
        contiguousLogProbs.dtype,
        WgslBufferAccess.read,
      ),
      storageBinding(
        1,
        'targets_buf',
        contiguousTargets.dtype,
        WgslBufferAccess.read,
      ),
      storageBinding(
        2,
        'loss_buf',
        sampleLosses.dtype,
        WgslBufferAccess.readWrite,
      ),
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
        customTypeName: 'CrossEntropyForwardUniforms',
      ),
    ];

    final loadLogProbFn = wgslLoadFloat(
      contiguousLogProbs.dtype,
      'log_probs_buf',
      'load_log_prob',
    );
    final loadTargetFn = wgslLoadIndex(
      contiguousTargets.dtype,
      'targets_buf',
      'load_target',
    );
    final storeLossFn = wgslStoreFloat(
      sampleLosses.dtype,
      'loss_buf',
      'store_loss',
    );

    final code =
        '''
$wgslF64ConversionHelpers
struct CrossEntropyForwardUniforms {
  num_samples: u32, num_classes: u32, log_probs_offset: u32, targets_offset: u32,
}
${bindings.map((b) => b.toWgslDeclaration()).join('\n')}
$loadLogProbFn
$loadTargetFn
$storeLossFn
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>) {
  let sample_index = global_id.x + global_id.y * (num_wg.x * 256u);
  if (sample_index >= uniforms.num_samples) { return; }
  let class_index = load_target(uniforms.targets_offset + sample_index);
  if (class_index < 0 || class_index >= i32(uniforms.num_classes)) {
    let prev = atomicCompareExchangeWeak(&status_buf[0], 0, 1);
    if (prev.exchanged) { atomicStore(&status_buf[1], class_index); }
    return;
  }
  let prob_index = uniforms.log_probs_offset + sample_index * uniforms.num_classes + u32(class_index);
  store_loss(sample_index, -load_log_prob(prob_index));
}
''';

    dispatch1DKernel(
      device: logProbabilities.device,
      name:
          'nn_cross_entropy_forward_${sampleLosses.dtype.name}_${contiguousTargets.dtype.name}',
      code: code,
      bindings: bindings,
      buffers: [
        contiguousLogProbs.buffer,
        contiguousTargets.buffer,
        sampleLosses.buffer,
        statusBuffer,
      ],
      uniforms: [
        numSamples,
        numClasses,
        contiguousLogProbs.offsetElements,
        contiguousTargets.offsetElements,
      ],
      totalElements: numSamples,
    );

    final statusBytes = statusBuffer.readBytes();
    final statusView = ByteData.sublistView(statusBytes);
    if (statusView.getInt32(0, Endian.little) != 0) {
      final offendingIndex = statusView.getInt32(4, Endian.little);
      sampleLosses.dispose();
      RangeError.checkValueInInterval(
        offendingIndex,
        0,
        numClasses - 1,
        'targets',
      );
    }
  } finally {
    statusBuffer.dispose();
    if (!identical(contiguousLogProbs, logProbabilities)) {
      contiguousLogProbs.dispose();
    }
    if (!identical(contiguousTargets, targets)) {
      contiguousTargets.dispose();
    }
  }
}

/// Dispatches the `CrossEntropyBackward` kernel (`probs - oneHot(targets)`) on the GPU.
void dispatchCrossEntropyBackward({
  required GpuArray<DTypeTag> probabilities,
  required GpuArray<DTypeTag> targets,
  required GpuArray<DTypeTag> gradLogits,
  required int numSamples,
  required int numClasses,
}) {
  final totalElements = numSamples * numClasses;
  if (totalElements == 0) return;

  final contiguousProbs = probabilities.isContiguous
      ? probabilities
      : probabilities.copy();
  final contiguousTargets = targets.isContiguous ? targets : targets.copy();

  try {
    final bindings = [
      storageBinding(
        0,
        'probs_buf',
        contiguousProbs.dtype,
        WgslBufferAccess.read,
      ),
      storageBinding(
        1,
        'targets_buf',
        contiguousTargets.dtype,
        WgslBufferAccess.read,
      ),
      storageBinding(
        2,
        'grad_buf',
        gradLogits.dtype,
        WgslBufferAccess.readWrite,
      ),
      const WgslBinding(
        group: 0,
        binding: 3,
        name: 'uniforms',
        isUniform: true,
        customTypeName: 'CrossEntropyBackwardUniforms',
      ),
    ];

    final loadProbFn = wgslLoadFloat(
      contiguousProbs.dtype,
      'probs_buf',
      'load_prob',
    );
    final loadTargetFn = wgslLoadIndex(
      contiguousTargets.dtype,
      'targets_buf',
      'load_target',
    );
    final storeGradFn = wgslStoreFloat(
      gradLogits.dtype,
      'grad_buf',
      'store_grad',
    );

    final code =
        '''
$wgslF64ConversionHelpers
struct CrossEntropyBackwardUniforms {
  total_elements: u32, num_samples: u32, num_classes: u32, probs_offset: u32,
  targets_offset: u32, pad0: u32, pad1: u32, pad2: u32,
}
${bindings.map((b) => b.toWgslDeclaration()).join('\n')}
$loadProbFn
$loadTargetFn
$storeGradFn
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>) {
  let thread_index = global_id.x + global_id.y * (num_wg.x * 256u);
  if (thread_index >= uniforms.total_elements) { return; }
  let sample_index = thread_index / uniforms.num_classes;
  let class_col = i32(thread_index % uniforms.num_classes);
  let target_class = load_target(uniforms.targets_offset + sample_index);
  let prob = load_prob(uniforms.probs_offset + thread_index);
  store_grad(thread_index, prob - select(0.0, 1.0, class_col == target_class));
}
''';

    dispatch1DKernel(
      device: gradLogits.device,
      name:
          'autograd_cross_entropy_backward_${gradLogits.dtype.name}_${contiguousTargets.dtype.name}',
      code: code,
      bindings: bindings,
      buffers: [
        contiguousProbs.buffer,
        contiguousTargets.buffer,
        gradLogits.buffer,
      ],
      uniforms: [
        totalElements,
        numSamples,
        numClasses,
        contiguousProbs.offsetElements,
        contiguousTargets.offsetElements,
        0,
        0,
        0,
      ],
      totalElements: totalElements,
    );
  } finally {
    if (!identical(contiguousProbs, probabilities)) {
      contiguousProbs.dispose();
    }
    if (!identical(contiguousTargets, targets)) {
      contiguousTargets.dispose();
    }
  }
}

void _dispatchBinaryFloatLossKernel({
  required GpuArray<DTypeTag> input,
  required GpuArray<DTypeTag> targetTensor,
  required GpuArray<DTypeTag> output,
  required String kernelPrefix,
  required String bodyWgsl,
}) {
  final totalElements = output.size;
  if (totalElements == 0) return;

  final contiguousInput = input.isContiguous ? input : input.copy();
  final contiguousTarget = targetTensor.isContiguous
      ? targetTensor
      : targetTensor.copy();

  try {
    final bindings = [
      storageBinding(0, 'in_buf', contiguousInput.dtype, WgslBufferAccess.read),
      storageBinding(
        1,
        'tgt_buf',
        contiguousTarget.dtype,
        WgslBufferAccess.read,
      ),
      storageBinding(2, 'out_buf', output.dtype, WgslBufferAccess.readWrite),
      const WgslBinding(
        group: 0,
        binding: 3,
        name: 'uniforms',
        isUniform: true,
        customTypeName: 'BinaryLossUniforms',
      ),
    ];

    final loadInFn = wgslLoadFloat(contiguousInput.dtype, 'in_buf', 'load_in');
    final loadTgtFn = wgslLoadFloat(
      contiguousTarget.dtype,
      'tgt_buf',
      'load_tgt',
    );
    final storeOutFn = wgslStoreFloat(output.dtype, 'out_buf', 'store_out');

    final code =
        '''
$wgslF64ConversionHelpers
struct BinaryLossUniforms {
  total_elements: u32, in_offset: u32, tgt_offset: u32, pad0: u32,
}
${bindings.map((b) => b.toWgslDeclaration()).join('\n')}
$loadInFn
$loadTgtFn
$storeOutFn
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>) {
  let thread_index = global_id.x + global_id.y * (num_wg.x * 256u);
  if (thread_index >= uniforms.total_elements) { return; }
  let in_val = load_in(uniforms.in_offset + thread_index);
  let tgt_val = load_tgt(uniforms.tgt_offset + thread_index);
  $bodyWgsl
}
''';

    dispatch1DKernel(
      device: output.device,
      name: '${kernelPrefix}_${output.dtype.name}',
      code: code,
      bindings: bindings,
      buffers: [contiguousInput.buffer, contiguousTarget.buffer, output.buffer],
      uniforms: [
        totalElements,
        contiguousInput.offsetElements,
        contiguousTarget.offsetElements,
        0,
      ],
      totalElements: totalElements,
    );
  } finally {
    if (!identical(contiguousInput, input)) {
      contiguousInput.dispose();
    }
    if (!identical(contiguousTarget, targetTensor)) {
      contiguousTarget.dispose();
    }
  }
}

/// Dispatches the `L1LossBackward` sign kernel ($\text{sign}(\text{input} - \text{target})$) on the GPU.
void dispatchL1LossBackward({
  required GpuArray<DTypeTag> input,
  required GpuArray<DTypeTag> targetTensor,
  required GpuArray<DTypeTag> signOutput,
}) => _dispatchBinaryFloatLossKernel(
  input: input,
  targetTensor: targetTensor,
  output: signOutput,
  kernelPrefix: 'autograd_l1_backward',
  bodyWgsl: '''
  let diff = in_val - tgt_val;
  store_out(thread_index, select(select(0.0, -1.0, diff < 0.0), 1.0, diff > 0.0));''',
);

/// Dispatches the `binaryCrossEntropy` forward elementwise kernel on the GPU.
void dispatchBceForward({
  required GpuArray<DTypeTag> input,
  required GpuArray<DTypeTag> targetTensor,
  required GpuArray<DTypeTag> output,
}) => _dispatchBinaryFloatLossKernel(
  input: input,
  targetTensor: targetTensor,
  output: output,
  kernelPrefix: 'nn_bce_forward',
  bodyWgsl: '''
  let p = clamp(in_val, 1e-7, 1.0 - 1e-7);
  store_out(thread_index, -(tgt_val * log(p) + (1.0 - tgt_val) * log(1.0 - p)));''',
);

/// Dispatches the `BinaryCrossEntropyBackward` elementwise derivative kernel on the GPU.
void dispatchBceBackward({
  required GpuArray<DTypeTag> input,
  required GpuArray<DTypeTag> targetTensor,
  required GpuArray<DTypeTag> gradInput,
}) => _dispatchBinaryFloatLossKernel(
  input: input,
  targetTensor: targetTensor,
  output: gradInput,
  kernelPrefix: 'autograd_bce_backward',
  bodyWgsl: '''
  let p = clamp(in_val, 1e-7, 1.0 - 1e-7);
  store_out(thread_index, (p - tgt_val) / (p * (1.0 - p)));''',
);
