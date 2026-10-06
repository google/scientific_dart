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

import 'dart:math' as math;

import '../autograd/autograd.dart';
import '../dtype.dart';
import '../gpu_array.dart';
import 'functional.dart';
import 'nn_wgsl.dart';

/// Applies 1D Batch Normalization over a 2D (`[N, C]`) or 3D (`[N, C, L]`) input tensor.
GpuArray<T> batchNorm1d<T extends DTypeTag>(
  GpuArray<T> input, {
  GpuArray<DTypeTag>? runningMean,
  GpuArray<DTypeTag>? runningVar,
  GpuArray<DTypeTag>? weight,
  GpuArray<DTypeTag>? bias,
  bool training = false,
  double momentum = 0.1,
  double eps = 1e-5,
}) {
  if (input.rank != 2 && input.rank != 3) {
    throw ArgumentError.value(
      input.shape,
      'input',
      'Must be a 2D ([batchSize, numFeatures]) or 3D ([batchSize, numFeatures, length]) tensor.',
    );
  }
  final numFeatures = input.shape[1];
  final is3D = input.rank == 3;
  final batchSize = input.shape[0];
  final seqLength = is3D ? input.shape[2] : 1;
  final sampleCount = batchSize * seqLength;

  final flatInput = is3D
      ? input.transpose([0, 2, 1]).reshape([sampleCount, numFeatures])
      : input;

  final GpuArray<T> normalizedFlat;
  if (training || runningMean == null || runningVar == null) {
    final invCount = 1.0 / sampleCount;
    final batchMean = flatInput.sum(axis: 0, keepDims: true) * invCount;
    final centered = flatInput - batchMean;
    final batchVar =
        (centered * centered).sum(axis: 0, keepDims: true) * invCount;

    if (training && runningMean != null && runningVar != null) {
      noGrad(() {
        final mean1D = batchMean.reshape([numFeatures]);
        final var1D = batchVar.reshape([numFeatures]);
        final matchedMean = mean1D.dtype == runningMean.dtype
            ? mean1D
            : mean1D.astype(runningMean.dtype);
        final matchedVar = var1D.dtype == runningVar.dtype
            ? var1D
            : var1D.astype(runningVar.dtype);
        final besselScale = sampleCount > 1
            ? sampleCount / (sampleCount - 1.0)
            : 1.0;
        final updatedMean =
            runningMean * (1.0 - momentum) + matchedMean * momentum;
        final updatedVar =
            runningVar * (1.0 - momentum) +
            matchedVar * (besselScale * momentum);
        updatedMean.buffer.copyToBuffer(
          runningMean.buffer,
          runningMean.byteSize,
        );
        updatedVar.buffer.copyToBuffer(runningVar.buffer, runningVar.byteSize);
        updatedMean.dispose();
        updatedVar.dispose();
        if (!identical(matchedMean, mean1D)) {
          matchedMean.dispose();
        }
        if (!identical(matchedVar, var1D)) {
          matchedVar.dispose();
        }
        mean1D.dispose();
        var1D.dispose();
      });
    }

    normalizedFlat = centered / (batchVar + eps).sqrt();
  } else {
    final meanView = runningMean.reshape([1, numFeatures]);
    final varView = runningVar.reshape([1, numFeatures]);
    final stdView = (varView + eps).sqrt();
    normalizedFlat = (flatInput - meanView) / stdView;
  }

  var transformed = normalizedFlat;
  if (weight != null) {
    transformed = transformed * weight.reshape([1, numFeatures]);
  }
  if (bias != null) {
    transformed = transformed + bias.reshape([1, numFeatures]);
  }

  return is3D
      ? transformed.reshape([batchSize, seqLength, numFeatures]).transpose([
          0,
          2,
          1,
        ])
      : transformed;
}

/// Computes Scaled Dot-Product Attention (SDPA):
/// $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$
///
/// Supports batched queries, keys, and values (e.g. `[B, H, N, D]` or `[N, D]`),
/// causal upper-triangular masking when [isCausal] is `true`, custom boolean or
/// additive float [attnMask], dropout probability [dropoutP], and custom [scale].
GpuArray<T> scaledDotProductAttention<T extends DTypeTag>(
  GpuArray<T> query,
  GpuArray<T> key,
  GpuArray<T> value, {
  GpuArray<DTypeTag>? attnMask,
  double dropoutP = 0.0,
  bool isCausal = false,
  double? scale,
}) {
  if (dropoutP < 0.0 || dropoutP >= 1.0) {
    throw ArgumentError.value(
      dropoutP,
      'dropoutP',
      'Must be in the half-open interval [0.0, 1.0).',
    );
  }
  final keyDimension = query.shape[query.rank - 1];
  final scaleFactor = scale ?? (1.0 / math.sqrt(keyDimension));

  final keyTransposed = key.swapaxes(-1, -2);
  var scores = query.matmul(keyTransposed) * scaleFactor;

  final querySeqLength = query.shape[query.rank - 2];
  final keySeqLength = key.shape[key.rank - 2];

  if (isCausal) {
    final causalMask = GpuArray.empty(
      [querySeqLength, keySeqLength],
      scores.dtype,
      device: scores.device,
    );
    dispatchCausalMask(
      causalMask: causalMask,
      querySeqLength: querySeqLength,
      keySeqLength: keySeqLength,
    );
    scores = scores + causalMask;
  }

  if (attnMask != null) {
    if (attnMask.dtype == DType.boolean) {
      final additiveMask = GpuArray.empty(
        attnMask.shape,
        scores.dtype,
        device: scores.device,
      );
      dispatchBooleanAdditiveMask(
        boolMask: attnMask,
        additiveMask: additiveMask,
      );
      scores = scores + additiveMask;
    } else {
      scores = scores + attnMask;
    }
  }

  var attentionWeights = softmax(scores, axis: -1);

  if (dropoutP > 0.0) {
    attentionWeights = dropout(attentionWeights, p: dropoutP, training: true);
  }

  return attentionWeights.matmul(value);
}
