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

import '../gpu_array.dart';
import 'autograd_core.dart';
import 'autograd_wgsl.dart';
import 'loss_wgsl.dart';

/// Extracts 2D spatial patches from [input] (`[N, C, H, W]`) into a 2D column matrix
/// of shape `[N * outHeight * outWidth, C * kernelSize * kernelSize]`.
GpuArray<DTypeTag> extractIm2ColPatches(
  GpuArray<DTypeTag> input, {
  required int kernelSize,
  required int stride,
  required int padding,
  required int outHeight,
  required int outWidth,
}) {
  final batchSize = input.shape[0];
  final inChannels = input.shape[1];
  final patchSize = inChannels * kernelSize * kernelSize;
  final totalPatches = batchSize * outHeight * outWidth;

  final columns = GpuArray.zeros(
    [totalPatches, patchSize],
    input.dtype,
    device: input.device,
  );
  dispatchIm2Col(
    input: input,
    columns: columns,
    kernelSize: kernelSize,
    stride: stride,
    padding: padding,
    outHeight: outHeight,
    outWidth: outWidth,
  );
  return columns;
}

/// Accumulates 2D column gradients (`[N * outHeight * outWidth, C * kernelSize * kernelSize]`)
/// back into a 4D input gradient tensor of [inputShape] (`[N, C, H, W]`).
GpuArray<DTypeTag> accumulateCol2ImPatches(
  GpuArray<DTypeTag> gradColumns, {
  required List<int> inputShape,
  required int kernelSize,
  required int stride,
  required int padding,
  required int outHeight,
  required int outWidth,
}) {
  final gradInput = GpuArray.zeros(
    inputShape,
    gradColumns.dtype,
    device: gradColumns.device,
  );
  dispatchCol2Im(
    gradColumns: gradColumns,
    gradInput: gradInput,
    kernelSize: kernelSize,
    stride: stride,
    padding: padding,
    outHeight: outHeight,
    outWidth: outWidth,
  );
  return gradInput;
}

/// Backward node for 2D spatial convolution (`im2col` + GPU `matmul` / `col2im`).
final class Conv2dBackward extends GradFn {
  /// Forward 4D input tensor of shape `[N, C_in, H_in, W_in]`.
  final GpuArray<DTypeTag> input;

  /// Convolution filter weights of shape `[C_out, C_in, K, K]`.
  final GpuArray<DTypeTag> weight;

  /// Optional per-channel bias tensor of shape `[C_out]`.
  final GpuArray<DTypeTag>? bias;

  /// Convolution spatial stride.
  final int stride;

  /// Zero-padding added to both sides of the input spatial dimensions.
  final int padding;

  /// Spatial height and width of the square convolution kernel.
  final int kernelSize;

  /// Creates a [Conv2dBackward] node.
  const Conv2dBackward({
    required this.input,
    required this.weight,
    this.bias,
    required this.stride,
    required this.padding,
    required this.kernelSize,
  });

  @override
  String get name => 'Conv2dBackward';

  @override
  List<GpuArray<DTypeTag>> get inputs => [input, weight, ?bias];

  @override
  List<GpuArray<DTypeTag>?> backward(GpuArray<DTypeTag> gradOutput) {
    final batchSize = input.shape[0];
    final inChannels = input.shape[1];
    final outChannels = weight.shape[0];
    final outHeight = gradOutput.shape[2];
    final outWidth = gradOutput.shape[3];
    final patchSize = inChannels * kernelSize * kernelSize;
    final totalPatches = batchSize * outHeight * outWidth;

    final gradPermuted = gradOutput.transpose([0, 2, 3, 1]);
    final gradContiguous = gradPermuted.isContiguous
        ? gradPermuted
        : gradPermuted.copy();
    if (!identical(gradContiguous, gradPermuted)) {
      gradPermuted.dispose();
    }
    final gradOutMatrix = gradContiguous.reshape([totalPatches, outChannels]);
    gradContiguous.dispose();

    GpuArray<DTypeTag>? gradInput;
    GpuArray<DTypeTag>? gradWeight;
    GpuArray<DTypeTag>? gradBias;

    try {
      if (weight.requiresGrad) {
        final columns = extractIm2ColPatches(
          input,
          kernelSize: kernelSize,
          stride: stride,
          padding: padding,
          outHeight: outHeight,
          outWidth: outWidth,
        );
        final gradOutTransposed = gradOutMatrix.swapaxes(-1, -2);
        final gradWeightFlat = gradOutTransposed.matmul(columns);
        gradOutTransposed.dispose();
        columns.dispose();
        gradWeight = gradWeightFlat.reshape(weight.shape);
        gradWeightFlat.dispose();
      }

      if (input.requiresGrad) {
        final weightMatrix = weight.reshape([outChannels, patchSize]);
        final gradColumns = gradOutMatrix.matmul(weightMatrix);
        weightMatrix.dispose();
        gradInput = accumulateCol2ImPatches(
          gradColumns,
          inputShape: input.shape,
          kernelSize: kernelSize,
          stride: stride,
          padding: padding,
          outHeight: outHeight,
          outWidth: outWidth,
        );
        gradColumns.dispose();
      }

      if (bias != null && bias!.requiresGrad) {
        gradBias = gradOutMatrix.sum(axis: 0);
      }
    } finally {
      gradOutMatrix.dispose();
    }

    return [gradInput, gradWeight, ?gradBias];
  }
}

/// Backward node for 1D Batch Normalization (`BatchNorm1d`).
final class BatchNorm1dBackward extends GradFn {
  /// Forward input tensor of shape `[N, C]`.
  final GpuArray<DTypeTag> input;

  /// Optional learnable scale parameter ($\gamma$) of shape `[C]`.
  final GpuArray<DTypeTag>? weight;

  /// Optional learnable shift parameter ($\beta$) of shape `[C]`.
  final GpuArray<DTypeTag>? bias;

  /// Cached centered input ($x - \mu$) of shape `[N, C]`.
  final GpuArray<DTypeTag> centered;

  /// Cached inverse standard deviation ($1 / \sqrt{\sigma^2 + \epsilon}$) of shape `[1, C]`.
  final GpuArray<DTypeTag> invStd;

  /// Creates a [BatchNorm1dBackward] node.
  const BatchNorm1dBackward({
    required this.input,
    required this.centered,
    required this.invStd,
    this.weight,
    this.bias,
  });

  @override
  String get name => 'BatchNorm1dBackward';

  @override
  List<GpuArray<DTypeTag>> get inputs => [input, ?weight, ?bias];

  @override
  List<GpuArray<DTypeTag>?> backward(GpuArray<DTypeTag> gradOutput) {
    final batchSize = input.shape[0];
    final normalized = centered * invStd;

    GpuArray<DTypeTag>? gradWeight;
    if (weight != null && weight!.requiresGrad) {
      final prod = gradOutput * normalized;
      gradWeight = prod.sum(axis: 0);
      prod.dispose();
    }

    GpuArray<DTypeTag>? gradBias;
    if (bias != null && bias!.requiresGrad) {
      gradBias = gradOutput.sum(axis: 0);
    }

    GpuArray<DTypeTag>? gradInput;
    if (input.requiresGrad) {
      final dNormalized = weight != null ? gradOutput * weight : gradOutput;
      final sumDNorm = dNormalized.sum(axis: 0, keepDims: true);
      final dNormTimesNorm = dNormalized * normalized;
      final sumDNormTimesNorm = dNormTimesNorm.sum(axis: 0, keepDims: true);
      dNormTimesNorm.dispose();

      final scaledDNorm = dNormalized * batchSize.toDouble();
      if (weight != null) {
        dNormalized.dispose();
      }
      final diff1 = scaledDNorm - sumDNorm;
      scaledDNorm.dispose();
      sumDNorm.dispose();
      final projTerm = normalized * sumDNormTimesNorm;
      sumDNormTimesNorm.dispose();
      final numer = diff1 - projTerm;
      diff1.dispose();
      projTerm.dispose();
      final scale = invStd * (1.0 / batchSize);
      gradInput = numer * scale;
      numer.dispose();
      scale.dispose();
    }

    normalized.dispose();
    return [gradInput, ?gradWeight, ?gradBias];
  }
}

/// Backward node for categorical cross-entropy loss.
final class CrossEntropyBackward extends GradFn {
  /// Unnormalized forward logits tensor.
  final GpuArray<DTypeTag> logits;

  /// Ground-truth integer target class indices.
  final GpuArray<DTypeTag> targets;

  /// Cached softmax probabilities tensor from the forward pass.
  final GpuArray<DTypeTag> probabilities;

  /// Reduction mode applied to the forward loss.
  final LossReduction reduction;

  /// Creates a [CrossEntropyBackward] node.
  const CrossEntropyBackward(
    this.logits,
    this.targets,
    this.probabilities, {
    this.reduction = LossReduction.mean,
  });

  @override
  String get name => 'CrossEntropyBackward';

  @override
  List<GpuArray<DTypeTag>> get inputs => [logits];

  @override
  List<GpuArray<DTypeTag>?> backward(GpuArray<DTypeTag> gradOutput) {
    if (!logits.requiresGrad) return [null];
    final numSamples = logits.shape[0];
    final numClasses = logits.shape[logits.rank - 1];

    final grad = GpuArray.empty(
      logits.shape,
      logits.dtype,
      device: logits.device,
    );
    dispatchCrossEntropyBackward(
      probabilities: probabilities,
      targets: targets,
      gradLogits: grad,
      numSamples: numSamples,
      numClasses: numClasses,
    );

    switch (reduction) {
      case LossReduction.mean:
        final scaledBase = grad * (1.0 / numSamples);
        grad.dispose();
        final scaled = scaledBase * gradOutput;
        scaledBase.dispose();
        return [scaled];
      case LossReduction.sum:
        final scaled = grad * gradOutput;
        grad.dispose();
        return [scaled];
      case LossReduction.none:
        final gradUnsqueezed = gradOutput.unsqueeze(-1);
        final scaled = grad * gradUnsqueezed;
        gradUnsqueezed.dispose();
        grad.dispose();
        return [scaled];
    }
  }
}

/// Backward node for L1 (mean absolute error) loss.
final class L1LossBackward extends GradFn {
  /// Forward prediction tensor.
  final GpuArray<DTypeTag> input;

  /// Ground-truth target tensor.
  final GpuArray<DTypeTag> target;

  /// Reduction mode applied to the forward loss.
  final LossReduction reduction;

  /// Creates an [L1LossBackward] node.
  const L1LossBackward(
    this.input,
    this.target, {
    this.reduction = LossReduction.mean,
  });

  @override
  String get name => 'L1LossBackward';

  @override
  List<GpuArray<DTypeTag>> get inputs => [input, target];

  @override
  List<GpuArray<DTypeTag>?> backward(GpuArray<DTypeTag> gradOutput) {
    if (!input.requiresGrad && !target.requiresGrad) return [null, null];

    final signTensor = GpuArray.empty(
      input.shape,
      input.dtype,
      device: input.device,
    );
    dispatchL1LossBackward(
      input: input,
      targetTensor: target,
      signOutput: signTensor,
    );

    final GpuArray<DTypeTag> baseGrad;
    switch (reduction) {
      case LossReduction.mean:
        final scaledSign = signTensor * (1.0 / input.size);
        signTensor.dispose();
        baseGrad = scaledSign * gradOutput;
        scaledSign.dispose();
      case LossReduction.sum:
      case LossReduction.none:
        baseGrad = signTensor * gradOutput;
        signTensor.dispose();
    }

    final gradInput = input.requiresGrad ? baseGrad : null;
    final gradTarget = target.requiresGrad ? baseGrad.negate() : null;
    if (!input.requiresGrad) {
      baseGrad.dispose();
    }
    return [gradInput, gradTarget];
  }
}

/// Backward node for binary cross-entropy loss.
final class BinaryCrossEntropyBackward extends GradFn {
  /// Forward predicted probabilities tensor.
  final GpuArray<DTypeTag> input;

  /// Ground-truth target probabilities tensor.
  final GpuArray<DTypeTag> target;

  /// Reduction mode applied to the forward loss.
  final LossReduction reduction;

  /// Creates a [BinaryCrossEntropyBackward] node.
  const BinaryCrossEntropyBackward(
    this.input,
    this.target, {
    this.reduction = LossReduction.mean,
  });

  @override
  String get name => 'BinaryCrossEntropyBackward';

  @override
  List<GpuArray<DTypeTag>> get inputs => [input];

  @override
  List<GpuArray<DTypeTag>?> backward(GpuArray<DTypeTag> gradOutput) {
    if (!input.requiresGrad) return [null];
    final rawGrad = GpuArray.empty(
      input.shape,
      input.dtype,
      device: input.device,
    );
    dispatchBceBackward(input: input, targetTensor: target, gradInput: rawGrad);

    switch (reduction) {
      case LossReduction.mean:
        final scaledBase = rawGrad * (1.0 / input.size);
        rawGrad.dispose();
        final scaled = scaledBase * gradOutput;
        scaledBase.dispose();
        return [scaled];
      case LossReduction.sum:
      case LossReduction.none:
        final scaled = rawGrad * gradOutput;
        rawGrad.dispose();
        return [scaled];
    }
  }
}
