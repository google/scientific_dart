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
import '../device.dart';
import '../dtype.dart';
import '../gpu_array.dart';
import '../random/random.dart' as random_ops;
import 'functional.dart' as functional;
import 'module.dart';

GpuArray<DTypeTag> _sampleUniformParameter({
  required double low,
  required double high,
  required List<int> shape,
  required DType<DTypeTag> dtype,
  required GpuDevice device,
}) {
  if (!dtype.isFloating) {
    throw ArgumentError.value(
      dtype,
      'dtype',
      'Must be a floating-point DType.',
    );
  }
  final sampled = random_ops.uniform(
    low: low,
    high: high,
    shape: shape,
    device: device,
  );
  if (dtype == DType.float64) {
    sampled.requiresGrad = true;
    return sampled;
  }
  final converted = sampled.astype(dtype)..requiresGrad = true;
  sampled.dispose();
  return converted;
}

/// Applies an affine linear transformation to incoming data: $y = x A^T + b$.
final class Linear extends Module {
  /// Size of each input sample.
  final int inFeatures;

  /// Size of each output sample.
  final int outFeatures;

  /// Whether this layer learns an additive bias vector.
  final bool hasBias;

  /// Learnable weight matrix of shape `[outFeatures, inFeatures]`.
  late final GpuArray<DTypeTag> weight;

  /// Optional learnable bias vector of shape `[outFeatures]`.
  late final GpuArray<DTypeTag>? bias;

  /// Creates a [Linear] layer mapping [inFeatures] to [outFeatures].
  ///
  /// Both [inFeatures] and [outFeatures] must be positive integers.
  Linear(
    this.inFeatures,
    this.outFeatures, {
    this.hasBias = true,
    DType<DTypeTag> dtype = DType.float64,
    GpuDevice? device,
  }) {
    RangeError.checkValueInInterval(inFeatures, 1, 0x7fffffff, 'inFeatures');
    RangeError.checkValueInInterval(outFeatures, 1, 0x7fffffff, 'outFeatures');

    final targetDevice = device ?? GpuDevice.defaultDevice;
    final bound = 1.0 / math.sqrt(inFeatures);

    final sampledWeight = _sampleUniformParameter(
      low: -bound,
      high: bound,
      shape: [outFeatures, inFeatures],
      dtype: dtype,
      device: targetDevice,
    );
    weight = registerParameter('weight', sampledWeight);

    if (hasBias) {
      final sampledBias = _sampleUniformParameter(
        low: -bound,
        high: bound,
        shape: [outFeatures],
        dtype: dtype,
        device: targetDevice,
      );
      bias = registerParameter('bias', sampledBias);
    } else {
      bias = null;
    }
  }

  @override
  GpuArray<T> forward<T extends DTypeTag>(GpuArray<T> input) {
    checkNotDisposed();
    if (input.rank == 0 || input.shape[input.rank - 1] != inFeatures) {
      throw ArgumentError.value(
        input.shape,
        'input',
        'Must have trailing feature dimension equal to inFeatures ($inFeatures).',
      );
    }
    final weightTransposed = weight.swapaxes(-1, -2);
    final output = input.matmul(weightTransposed);
    if (bias != null) {
      return output + bias;
    }
    return output;
  }
}

/// Applies a 2D spatial convolution over a 4D input signal (`[N, C_in, H_in, W_in]`).
final class Conv2d extends Module {
  /// Number of channels in the input image.
  final int inChannels;

  /// Number of channels produced by the convolution.
  final int outChannels;

  /// Spatial height and width of the square convolution kernel.
  final int kernelSize;

  /// Stride of the convolution along height and width.
  final int stride;

  /// Zero-padding added to all four spatial borders of the input.
  final int padding;

  /// Whether this layer learns an additive per-channel bias vector.
  final bool hasBias;

  /// Learnable filter weights of shape `[outChannels, inChannels, kernelSize, kernelSize]`.
  late final GpuArray<DTypeTag> weight;

  /// Optional learnable bias vector of shape `[outChannels]`.
  late final GpuArray<DTypeTag>? bias;

  /// Creates a [Conv2d] layer.
  ///
  /// The [inChannels], [outChannels], [kernelSize], and [stride] must be positive,
  /// and [padding] must be non-negative.
  Conv2d(
    this.inChannels,
    this.outChannels,
    this.kernelSize, {
    this.stride = 1,
    this.padding = 0,
    this.hasBias = true,
    DType<DTypeTag> dtype = DType.float64,
    GpuDevice? device,
  }) {
    if (inChannels <= 0) {
      throw ArgumentError.value(inChannels, 'inChannels', 'Must be positive.');
    }
    if (outChannels <= 0) {
      throw ArgumentError.value(
        outChannels,
        'outChannels',
        'Must be positive.',
      );
    }
    if (kernelSize <= 0) {
      throw ArgumentError.value(kernelSize, 'kernelSize', 'Must be positive.');
    }
    if (stride <= 0) {
      throw ArgumentError.value(stride, 'stride', 'Must be positive.');
    }
    RangeError.checkNotNegative(padding, 'padding');

    final targetDevice = device ?? GpuDevice.defaultDevice;
    final bound = 1.0 / math.sqrt(inChannels * kernelSize * kernelSize);

    final sampledWeight = _sampleUniformParameter(
      low: -bound,
      high: bound,
      shape: [outChannels, inChannels, kernelSize, kernelSize],
      dtype: dtype,
      device: targetDevice,
    );
    weight = registerParameter('weight', sampledWeight);

    if (hasBias) {
      final sampledBias = _sampleUniformParameter(
        low: -bound,
        high: bound,
        shape: [outChannels],
        dtype: dtype,
        device: targetDevice,
      );
      bias = registerParameter('bias', sampledBias);
    } else {
      bias = null;
    }
  }

  @override
  GpuArray<T> forward<T extends DTypeTag>(GpuArray<T> input) {
    checkNotDisposed();
    if (input.rank != 4) {
      throw ArgumentError.value(
        input.shape,
        'input',
        'Must be a 4D tensor of shape [batchSize, inChannels, height, width].',
      );
    }
    if (input.shape[1] != inChannels) {
      throw ArgumentError.value(
        input.shape,
        'input',
        'Must have channel dimension equal to inChannels ($inChannels).',
      );
    }
    final batchSize = input.shape[0];
    final inHeight = input.shape[2];
    final inWidth = input.shape[3];

    final outHeight = ((inHeight + 2 * padding - kernelSize) ~/ stride) + 1;
    final outWidth = ((inWidth + 2 * padding - kernelSize) ~/ stride) + 1;
    if (outHeight <= 0 || outWidth <= 0) {
      throw ArgumentError.value(
        input.shape,
        'input',
        'Must have spatial dimensions large enough for kernelSize ($kernelSize).',
      );
    }
    final patchSize = inChannels * kernelSize * kernelSize;

    final output = noGrad(() {
      final columns = extractIm2ColPatches(
        input,
        kernelSize: kernelSize,
        stride: stride,
        padding: padding,
        outHeight: outHeight,
        outWidth: outWidth,
      );
      final effectiveWeight = weight.dtype == input.dtype
          ? weight
          : weight.astype(input.dtype);
      final weightMatrix = effectiveWeight.reshape([outChannels, patchSize]);
      final weightTransposed = weightMatrix.swapaxes(-1, -2);
      var outMatrix = columns.matmul(weightTransposed);
      columns.dispose();
      weightTransposed.dispose();
      weightMatrix.dispose();
      if (!identical(effectiveWeight, weight)) {
        effectiveWeight.dispose();
      }

      if (bias != null) {
        final effectiveBias = bias!.dtype == input.dtype
            ? bias!
            : bias!.astype(input.dtype);
        final withBias = outMatrix + effectiveBias;
        outMatrix.dispose();
        if (!identical(effectiveBias, bias)) {
          effectiveBias.dispose();
        }
        outMatrix = withBias;
      }

      final reshaped = outMatrix.reshape([
        batchSize,
        outHeight,
        outWidth,
        outChannels,
      ]);
      outMatrix.dispose();
      final permuted = reshaped.transpose([0, 3, 1, 2]);
      reshaped.dispose();
      final contiguousOut = permuted.copy();
      permuted.dispose();
      return contiguousOut as GpuArray<T>;
    });

    if (isGradEnabled &&
        (input.requiresGrad ||
            weight.requiresGrad ||
            (bias != null && bias!.requiresGrad))) {
      output.requiresGrad = true;
      output.gradFn = Conv2dBackward(
        input: input,
        weight: weight,
        bias: bias,
        stride: stride,
        padding: padding,
        kernelSize: kernelSize,
      );
    }

    return output;
  }
}

/// Applies Layer Normalization over a mini-batch of inputs.
final class LayerNorm extends Module {
  /// Normalized trailing dimensions shape.
  final List<int> normalizedShape;

  /// Small constant added to the denominator for numerical stability.
  final double eps;

  /// Learnable elementwise affine scale parameter ($\gamma$).
  late final GpuArray<DTypeTag> weight;

  /// Learnable elementwise affine shift parameter ($\beta$).
  late final GpuArray<DTypeTag> bias;

  /// Creates a [LayerNorm] module for [normalizedShape].
  LayerNorm(
    List<int> normalizedShape, {
    this.eps = 1e-5,
    DType<DTypeTag> dtype = DType.float64,
    GpuDevice? device,
  }) : normalizedShape = List<int>.unmodifiable(normalizedShape) {
    if (this.normalizedShape.isEmpty) {
      throw ArgumentError.value(
        normalizedShape,
        'normalizedShape',
        'Must not be empty.',
      );
    }
    if (eps <= 0.0) {
      throw ArgumentError.value(eps, 'eps', 'Must be positive.');
    }
    if (!dtype.isFloating) {
      throw ArgumentError.value(
        dtype,
        'dtype',
        'Must be a floating-point DType.',
      );
    }
    final targetDevice = device ?? GpuDevice.defaultDevice;
    weight = registerParameter(
      'weight',
      GpuArray.ones(
        this.normalizedShape,
        dtype,
        device: targetDevice,
        requiresGrad: true,
      ),
    );
    bias = registerParameter(
      'bias',
      GpuArray.zeros(
        this.normalizedShape,
        dtype,
        device: targetDevice,
        requiresGrad: true,
      ),
    );
  }

  @override
  GpuArray<T> forward<T extends DTypeTag>(GpuArray<T> input) {
    checkNotDisposed();
    final mean = input.mean(axis: -1, keepDims: true);
    final centered = input - mean;
    final variance = (centered * centered).mean(axis: -1, keepDims: true);
    final stdInv = (variance + eps).sqrt();
    final normalized = _preserveLayerDType(input, centered / stdInv);
    return normalized * weight + bias;
  }
}

GpuArray<T> _preserveLayerDType<T extends DTypeTag>(
  GpuArray<T> reference,
  GpuArray<DTypeTag> res,
) {
  if (res.dtype == reference.dtype && res is GpuArray<T>) return res;
  final casted = res.astype(reference.dtype);
  if (res.requiresGrad) {
    casted.requiresGrad = true;
    casted.gradFn = res.gradFn;
    res.gradFn = null;
  }
  res.dispose();
  return casted;
}

/// Applies Root Mean Square Layer Normalization (RMSNorm) over a mini-batch of inputs:
/// $$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma$$
final class RMSNorm extends Module {
  /// Normalized trailing dimensions shape.
  final List<int> normalizedShape;

  /// Small constant added to the mean square for numerical stability.
  final double eps;

  /// Learnable elementwise scale parameter ($\gamma$).
  late final GpuArray<DTypeTag> weight;

  /// Creates an [RMSNorm] module for [normalizedShape].
  RMSNorm(
    List<int> normalizedShape, {
    this.eps = 1e-6,
    DType<DTypeTag> dtype = DType.float64,
    GpuDevice? device,
  }) : normalizedShape = List<int>.unmodifiable(normalizedShape) {
    if (this.normalizedShape.isEmpty) {
      throw ArgumentError.value(
        normalizedShape,
        'normalizedShape',
        'Must not be empty.',
      );
    }
    if (eps <= 0.0) {
      throw ArgumentError.value(eps, 'eps', 'Must be positive.');
    }
    if (!dtype.isFloating) {
      throw ArgumentError.value(
        dtype,
        'dtype',
        'Must be a floating-point DType.',
      );
    }
    final targetDevice = device ?? GpuDevice.defaultDevice;
    weight = registerParameter(
      'weight',
      GpuArray.ones(
        this.normalizedShape,
        dtype,
        device: targetDevice,
        requiresGrad: true,
      ),
    );
  }

  @override
  GpuArray<T> forward<T extends DTypeTag>(GpuArray<T> input) {
    checkNotDisposed();
    final xSquared = input * input;
    final meanSquared = xSquared.mean(axis: -1, keepDims: true);
    final rms = (meanSquared + eps).sqrt();
    final normalized = _preserveLayerDType(input, input / rms);
    return normalized * weight;
  }
}

/// Applies 1D Batch Normalization over a 2D (`[N, C]`) or 3D (`[N, C, L]`) input tensor.
final class BatchNorm1d extends Module {
  /// Number of feature channels $C$ expected in the input.
  final int numFeatures;

  /// Small constant added to the mini-batch variance for numerical stability.
  final double eps;

  /// Exponential moving average factor for running statistics.
  final double momentum;

  /// Whether this module learns affine scale ([weight]) and shift ([bias]) parameters.
  final bool affine;

  /// Whether this module tracks running mean and variance during training.
  final bool trackRunningStats;

  /// Optional learnable scale parameter ($\gamma$) of shape `[numFeatures]`.
  late final GpuArray<DTypeTag>? weight;

  /// Optional learnable shift parameter ($\beta$) of shape `[numFeatures]`.
  late final GpuArray<DTypeTag>? bias;

  /// Optional running mean buffer of shape `[numFeatures]`.
  late final GpuArray<DTypeTag>? runningMean;

  /// Optional running variance buffer of shape `[numFeatures]`.
  late final GpuArray<DTypeTag>? runningVar;

  /// Creates a [BatchNorm1d] layer for [numFeatures] channels.
  BatchNorm1d(
    this.numFeatures, {
    this.eps = 1e-5,
    this.momentum = 0.1,
    this.affine = true,
    this.trackRunningStats = true,
    DType<DTypeTag> dtype = DType.float64,
    GpuDevice? device,
  }) {
    if (numFeatures <= 0) {
      throw ArgumentError.value(
        numFeatures,
        'numFeatures',
        'Must be positive.',
      );
    }
    if (eps <= 0.0 || eps.isNaN) {
      throw ArgumentError.value(eps, 'eps', 'Must be positive.');
    }
    if (momentum < 0.0 || momentum > 1.0 || momentum.isNaN) {
      throw ArgumentError.value(
        momentum,
        'momentum',
        'Must be in the closed interval [0.0, 1.0].',
      );
    }
    if (!dtype.isFloating) {
      throw ArgumentError.value(
        dtype,
        'dtype',
        'Must be a floating-point DType.',
      );
    }

    final targetDevice = device ?? GpuDevice.defaultDevice;
    if (affine) {
      weight = registerParameter(
        'weight',
        GpuArray.ones(
          [numFeatures],
          dtype,
          device: targetDevice,
          requiresGrad: true,
        ),
      );
      bias = registerParameter(
        'bias',
        GpuArray.zeros(
          [numFeatures],
          dtype,
          device: targetDevice,
          requiresGrad: true,
        ),
      );
    } else {
      weight = null;
      bias = null;
    }

    if (trackRunningStats) {
      runningMean = registerBuffer(
        'runningMean',
        GpuArray.zeros([numFeatures], dtype, device: targetDevice),
      );
      runningVar = registerBuffer(
        'runningVar',
        GpuArray.ones([numFeatures], dtype, device: targetDevice),
      );
    } else {
      runningMean = null;
      runningVar = null;
    }
  }

  @override
  GpuArray<T> forward<T extends DTypeTag>(GpuArray<T> input) {
    checkNotDisposed();
    if ((input.rank != 2 && input.rank != 3) || input.shape[1] != numFeatures) {
      throw ArgumentError.value(
        input.shape,
        'input',
        'Must be a 2D or 3D tensor with channel dimension equal to numFeatures ($numFeatures).',
      );
    }
    return functional.batchNorm1d<T>(
      input,
      runningMean: runningMean,
      runningVar: runningVar,
      weight: weight,
      bias: bias,
      training: isTraining,
      momentum: momentum,
      eps: eps,
    );
  }
}
