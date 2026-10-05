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
import '../nn/nn_wgsl.dart';
import '../slice.dart';
import 'autograd_core.dart';
import 'autograd_wgsl.dart';

/// Backward node for Rectified Linear Unit (ReLU): $y = \max(0, x)$.
final class ReluBackward extends GradFn {
  /// Forward input tensor.
  final GpuArray<DTypeTag> x;

  /// Creates a [ReluBackward] node for [x].
  const ReluBackward(this.x);

  @override
  String get name => 'ReluBackward';

  @override
  List<GpuArray<DTypeTag>> get inputs => [x];

  @override
  List<GpuArray<DTypeTag>?> backward(GpuArray<DTypeTag> gradOutput) {
    if (!x.requiresGrad) return [null];
    final gradInput = GpuArray.empty(x.shape, x.dtype, device: x.device);
    dispatchUnaryActivationBackward(
      gradOutput: gradOutput,
      savedTensor: x,
      gradInput: gradInput,
      op: 'relu',
    );
    return [gradInput];
  }
}

/// Backward node for Sigmoid activation: $y = \sigma(x) = \frac{1}{1 + e^{-x}}$.
final class SigmoidBackward extends GradFn {
  /// Forward input tensor.
  final GpuArray<DTypeTag> input;

  /// Cached forward output tensor $y = \sigma(x)$.
  final GpuArray<DTypeTag>? y;

  /// Creates a [SigmoidBackward] node with forward [input] and optional cached output [y].
  const SigmoidBackward(this.input, [this.y]);

  @override
  String get name => 'SigmoidBackward';

  @override
  List<GpuArray<DTypeTag>> get inputs => [input];

  @override
  List<GpuArray<DTypeTag>?> backward(GpuArray<DTypeTag> gradOutput) {
    if (!input.requiresGrad) return [null];
    final gradInput = GpuArray.empty(
      input.shape,
      input.dtype,
      device: input.device,
    );
    dispatchUnaryActivationBackward(
      gradOutput: gradOutput,
      savedTensor: input,
      gradInput: gradInput,
      op: 'sigmoid',
    );
    return [gradInput];
  }
}

/// Backward node for Hyperbolic Tangent activation: $y = \tanh(x)$.
final class TanhBackward extends GradFn {
  /// Forward input tensor.
  final GpuArray<DTypeTag> input;

  /// Cached forward output tensor $y = \tanh(x)$.
  final GpuArray<DTypeTag>? y;

  /// Creates a [TanhBackward] node with forward [input] and optional cached output [y].
  const TanhBackward(this.input, [this.y]);

  @override
  String get name => 'TanhBackward';

  @override
  List<GpuArray<DTypeTag>> get inputs => [input];

  @override
  List<GpuArray<DTypeTag>?> backward(GpuArray<DTypeTag> gradOutput) {
    if (!input.requiresGrad) return [null];
    final gradInput = GpuArray.empty(
      input.shape,
      input.dtype,
      device: input.device,
    );
    dispatchUnaryActivationBackward(
      gradOutput: gradOutput,
      savedTensor: input,
      gradInput: gradInput,
      op: 'tanh',
    );
    return [gradInput];
  }
}

/// Backward node for Gaussian Error Linear Unit (GELU) activation.
final class GeluBackward extends GradFn {
  /// Forward input tensor.
  final GpuArray<DTypeTag> input;

  /// Creates a [GeluBackward] node for [input].
  const GeluBackward(this.input);

  @override
  String get name => 'GeluBackward';

  @override
  List<GpuArray<DTypeTag>> get inputs => [input];

  @override
  List<GpuArray<DTypeTag>?> backward(GpuArray<DTypeTag> gradOutput) {
    if (!input.requiresGrad) return [null];
    final gradInput = GpuArray.empty(
      input.shape,
      input.dtype,
      device: input.device,
    );
    dispatchUnaryActivationBackward(
      gradOutput: gradOutput,
      savedTensor: input,
      gradInput: gradInput,
      op: 'gelu',
    );
    return [gradInput];
  }
}

/// Backward node for Sigmoid Linear Unit (SiLU / Swish) activation.
final class SiluBackward extends GradFn {
  /// Forward input tensor.
  final GpuArray<DTypeTag> input;

  /// Creates a [SiluBackward] node for [input].
  const SiluBackward(this.input);

  @override
  String get name => 'SiluBackward';

  @override
  List<GpuArray<DTypeTag>> get inputs => [input];

  @override
  List<GpuArray<DTypeTag>?> backward(GpuArray<DTypeTag> gradOutput) {
    if (!input.requiresGrad) return [null];
    final gradInput = GpuArray.empty(
      input.shape,
      input.dtype,
      device: input.device,
    );
    dispatchUnaryActivationBackward(
      gradOutput: gradOutput,
      savedTensor: input,
      gradInput: gradInput,
      op: 'silu',
    );
    return [gradInput];
  }
}

/// Backward node for Leaky ReLU activation.
final class LeakyReluBackward extends GradFn {
  /// Forward input tensor.
  final GpuArray<DTypeTag> input;

  /// Negative slope coefficient.
  final double negativeSlope;

  /// Creates a [LeakyReluBackward] node for [input] with [negativeSlope].
  const LeakyReluBackward(this.input, {this.negativeSlope = 0.01});

  @override
  String get name => 'LeakyReluBackward';

  @override
  List<GpuArray<DTypeTag>> get inputs => [input];

  @override
  List<GpuArray<DTypeTag>?> backward(GpuArray<DTypeTag> gradOutput) {
    if (!input.requiresGrad) return [null];
    final gradInput = GpuArray.empty(
      input.shape,
      input.dtype,
      device: input.device,
    );
    dispatchUnaryActivationBackward(
      gradOutput: gradOutput,
      savedTensor: input,
      gradInput: gradInput,
      op: 'leaky_relu',
      param0: negativeSlope,
    );
    return [gradInput];
  }
}

/// Backward node for Exponential Linear Unit (ELU) activation.
final class EluBackward extends GradFn {
  /// Forward input tensor.
  final GpuArray<DTypeTag> input;

  /// Scale factor for negative inputs.
  final double alpha;

  /// Creates an [EluBackward] node for [input] with [alpha].
  const EluBackward(this.input, {this.alpha = 1.0});

  @override
  String get name => 'EluBackward';

  @override
  List<GpuArray<DTypeTag>> get inputs => [input];

  @override
  List<GpuArray<DTypeTag>?> backward(GpuArray<DTypeTag> gradOutput) {
    if (!input.requiresGrad) return [null];
    final gradInput = GpuArray.empty(
      input.shape,
      input.dtype,
      device: input.device,
    );
    dispatchUnaryActivationBackward(
      gradOutput: gradOutput,
      savedTensor: input,
      gradInput: gradInput,
      op: 'elu',
      param0: alpha,
    );
    return [gradInput];
  }
}

/// Backward node for Softplus activation.
final class SoftplusBackward extends GradFn {
  /// Forward input tensor.
  final GpuArray<DTypeTag> input;

  /// Inverse temperature scaling factor.
  final double beta;

  /// Linear stability threshold.
  final double threshold;

  /// Creates a [SoftplusBackward] node for [input] with [beta] and [threshold].
  const SoftplusBackward(this.input, {this.beta = 1.0, this.threshold = 20.0});

  @override
  String get name => 'SoftplusBackward';

  @override
  List<GpuArray<DTypeTag>> get inputs => [input];

  @override
  List<GpuArray<DTypeTag>?> backward(GpuArray<DTypeTag> gradOutput) {
    if (!input.requiresGrad) return [null];
    final gradInput = GpuArray.empty(
      input.shape,
      input.dtype,
      device: input.device,
    );
    dispatchUnaryActivationBackward(
      gradOutput: gradOutput,
      savedTensor: input,
      gradInput: gradInput,
      op: 'softplus',
      param0: beta,
      param1: threshold,
    );
    return [gradInput];
  }
}

/// Backward node for Softmax activation: $y = \text{softmax}(x, \text{axis})$.
final class SoftmaxBackward extends GradFn {
  /// Forward input tensor.
  final GpuArray<DTypeTag> input;

  /// Cached forward output tensor $y = \text{softmax}(x)$.
  final GpuArray<DTypeTag> y;

  /// Axis along which Softmax was computed.
  final int axis;

  /// Creates a [SoftmaxBackward] node with forward [input], cached output [y], and [axis].
  const SoftmaxBackward(this.input, this.y, {this.axis = -1});

  @override
  String get name => 'SoftmaxBackward';

  @override
  List<GpuArray<DTypeTag>> get inputs => [input];

  @override
  List<GpuArray<DTypeTag>?> backward(GpuArray<DTypeTag> gradOutput) {
    if (!input.requiresGrad) return [null];
    final gradTimesY = gradOutput * y;
    final sumGradTimesY = gradTimesY.sum(axis: axis, keepDims: true);
    gradTimesY.dispose();
    final centeredGrad = gradOutput - sumGradTimesY;
    sumGradTimesY.dispose();
    final gradInput = y * centeredGrad;
    centeredGrad.dispose();
    return [gradInput];
  }
}

/// Backward node for Log-Softmax activation: $y = \log(\text{softmax}(x, \text{axis}))$.
final class LogSoftmaxBackward extends GradFn {
  /// Forward input tensor.
  final GpuArray<DTypeTag> input;

  /// Cached forward output tensor $y = \text{logSoftmax}(x)$.
  final GpuArray<DTypeTag> y;

  /// Axis along which Log-Softmax was computed.
  final int axis;

  /// Creates a [LogSoftmaxBackward] node with forward [input], cached output [y], and [axis].
  const LogSoftmaxBackward(this.input, this.y, {this.axis = -1});

  @override
  String get name => 'LogSoftmaxBackward';

  @override
  List<GpuArray<DTypeTag>> get inputs => [input];

  @override
  List<GpuArray<DTypeTag>?> backward(GpuArray<DTypeTag> gradOutput) {
    if (!input.requiresGrad) return [null];
    final sumGrad = gradOutput.sum(axis: axis, keepDims: true);
    final expY = y.exp();
    final weightedSum = expY * sumGrad;
    expY.dispose();
    sumGrad.dispose();
    final gradInput = gradOutput - weightedSum;
    weightedSum.dispose();
    return [gradInput];
  }
}

/// Backward node for axis permutation / transposition.
final class TransposeBackward extends GradFn {
  /// Forward input tensor.
  final GpuArray<DTypeTag> input;

  /// Permutation of dimensions applied in the forward pass.
  final List<int> axes;

  /// Creates a [TransposeBackward] node for [input] and permutation [axes].
  TransposeBackward(this.input, List<int> axes)
    : axes = List<int>.unmodifiable(axes);

  @override
  String get name => 'TransposeBackward';

  @override
  List<GpuArray<DTypeTag>> get inputs => [input];

  @override
  List<GpuArray<DTypeTag>?> backward(GpuArray<DTypeTag> gradOutput) {
    if (!input.requiresGrad) return [null];
    final inverseAxes = List<int>.filled(axes.length, 0);
    for (var i = 0; i < axes.length; i++) {
      inverseAxes[axes[i]] = i;
    }
    return [gradOutput.transpose(inverseAxes)];
  }
}

/// Backward node for tensor reshape operations.
final class ReshapeBackward extends GradFn {
  /// Forward input tensor.
  final GpuArray<DTypeTag> input;

  /// Original shape of [input] prior to reshaping.
  final List<int> originalShape;

  /// Creates a [ReshapeBackward] node restoring [originalShape].
  ReshapeBackward(this.input, List<int> originalShape)
    : originalShape = List<int>.unmodifiable(originalShape);

  @override
  String get name => 'ReshapeBackward';

  @override
  List<GpuArray<DTypeTag>> get inputs => [input];

  @override
  List<GpuArray<DTypeTag>?> backward(GpuArray<DTypeTag> gradOutput) {
    if (!input.requiresGrad) return [null];
    return [gradOutput.reshape(originalShape)];
  }
}

/// Backward node for tensor squeeze operations.
final class SqueezeBackward extends GradFn {
  /// Forward input tensor.
  final GpuArray<DTypeTag> input;

  /// Original shape of [input] prior to squeezing unit dimensions.
  final List<int> originalShape;

  /// Creates a [SqueezeBackward] node restoring [originalShape].
  SqueezeBackward(this.input, List<int> originalShape)
    : originalShape = List<int>.unmodifiable(originalShape);

  @override
  String get name => 'SqueezeBackward';

  @override
  List<GpuArray<DTypeTag>> get inputs => [input];

  @override
  List<GpuArray<DTypeTag>?> backward(GpuArray<DTypeTag> gradOutput) {
    if (!input.requiresGrad) return [null];
    return [gradOutput.reshape(originalShape)];
  }
}

/// Backward node for tensor unsqueeze operations.
final class UnsqueezeBackward extends GradFn {
  /// Forward input tensor.
  final GpuArray<DTypeTag> input;

  /// Original shape of [input] prior to inserting a unit dimension.
  final List<int> originalShape;

  /// Creates an [UnsqueezeBackward] node restoring [originalShape].
  UnsqueezeBackward(this.input, List<int> originalShape)
    : originalShape = List<int>.unmodifiable(originalShape);

  @override
  String get name => 'UnsqueezeBackward';

  @override
  List<GpuArray<DTypeTag>> get inputs => [input];

  @override
  List<GpuArray<DTypeTag>?> backward(GpuArray<DTypeTag> gradOutput) {
    if (!input.requiresGrad) return [null];
    return [gradOutput.reshape(originalShape)];
  }
}

/// Backward node for subview slicing.
final class SliceBackward extends GradFn {
  /// Forward input tensor that was sliced.
  final GpuArray<DTypeTag> input;

  /// Slice specifications applied in the forward pass.
  final List<Object?> specs;

  /// Creates a [SliceBackward] node for [input] and slice [specs].
  SliceBackward(this.input, List<Object?> specs)
    : specs = List<Object?>.unmodifiable(specs);

  @override
  String get name => 'SliceBackward';

  @override
  List<GpuArray<DTypeTag>> get inputs => [input];

  @override
  List<GpuArray<DTypeTag>?> backward(GpuArray<DTypeTag> gradOutput) {
    if (!input.requiresGrad) return [null];
    final grad = GpuArray.zeros(input.shape, input.dtype, device: input.device);
    final sliceView = grad.slice(specs);
    dispatchSliceBackwardCopy(gradOutput, sliceView);
    sliceView.dispose();
    return [grad];
  }
}

/// Backward node for tensor concatenation along an axis.
final class ConcatenateBackward extends GradFn {
  @override
  final List<GpuArray<DTypeTag>> inputs;

  /// Axis along which the forward tensors were concatenated.
  final int axis;

  /// Creates a [ConcatenateBackward] node for [inputs] concatenated along [axis].
  ConcatenateBackward(List<GpuArray<DTypeTag>> inputs, {required this.axis})
    : inputs = List<GpuArray<DTypeTag>>.unmodifiable(inputs);

  @override
  String get name => 'ConcatenateBackward';

  @override
  List<GpuArray<DTypeTag>?> backward(GpuArray<DTypeTag> gradOutput) {
    final grads = <GpuArray<DTypeTag>?>[];
    var offset = 0;
    final rank = gradOutput.rank;
    final normalizedAxis = axis < 0 ? axis + rank : axis;

    for (final inputTensor in inputs) {
      final segmentLength = inputTensor.shape[normalizedAxis];
      if (inputTensor.requiresGrad) {
        final sliceSpecs = List<Object>.generate(rank, (dimension) {
          if (dimension == normalizedAxis) {
            return Slice(offset, offset + segmentLength);
          }
          return const All();
        });
        final sliceView = gradOutput.slice(sliceSpecs);
        grads.add(sliceView.copy());
        sliceView.dispose();
      } else {
        grads.add(null);
      }
      offset += segmentLength;
    }
    return grads;
  }
}

/// Backward node for embedding table lookup.
final class EmbeddingBackward extends GradFn {
  /// Embedding weight matrix of shape `[numEmbeddings, embeddingDim]`.
  final GpuArray<DTypeTag> weight;

  /// Integer index tensor used in the forward lookup.
  final GpuArray<DTypeTag> indices;

  /// Number of rows in the embedding dictionary.
  final int numEmbeddings;

  /// Dimension of each embedding vector.
  final int embeddingDim;

  /// Creates an [EmbeddingBackward] node.
  const EmbeddingBackward(
    this.weight,
    this.indices,
    this.numEmbeddings,
    this.embeddingDim,
  );

  @override
  String get name => 'EmbeddingBackward';

  @override
  List<GpuArray<DTypeTag>> get inputs => [weight];

  @override
  List<GpuArray<DTypeTag>?> backward(GpuArray<DTypeTag> gradOutput) {
    if (!weight.requiresGrad) return [null];
    final gradWeight = GpuArray.zeros(
      weight.shape,
      weight.dtype,
      device: weight.device,
    );
    dispatchEmbeddingBackward(
      gradOutput: gradOutput,
      indices: indices,
      gradWeight: gradWeight,
      numEmbeddings: numEmbeddings,
      embeddingDim: embeddingDim,
    );
    return [gradWeight];
  }
}
