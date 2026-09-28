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

// ignore_for_file: non_constant_identifier_names
import 'dart:ffi' as ffi;

import '../backend/compute_engine.dart';
import '../dtype.dart';
import '../gpu_array.dart';
import '../slice.dart';

/// Reduction mode applied to elementwise or per-sample loss values.
enum LossReduction {
  /// Returns the unreduced loss tensor without aggregation.
  none,

  /// Returns the arithmetic mean of all loss elements.
  mean,

  /// Returns the sum of all loss elements.
  sum,
}

/// Global flag controlling whether gradient tracking is enabled.
bool _gradEnabled = true;

/// Whether automatic differentiation graph recording is currently enabled.
bool get isGradEnabled => _gradEnabled;

/// Disables gradient calculation during the synchronous execution of [body].
R noGrad<R>(R Function() body) {
  final previous = _gradEnabled;
  _gradEnabled = false;
  try {
    return body();
  } finally {
    _gradEnabled = previous;
  }
}

/// Disables gradient calculation during the synchronous execution of [body].
///
/// Alias for [noGrad] provided for PyTorch naming parity.
R no_grad<R>(R Function() body) => noGrad(body);

/// Base class for reverse-mode automatic differentiation graph nodes.
abstract class GradFn {
  /// Creates a [GradFn] node.
  const GradFn();

  /// Human-readable operation name for this backward node.
  String get name;

  /// Input tensors connected to this operation in the forward pass.
  List<GpuArray<DTypeTag>> get inputs;

  /// Computes the vector-Jacobian product (VJP) given upstream [gradOutput].
  List<GpuArray<DTypeTag>?> backward(GpuArray<DTypeTag> gradOutput);
}

/// Reduces [grad] across broadcasted dimensions to match [targetShape].
GpuArray<DTypeTag> _unbroadcast(
  GpuArray<DTypeTag> grad,
  List<int> targetShape, {
  bool copyIfUnchanged = true,
}) {
  if (ShapeUtils.areEqual(grad.shape, targetShape)) {
    return copyIfUnchanged ? grad.copy() : grad;
  }

  var current = grad;
  final rankDifference = current.rank - targetShape.length;

  for (var i = 0; i < rankDifference; i++) {
    final next = current.sum(axis: 0);
    if (!identical(current, grad)) {
      current.dispose();
    }
    current = next;
  }

  for (var i = 0; i < targetShape.length; i++) {
    if (targetShape[i] == 1 && current.shape[i] > 1) {
      final next = current.sum(axis: i, keepDims: true);
      if (!identical(current, grad)) {
        current.dispose();
      }
      current = next;
    }
  }

  return current;
}

/// Reduces [intermediate] to [targetShape] and disposes [intermediate] if a new tensor is allocated.
GpuArray<DTypeTag> _unbroadcastOwned(
  GpuArray<DTypeTag> intermediate,
  List<int> targetShape,
) {
  final reduced = _unbroadcast(
    intermediate,
    targetShape,
    copyIfUnchanged: false,
  );
  if (!identical(reduced, intermediate)) {
    intermediate.dispose();
  }
  return reduced;
}

/// Reads a numerical element at flat [index] from [array], respecting strides and offset.
double _readElementAsDouble(GpuArray<DTypeTag> array, int index) {
  if (array.isContiguous) {
    return ComputeEngine.readValue(
      array.buffer,
      array.dtype,
      index,
      offsetElements: array.offsetElements,
    );
  }
  var remaining = index;
  var offset = array.offsetElements;
  final cStrides = ShapeUtils.computeCStrides(array.shape);
  for (var d = 0; d < array.rank; d++) {
    final coordinate = remaining ~/ cStrides[d];
    remaining %= cStrides[d];
    offset += coordinate * array.strides[d];
  }
  return ComputeEngine.readValue(array.buffer, array.dtype, offset);
}

/// Reads an integer element at flat [index] from [array], respecting strides and offset.
int _readElementAsInt(GpuArray<DTypeTag> array, int index) {
  return _readElementAsDouble(array, index).toInt();
}

/// Backward node for elementwise addition: $y = a + b$.
final class AddBackward extends GradFn {
  /// First forward operand.
  final GpuArray<DTypeTag> a;

  /// Second forward operand.
  final GpuArray<DTypeTag> b;

  /// Creates an [AddBackward] node for operands [a] and [b].
  const AddBackward(this.a, this.b);

  @override
  String get name => 'AddBackward';

  @override
  List<GpuArray<DTypeTag>> get inputs => [a, b];

  @override
  List<GpuArray<DTypeTag>?> backward(GpuArray<DTypeTag> gradOutput) {
    final gradA = a.requiresGrad ? _unbroadcast(gradOutput, a.shape) : null;
    final gradB = b.requiresGrad ? _unbroadcast(gradOutput, b.shape) : null;
    return [gradA, gradB];
  }
}

/// Backward node for elementwise subtraction: $y = a - b$.
final class SubBackward extends GradFn {
  /// First forward operand (minuend).
  final GpuArray<DTypeTag> a;

  /// Second forward operand (subtrahend).
  final GpuArray<DTypeTag> b;

  /// Creates a [SubBackward] node for operands [a] and [b].
  const SubBackward(this.a, this.b);

  @override
  String get name => 'SubBackward';

  @override
  List<GpuArray<DTypeTag>> get inputs => [a, b];

  @override
  List<GpuArray<DTypeTag>?> backward(GpuArray<DTypeTag> gradOutput) {
    final gradA = a.requiresGrad ? _unbroadcast(gradOutput, a.shape) : null;
    final gradB = b.requiresGrad
        ? _unbroadcastOwned(gradOutput.negate(), b.shape)
        : null;
    return [gradA, gradB];
  }
}

/// Backward node for elementwise multiplication: $y = a \cdot b$.
final class MulBackward extends GradFn {
  /// First forward operand.
  final GpuArray<DTypeTag> a;

  /// Second forward operand.
  final GpuArray<DTypeTag> b;

  /// Creates a [MulBackward] node for operands [a] and [b].
  const MulBackward(this.a, this.b);

  @override
  String get name => 'MulBackward';

  @override
  List<GpuArray<DTypeTag>> get inputs => [a, b];

  @override
  List<GpuArray<DTypeTag>?> backward(GpuArray<DTypeTag> gradOutput) {
    final gradA = a.requiresGrad
        ? _unbroadcastOwned(gradOutput * b, a.shape)
        : null;
    final gradB = b.requiresGrad
        ? _unbroadcastOwned(gradOutput * a, b.shape)
        : null;
    return [gradA, gradB];
  }
}

/// Backward node for elementwise division: $y = a / b$.
final class DivBackward extends GradFn {
  /// Numerator forward operand.
  final GpuArray<DTypeTag> a;

  /// Denominator forward operand.
  final GpuArray<DTypeTag> b;

  /// Creates a [DivBackward] node for operands [a] and [b].
  const DivBackward(this.a, this.b);

  @override
  String get name => 'DivBackward';

  @override
  List<GpuArray<DTypeTag>> get inputs => [a, b];

  @override
  List<GpuArray<DTypeTag>?> backward(GpuArray<DTypeTag> gradOutput) {
    final gradA = a.requiresGrad
        ? _unbroadcastOwned(gradOutput / b, a.shape)
        : null;
    GpuArray<DTypeTag>? gradB;
    if (b.requiresGrad) {
      final negatedA = a.negate();
      final numerator = gradOutput * negatedA;
      negatedA.dispose();
      final denominator = b * b;
      final quotient = numerator / denominator;
      numerator.dispose();
      denominator.dispose();
      gradB = _unbroadcastOwned(quotient, b.shape);
    }
    return [gradA, gradB];
  }
}

/// Backward node for elementwise power: $y = a^b$.
final class PowBackward extends GradFn {
  /// Base forward operand.
  final GpuArray<DTypeTag> a;

  /// Exponent forward operand.
  final GpuArray<DTypeTag> b;

  /// Creates a [PowBackward] node for base [a] and exponent [b].
  const PowBackward(this.a, this.b);

  @override
  String get name => 'PowBackward';

  @override
  List<GpuArray<DTypeTag>> get inputs => [a, b];

  @override
  List<GpuArray<DTypeTag>?> backward(GpuArray<DTypeTag> gradOutput) {
    GpuArray<DTypeTag>? gradA;
    GpuArray<DTypeTag>? gradB;

    if (a.requiresGrad) {
      final bMinusOne = b - 1.0;
      final aPower = a.pow(bMinusOne);
      bMinusOne.dispose();
      final gradTimesB = gradOutput * b;
      final rawGradA = gradTimesB * aPower;
      gradTimesB.dispose();
      aPower.dispose();
      gradA = _unbroadcastOwned(rawGradA, a.shape);
    }
    if (b.requiresGrad) {
      final forwardPower = a.pow(b);
      final logA = a.log();
      final gradTimesPower = gradOutput * forwardPower;
      forwardPower.dispose();
      final rawGradB = gradTimesPower * logA;
      gradTimesPower.dispose();
      logA.dispose();
      gradB = _unbroadcastOwned(rawGradB, b.shape);
    }

    return [gradA, gradB];
  }
}

/// Backward node for elementwise negation: $y = -x$.
final class NegBackward extends GradFn {
  /// Forward input tensor.
  final GpuArray<DTypeTag> input;

  /// Creates a [NegBackward] node for [input].
  const NegBackward(this.input);

  @override
  String get name => 'NegBackward';

  @override
  List<GpuArray<DTypeTag>> get inputs => [input];

  @override
  List<GpuArray<DTypeTag>?> backward(GpuArray<DTypeTag> gradOutput) {
    if (!input.requiresGrad) return [null];
    return [gradOutput.negate()];
  }
}

/// Backward node for elementwise square root: $y = \sqrt{x}$.
final class SqrtBackward extends GradFn {
  /// Forward input tensor.
  final GpuArray<DTypeTag> input;

  /// Cached forward output tensor $y = \sqrt{x}$.
  final GpuArray<DTypeTag> y;

  /// Creates a [SqrtBackward] node with forward [input] and cached output [y].
  const SqrtBackward(this.input, this.y);

  @override
  String get name => 'SqrtBackward';

  @override
  List<GpuArray<DTypeTag>> get inputs => [input];

  @override
  List<GpuArray<DTypeTag>?> backward(GpuArray<DTypeTag> gradOutput) {
    if (!input.requiresGrad) return [null];
    final twoY = y * 2.0;
    final gradInput = gradOutput / twoY;
    twoY.dispose();
    return [gradInput];
  }
}

/// Backward node for elementwise exponential: $y = e^x$.
final class ExpBackward extends GradFn {
  /// Forward input tensor.
  final GpuArray<DTypeTag> input;

  /// Cached forward output tensor $y = e^x$.
  final GpuArray<DTypeTag> y;

  /// Creates an [ExpBackward] node with forward [input] and cached output [y].
  const ExpBackward(this.input, this.y);

  @override
  String get name => 'ExpBackward';

  @override
  List<GpuArray<DTypeTag>> get inputs => [input];

  @override
  List<GpuArray<DTypeTag>?> backward(GpuArray<DTypeTag> gradOutput) {
    if (!input.requiresGrad) return [null];
    return [gradOutput * y];
  }
}

/// Backward node for elementwise natural logarithm: $y = \ln(x)$.
final class LogBackward extends GradFn {
  /// Forward input tensor.
  final GpuArray<DTypeTag> input;

  /// Creates a [LogBackward] node for [input].
  const LogBackward(this.input);

  @override
  String get name => 'LogBackward';

  @override
  List<GpuArray<DTypeTag>> get inputs => [input];

  @override
  List<GpuArray<DTypeTag>?> backward(GpuArray<DTypeTag> gradOutput) {
    if (!input.requiresGrad) return [null];
    return [gradOutput / input];
  }
}

/// Backward node for matrix multiplication: $Y = A B$.
final class MatmulBackward extends GradFn {
  /// Left matrix operand.
  final GpuArray<DTypeTag> a;

  /// Right matrix operand.
  final GpuArray<DTypeTag> b;

  /// Creates a [MatmulBackward] node for operands [a] and [b].
  const MatmulBackward(this.a, this.b);

  @override
  String get name => 'MatmulBackward';

  @override
  List<GpuArray<DTypeTag>> get inputs => [a, b];

  @override
  List<GpuArray<DTypeTag>?> backward(GpuArray<DTypeTag> gradOutput) {
    GpuArray<DTypeTag>? gradA;
    GpuArray<DTypeTag>? gradB;

    if (a.requiresGrad) {
      GpuArray<DTypeTag> rawGradA;
      if (a.rank == 1 && b.rank == 1) {
        rawGradA = b * gradOutput;
      } else if (a.rank == 1) {
        final bTransposed = b.swapaxes(-1, -2);
        rawGradA = gradOutput.matmul(bTransposed);
        bTransposed.dispose();
      } else if (b.rank == 1) {
        final gradExpanded = gradOutput.unsqueeze(-1);
        final bExpanded = b.unsqueeze(0);
        rawGradA = gradExpanded.matmul(bExpanded);
        gradExpanded.dispose();
        bExpanded.dispose();
      } else {
        final bTransposed = b.swapaxes(-1, -2);
        rawGradA = gradOutput.matmul(bTransposed);
        bTransposed.dispose();
      }
      gradA = _unbroadcastOwned(rawGradA, a.shape);
    }

    if (b.requiresGrad) {
      GpuArray<DTypeTag> rawGradB;
      if (a.rank == 1 && b.rank == 1) {
        rawGradB = a * gradOutput;
      } else if (a.rank == 1) {
        final aExpanded = a.unsqueeze(-1);
        final gradExpanded = gradOutput.unsqueeze(0);
        rawGradB = aExpanded.matmul(gradExpanded);
        aExpanded.dispose();
        gradExpanded.dispose();
      } else {
        final aTransposed = a.swapaxes(-1, -2);
        rawGradB = aTransposed.matmul(gradOutput);
        aTransposed.dispose();
      }
      gradB = _unbroadcastOwned(rawGradB, b.shape);
    }

    return [gradA, gradB];
  }
}

/// Backward node for sum reduction: $y = \sum a$.
final class SumBackward extends GradFn {
  /// Forward input tensor.
  final GpuArray<DTypeTag> a;

  /// Reduction axis, or `null` when reduced over all elements.
  final int? axis;

  /// Whether reduced dimensions were retained with size 1.
  final bool keepDims;

  /// Creates a [SumBackward] node for [a].
  const SumBackward(this.a, {this.axis, this.keepDims = false});

  @override
  String get name => 'SumBackward';

  @override
  List<GpuArray<DTypeTag>> get inputs => [a];

  @override
  List<GpuArray<DTypeTag>?> backward(GpuArray<DTypeTag> gradOutput) {
    if (!a.requiresGrad) return [null];

    var expandedGrad = gradOutput;
    var ownsExpanded = false;
    if (axis != null && !keepDims) {
      final normalizedAxis = axis! < 0 ? axis! + a.rank : axis!;
      expandedGrad = gradOutput.unsqueeze(normalizedAxis);
      ownsExpanded = true;
    }

    final ones = GpuArray.ones(a.shape, a.dtype, device: a.device);
    final result = expandedGrad * ones;
    ones.dispose();
    if (ownsExpanded) {
      expandedGrad.dispose();
    }
    return [result];
  }
}

/// Backward node for arithmetic mean reduction: $y = \text{mean}(a)$.
final class MeanBackward extends GradFn {
  /// Forward input tensor.
  final GpuArray<DTypeTag> a;

  /// Reduction axis, or `null` when reduced over all elements.
  final int? axis;

  /// Whether reduced dimensions were retained with size 1.
  final bool keepDims;

  /// Creates a [MeanBackward] node for [a].
  const MeanBackward(this.a, {this.axis, this.keepDims = false});

  @override
  String get name => 'MeanBackward';

  @override
  List<GpuArray<DTypeTag>> get inputs => [a];

  @override
  List<GpuArray<DTypeTag>?> backward(GpuArray<DTypeTag> gradOutput) {
    if (!a.requiresGrad) return [null];

    final count = (axis == null)
        ? a.size
        : a.shape[axis! < 0 ? axis! + a.rank : axis!];
    final scale = 1.0 / count;

    final scaledGrad = gradOutput * scale;
    var expandedGrad = scaledGrad;
    if (axis != null && !keepDims) {
      final normalizedAxis = axis! < 0 ? axis! + a.rank : axis!;
      expandedGrad = scaledGrad.unsqueeze(normalizedAxis);
      scaledGrad.dispose();
    }

    final ones = GpuArray.ones(a.shape, a.dtype, device: a.device);
    final result = expandedGrad * ones;
    ones.dispose();
    expandedGrad.dispose();
    return [result];
  }
}

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
    final positiveMask = x.greater(0.0);
    final mask = positiveMask.astype(x.dtype);
    positiveMask.dispose();
    final result = gradOutput * mask;
    mask.dispose();
    return [result];
  }
}

/// Backward node for Sigmoid activation: $y = \sigma(x) = \frac{1}{1 + e^{-x}}$.
final class SigmoidBackward extends GradFn {
  /// Forward input tensor.
  final GpuArray<DTypeTag> input;

  /// Cached forward output tensor $y = \sigma(x)$.
  final GpuArray<DTypeTag> y;

  /// Creates a [SigmoidBackward] node with forward [input] and cached output [y].
  const SigmoidBackward(this.input, this.y);

  @override
  String get name => 'SigmoidBackward';

  @override
  List<GpuArray<DTypeTag>> get inputs => [input];

  @override
  List<GpuArray<DTypeTag>?> backward(GpuArray<DTypeTag> gradOutput) {
    if (!input.requiresGrad) return [null];
    final negatedY = y.negate();
    final oneMinusY = negatedY + 1.0;
    negatedY.dispose();
    final gradTimesY = gradOutput * y;
    final result = gradTimesY * oneMinusY;
    gradTimesY.dispose();
    oneMinusY.dispose();
    return [result];
  }
}

/// Backward node for Hyperbolic Tangent activation: $y = \tanh(x)$.
final class TanhBackward extends GradFn {
  /// Forward input tensor.
  final GpuArray<DTypeTag> input;

  /// Cached forward output tensor $y = \tanh(x)$.
  final GpuArray<DTypeTag> y;

  /// Creates a [TanhBackward] node with forward [input] and cached output [y].
  const TanhBackward(this.input, this.y);

  @override
  String get name => 'TanhBackward';

  @override
  List<GpuArray<DTypeTag>> get inputs => [input];

  @override
  List<GpuArray<DTypeTag>?> backward(GpuArray<DTypeTag> gradOutput) {
    if (!input.requiresGrad) return [null];
    final ySquared = y * y;
    final negatedYSquared = ySquared.negate();
    ySquared.dispose();
    final oneMinusYSquared = negatedYSquared + 1.0;
    negatedYSquared.dispose();
    final result = gradOutput * oneMinusYSquared;
    oneMinusYSquared.dispose();
    return [result];
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
    final totalElements = sliceView.size;
    final viewRank = sliceView.rank;
    final viewCStrides = ShapeUtils.computeCStrides(sliceView.shape);

    for (var i = 0; i < totalElements; i++) {
      var remaining = i;
      var srcOffset = gradOutput.offsetElements;
      var dstOffset = sliceView.offsetElements;
      for (var d = 0; d < viewRank; d++) {
        final coordinate = remaining ~/ viewCStrides[d];
        remaining %= viewCStrides[d];
        if (d < gradOutput.rank) {
          srcOffset += coordinate * gradOutput.strides[d];
        }
        dstOffset += coordinate * sliceView.strides[d];
      }
      final value = ComputeEngine.readAny(
        gradOutput.buffer,
        gradOutput.dtype,
        srcOffset,
      );
      ComputeEngine.writeAny(
        sliceView.buffer,
        sliceView.dtype,
        dstOffset,
        value,
      );
    }
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
    final indexCount = indices.size;
    final contiguousGrad = gradOutput.isContiguous
        ? gradOutput
        : gradOutput.copy();

    try {
      if (weight.dtype == DType.float64 &&
          contiguousGrad.dtype == DType.float64) {
        final dstPtr = gradWeight.buffer.pointer.cast<ffi.Double>();
        final srcPtr = contiguousGrad.buffer.pointer.cast<ffi.Double>();
        final srcBase = contiguousGrad.offsetElements;
        for (var i = 0; i < indexCount; i++) {
          final tokenIndex = _readElementAsInt(indices, i);
          if (tokenIndex < 0 || tokenIndex >= numEmbeddings) continue;
          final dstRowOffset = tokenIndex * embeddingDim;
          final srcRowOffset = srcBase + i * embeddingDim;
          for (var d = 0; d < embeddingDim; d++) {
            dstPtr[dstRowOffset + d] += srcPtr[srcRowOffset + d];
          }
        }
      } else {
        for (var i = 0; i < indexCount; i++) {
          final tokenIndex = _readElementAsInt(indices, i);
          if (tokenIndex < 0 || tokenIndex >= numEmbeddings) continue;
          final dstRowOffset = tokenIndex * embeddingDim;
          final srcRowOffset = i * embeddingDim;
          for (var d = 0; d < embeddingDim; d++) {
            final currentVal = ComputeEngine.readValue(
              gradWeight.buffer,
              gradWeight.dtype,
              dstRowOffset + d,
            );
            final addVal = ComputeEngine.readValue(
              contiguousGrad.buffer,
              contiguousGrad.dtype,
              srcRowOffset + d,
              offsetElements: contiguousGrad.offsetElements,
            );
            ComputeEngine.writeValue(
              gradWeight.buffer,
              gradWeight.dtype,
              dstRowOffset + d,
              currentVal + addVal,
            );
          }
        }
      }
    } finally {
      if (!identical(contiguousGrad, gradOutput)) {
        contiguousGrad.dispose();
      }
    }

    return [gradWeight];
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

    final grad = probabilities.copy();
    for (var i = 0; i < numSamples; i++) {
      final targetClass = _readElementAsInt(targets, i);
      if (targetClass >= 0 && targetClass < numClasses) {
        final flatIndex = i * numClasses + targetClass;
        final currentProb = ComputeEngine.readValue(
          grad.buffer,
          grad.dtype,
          flatIndex,
          offsetElements: grad.offsetElements,
        );
        ComputeEngine.writeValue(
          grad.buffer,
          grad.dtype,
          flatIndex,
          currentProb - 1.0,
          offsetElements: grad.offsetElements,
        );
      }
    }

    switch (reduction) {
      case LossReduction.mean:
        final scale = 1.0 / numSamples;
        final upstreamScalar = (gradOutput.rank == 0 || gradOutput.size == 1)
            ? (gradOutput.scalar as num).toDouble()
            : 1.0;
        final scaled = grad * (scale * upstreamScalar);
        grad.dispose();
        return [scaled];
      case LossReduction.sum:
        final upstreamScalar = (gradOutput.rank == 0 || gradOutput.size == 1)
            ? (gradOutput.scalar as num).toDouble()
            : 1.0;
        final scaled = grad * upstreamScalar;
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
  final inHeight = input.shape[2];
  final inWidth = input.shape[3];
  final patchSize = inChannels * kernelSize * kernelSize;
  final totalPatches = batchSize * outHeight * outWidth;

  final columns = GpuArray.zeros(
    [totalPatches, patchSize],
    input.dtype,
    device: input.device,
  );
  final contiguousInput = input.isContiguous ? input : input.copy();

  try {
    if (input.dtype == DType.float64) {
      final srcPtr = contiguousInput.buffer.pointer.cast<ffi.Double>();
      final dstPtr = columns.buffer.pointer.cast<ffi.Double>();
      final srcBase = contiguousInput.offsetElements;

      for (var b = 0; b < batchSize; b++) {
        for (var oh = 0; oh < outHeight; oh++) {
          final ihStart = oh * stride - padding;
          for (var ow = 0; ow < outWidth; ow++) {
            final iwStart = ow * stride - padding;
            final patchRow = ((b * outHeight + oh) * outWidth + ow) * patchSize;

            for (var ic = 0; ic < inChannels; ic++) {
              final channelBase = (b * inChannels + ic) * inHeight;
              final patchChannelBase = patchRow + ic * kernelSize * kernelSize;
              for (var kh = 0; kh < kernelSize; kh++) {
                final ih = ihStart + kh;
                if (ih < 0 || ih >= inHeight) continue;
                final inputRowBase = (channelBase + ih) * inWidth;
                final patchRowBase = patchChannelBase + kh * kernelSize;
                for (var kw = 0; kw < kernelSize; kw++) {
                  final iw = iwStart + kw;
                  if (iw < 0 || iw >= inWidth) continue;
                  dstPtr[patchRowBase + kw] =
                      srcPtr[srcBase + inputRowBase + iw];
                }
              }
            }
          }
        }
      }
    } else {
      for (var b = 0; b < batchSize; b++) {
        for (var oh = 0; oh < outHeight; oh++) {
          final ihStart = oh * stride - padding;
          for (var ow = 0; ow < outWidth; ow++) {
            final iwStart = ow * stride - padding;
            final patchRow = ((b * outHeight + oh) * outWidth + ow) * patchSize;

            for (var ic = 0; ic < inChannels; ic++) {
              final channelBase = (b * inChannels + ic) * inHeight;
              final patchChannelBase = patchRow + ic * kernelSize * kernelSize;
              for (var kh = 0; kh < kernelSize; kh++) {
                final ih = ihStart + kh;
                if (ih < 0 || ih >= inHeight) continue;
                final inputRowBase = (channelBase + ih) * inWidth;
                final patchRowBase = patchChannelBase + kh * kernelSize;
                for (var kw = 0; kw < kernelSize; kw++) {
                  final iw = iwStart + kw;
                  if (iw < 0 || iw >= inWidth) continue;
                  final val = ComputeEngine.readValue(
                    contiguousInput.buffer,
                    contiguousInput.dtype,
                    inputRowBase + iw,
                    offsetElements: contiguousInput.offsetElements,
                  );
                  ComputeEngine.writeValue(
                    columns.buffer,
                    columns.dtype,
                    patchRowBase + kw,
                    val,
                  );
                }
              }
            }
          }
        }
      }
    }
  } finally {
    if (!identical(contiguousInput, input)) {
      contiguousInput.dispose();
    }
  }

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
  final batchSize = inputShape[0];
  final inChannels = inputShape[1];
  final inHeight = inputShape[2];
  final inWidth = inputShape[3];
  final patchSize = inChannels * kernelSize * kernelSize;

  final gradInput = GpuArray.zeros(
    inputShape,
    gradColumns.dtype,
    device: gradColumns.device,
  );
  final contiguousColumns = gradColumns.isContiguous
      ? gradColumns
      : gradColumns.copy();

  try {
    if (gradColumns.dtype == DType.float64) {
      final srcPtr = contiguousColumns.buffer.pointer.cast<ffi.Double>();
      final dstPtr = gradInput.buffer.pointer.cast<ffi.Double>();
      final srcBase = contiguousColumns.offsetElements;

      for (var b = 0; b < batchSize; b++) {
        for (var oh = 0; oh < outHeight; oh++) {
          final ihStart = oh * stride - padding;
          for (var ow = 0; ow < outWidth; ow++) {
            final iwStart = ow * stride - padding;
            final patchRow = ((b * outHeight + oh) * outWidth + ow) * patchSize;

            for (var ic = 0; ic < inChannels; ic++) {
              final channelBase = (b * inChannels + ic) * inHeight;
              final patchChannelBase = patchRow + ic * kernelSize * kernelSize;
              for (var kh = 0; kh < kernelSize; kh++) {
                final ih = ihStart + kh;
                if (ih < 0 || ih >= inHeight) continue;
                final inputRowBase = (channelBase + ih) * inWidth;
                final patchRowBase = patchChannelBase + kh * kernelSize;
                for (var kw = 0; kw < kernelSize; kw++) {
                  final iw = iwStart + kw;
                  if (iw < 0 || iw >= inWidth) continue;
                  dstPtr[inputRowBase + iw] +=
                      srcPtr[srcBase + patchRowBase + kw];
                }
              }
            }
          }
        }
      }
    } else {
      for (var b = 0; b < batchSize; b++) {
        for (var oh = 0; oh < outHeight; oh++) {
          final ihStart = oh * stride - padding;
          for (var ow = 0; ow < outWidth; ow++) {
            final iwStart = ow * stride - padding;
            final patchRow = ((b * outHeight + oh) * outWidth + ow) * patchSize;

            for (var ic = 0; ic < inChannels; ic++) {
              final channelBase = (b * inChannels + ic) * inHeight;
              final patchChannelBase = patchRow + ic * kernelSize * kernelSize;
              for (var kh = 0; kh < kernelSize; kh++) {
                final ih = ihStart + kh;
                if (ih < 0 || ih >= inHeight) continue;
                final inputRowBase = (channelBase + ih) * inWidth;
                final patchRowBase = patchChannelBase + kh * kernelSize;
                for (var kw = 0; kw < kernelSize; kw++) {
                  final iw = iwStart + kw;
                  if (iw < 0 || iw >= inWidth) continue;
                  final currentVal = ComputeEngine.readValue(
                    gradInput.buffer,
                    gradInput.dtype,
                    inputRowBase + iw,
                  );
                  final addVal = ComputeEngine.readValue(
                    contiguousColumns.buffer,
                    contiguousColumns.dtype,
                    patchRowBase + kw,
                    offsetElements: contiguousColumns.offsetElements,
                  );
                  ComputeEngine.writeValue(
                    gradInput.buffer,
                    gradInput.dtype,
                    inputRowBase + iw,
                    currentVal + addVal,
                  );
                }
              }
            }
          }
        }
      }
    }
  } finally {
    if (!identical(contiguousColumns, gradColumns)) {
      contiguousColumns.dispose();
    }
  }

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

/// Executes reverse-mode automatic differentiation starting from [root].
///
/// The [root] tensor must have [GpuArray.requiresGrad] set to `true`.
/// If [root] contains more than one element, an explicit upstream [gradient]
/// matching [root]'s shape must be provided.
void runBackward(
  GpuArray<DTypeTag> root, {
  GpuArray<DTypeTag>? gradient,
  bool retainGraph = false,
}) {
  if (!root.requiresGrad) {
    throw StateError(
      'Cannot call backward() on a tensor with requiresGrad = false.',
    );
  }

  noGrad(() {
    final bool createdSeed;
    final GpuArray<DTypeTag> seedGrad;
    if (gradient != null) {
      seedGrad = gradient;
      createdSeed = false;
    } else if (root.rank == 0 || root.size == 1) {
      seedGrad = GpuArray.ones(root.shape, root.dtype, device: root.device);
      createdSeed = true;
    } else {
      throw ArgumentError.value(
        gradient,
        'gradient',
        'Must be provided when root tensor is not a scalar (shape ${root.shape}).',
      );
    }

    if (root.grad == null) {
      root.grad = seedGrad;
    } else {
      final previousGrad = root.grad!;
      root.grad = previousGrad + seedGrad;
      previousGrad.dispose();
      if (createdSeed) {
        seedGrad.dispose();
      }
    }

    final orderedNodes = <GpuArray<DTypeTag>>[];
    final visited = <GpuArray<DTypeTag>>{};

    void buildTopologicalOrder(GpuArray<DTypeTag> node) {
      if (!visited.add(node)) return;

      final functionNode = node.gradFn;
      if (functionNode != null) {
        for (final input in functionNode.inputs) {
          if (input.requiresGrad) {
            buildTopologicalOrder(input);
          }
        }
      }
      orderedNodes.add(node);
    }

    buildTopologicalOrder(root);

    for (var i = orderedNodes.length - 1; i >= 0; i--) {
      final node = orderedNodes[i];
      final functionNode = node.gradFn;
      final nodeGrad = node.grad;

      if (functionNode != null && nodeGrad != null) {
        final inputGrads = functionNode.backward(nodeGrad);
        final inputs = functionNode.inputs;

        for (var j = 0; j < inputs.length; j++) {
          final inputTensor = inputs[j];
          final incomingGrad = (j < inputGrads.length) ? inputGrads[j] : null;

          if (incomingGrad != null) {
            if (inputTensor.requiresGrad) {
              if (inputTensor.grad == null) {
                inputTensor.grad = incomingGrad;
              } else {
                final previousGrad = inputTensor.grad!;
                inputTensor.grad = previousGrad + incomingGrad;
                previousGrad.dispose();
                if (!identical(incomingGrad, nodeGrad)) {
                  incomingGrad.dispose();
                }
              }
            } else if (!identical(incomingGrad, nodeGrad)) {
              incomingGrad.dispose();
            }
          }
        }

        if (!retainGraph) {
          node.gradFn = null;
          if (!identical(node, root)) {
            nodeGrad.dispose();
            node.grad = null;
          }
        }
      }
    }
  });
}
