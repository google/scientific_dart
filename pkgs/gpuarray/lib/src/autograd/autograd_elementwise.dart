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
    final gradA = a.requiresGrad ? unbroadcast(gradOutput, a.shape) : null;
    final gradB = b.requiresGrad ? unbroadcast(gradOutput, b.shape) : null;
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
    final gradA = a.requiresGrad ? unbroadcast(gradOutput, a.shape) : null;
    final gradB = b.requiresGrad
        ? unbroadcastOwned(gradOutput.negate(), b.shape)
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
        ? unbroadcastOwned(gradOutput * b, a.shape)
        : null;
    final gradB = b.requiresGrad
        ? unbroadcastOwned(gradOutput * a, b.shape)
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
        ? unbroadcastOwned(gradOutput / b, a.shape)
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
      gradB = unbroadcastOwned(quotient, b.shape);
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
      gradA = unbroadcastOwned(rawGradA, a.shape);
    }
    if (b.requiresGrad) {
      final forwardPower = a.pow(b);
      final logA = a.log();
      final gradTimesPower = gradOutput * forwardPower;
      forwardPower.dispose();
      final rawGradB = gradTimesPower * logA;
      gradTimesPower.dispose();
      logA.dispose();
      gradB = unbroadcastOwned(rawGradB, b.shape);
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

/// Backward node for elementwise sine: $y = \sin(x)$.
final class SinBackward extends GradFn {
  /// Forward input tensor.
  final GpuArray<DTypeTag> input;

  /// Creates a [SinBackward] node for [input].
  const SinBackward(this.input);

  @override
  String get name => 'SinBackward';

  @override
  List<GpuArray<DTypeTag>> get inputs => [input];

  @override
  List<GpuArray<DTypeTag>?> backward(GpuArray<DTypeTag> gradOutput) {
    if (!input.requiresGrad) return [null];
    final cosInput = input.cos();
    final gradInput = gradOutput * cosInput;
    cosInput.dispose();
    return [gradInput];
  }
}

/// Backward node for elementwise cosine: $y = \cos(x)$.
final class CosBackward extends GradFn {
  /// Forward input tensor.
  final GpuArray<DTypeTag> input;

  /// Creates a [CosBackward] node for [input].
  const CosBackward(this.input);

  @override
  String get name => 'CosBackward';

  @override
  List<GpuArray<DTypeTag>> get inputs => [input];

  @override
  List<GpuArray<DTypeTag>?> backward(GpuArray<DTypeTag> gradOutput) {
    if (!input.requiresGrad) return [null];
    final sinInput = input.sin();
    final negSin = sinInput.negate();
    sinInput.dispose();
    final gradInput = gradOutput * negSin;
    negSin.dispose();
    return [gradInput];
  }
}

/// Backward node for elementwise absolute value: $y = |x|$.
final class AbsBackward extends GradFn {
  /// Forward input tensor.
  final GpuArray<DTypeTag> input;

  /// Creates an [AbsBackward] node for [input].
  const AbsBackward(this.input);

  @override
  String get name => 'AbsBackward';

  @override
  List<GpuArray<DTypeTag>> get inputs => [input];

  @override
  List<GpuArray<DTypeTag>?> backward(GpuArray<DTypeTag> gradOutput) {
    if (!input.requiresGrad) return [null];
    final positiveBool = input.greater(0.0);
    final negativeBool = input.less(0.0);
    final positiveMask = positiveBool.astype(input.dtype);
    final negativeMask = negativeBool.astype(input.dtype);
    positiveBool.dispose();
    negativeBool.dispose();
    final signTensor = positiveMask - negativeMask;
    positiveMask.dispose();
    negativeMask.dispose();
    final gradInput = gradOutput * signTensor;
    signTensor.dispose();
    return [gradInput];
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
      gradA = unbroadcastOwned(rawGradA, a.shape);
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
      gradB = unbroadcastOwned(rawGradB, b.shape);
    }

    return [gradA, gradB];
  }
}
