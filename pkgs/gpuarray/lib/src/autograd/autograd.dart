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
import '../gpu_array.dart';
import '../backend/kernels.dart';
import '../slice.dart';

/// Global flag controlling whether gradient tracking is enabled.
bool _gradEnabled = true;

/// Returns true if autograd recording is currently enabled.
bool get isGradEnabled => _gradEnabled;

/// Disables gradient calculation during execution of [body].
R no_grad<R>(R Function() body) {
  final prev = _gradEnabled;
  _gradEnabled = false;
  try {
    return body();
  } finally {
    _gradEnabled = prev;
  }
}

/// Abstract base class for backward computation graph nodes.
abstract class GradFn {
  /// Name of the operation.
  String get name;

  /// Input nodes/tensors connected to this operation.
  List<GpuArray> get inputs;

  /// Computes the vector-Jacobian product (VJP) given upstream [gradOutput].
  List<GpuArray?> backward(GpuArray gradOutput);
}

/// Utility for reducing gradients across broadcasted dimensions.
GpuArray _unbroadcast(GpuArray grad, List<int> targetShape) {
  if (grad.shape == targetShape) return grad;

  var res = grad;
  final rankDiff = res.rank - targetShape.length;

  for (var i = 0; i < rankDiff; i++) {
    res = res.sum(axis: 0);
  }

  for (var i = 0; i < targetShape.length; i++) {
    if (targetShape[i] == 1 && res.shape[i] > 1) {
      res = res.sum(axis: i, keepDims: true);
    }
  }

  return res;
}

/// Backward node for Addition: $y = a + b$.
class AddBackward extends GradFn {
  final GpuArray a;
  final GpuArray b;

  AddBackward(this.a, this.b);

  @override
  String get name => 'AddBackward';

  @override
  List<GpuArray> get inputs => [a, b];

  @override
  List<GpuArray?> backward(GpuArray gradOutput) {
    final gradA = a.requiresGrad ? _unbroadcast(gradOutput, a.shape) : null;
    final gradB = b.requiresGrad ? _unbroadcast(gradOutput, b.shape) : null;
    return [gradA, gradB];
  }
}

/// Backward node for Subtraction: $y = a - b$.
class SubBackward extends GradFn {
  final GpuArray a;
  final GpuArray b;

  SubBackward(this.a, this.b);

  @override
  String get name => 'SubBackward';

  @override
  List<GpuArray> get inputs => [a, b];

  @override
  List<GpuArray?> backward(GpuArray gradOutput) {
    final gradA = a.requiresGrad ? _unbroadcast(gradOutput, a.shape) : null;
    final gradB = b.requiresGrad
        ? _unbroadcast(gradOutput.negate(), b.shape)
        : null;
    return [gradA, gradB];
  }
}

/// Backward node for Multiplication: $y = a \cdot b$.
class MulBackward extends GradFn {
  final GpuArray a;
  final GpuArray b;

  MulBackward(this.a, this.b);

  @override
  String get name => 'MulBackward';

  @override
  List<GpuArray> get inputs => [a, b];

  @override
  List<GpuArray?> backward(GpuArray gradOutput) {
    final gradA = a.requiresGrad ? _unbroadcast(gradOutput * b, a.shape) : null;
    final gradB = b.requiresGrad ? _unbroadcast(gradOutput * a, b.shape) : null;
    return [gradA, gradB];
  }
}

/// Backward node for Division: $y = a / b$.
class DivBackward extends GradFn {
  final GpuArray a;
  final GpuArray b;

  DivBackward(this.a, this.b);

  @override
  String get name => 'DivBackward';

  @override
  List<GpuArray> get inputs => [a, b];

  @override
  List<GpuArray?> backward(GpuArray gradOutput) {
    final gradA = a.requiresGrad ? _unbroadcast(gradOutput / b, a.shape) : null;
    final gradB = b.requiresGrad
        ? _unbroadcast((gradOutput * a.negate()) / (b * b), b.shape)
        : null;
    return [gradA, gradB];
  }
}

/// Backward node for Power: $y = a^b$.
class PowBackward extends GradFn {
  final GpuArray a;
  final GpuArray b;

  PowBackward(this.a, this.b);

  @override
  String get name => 'PowBackward';

  @override
  List<GpuArray> get inputs => [a, b];

  @override
  List<GpuArray?> backward(GpuArray gradOutput) {
    GpuArray? gradA;
    GpuArray? gradB;

    if (a.requiresGrad) {
      final bMinus1 = b - 1.0;
      final aPow = a.pow(bMinus1);
      final da = gradOutput * b * aPow;
      gradA = _unbroadcast(da, a.shape);
    }
    if (b.requiresGrad) {
      final y = a.pow(b);
      final logA = a.log();
      final db = gradOutput * y * logA;
      gradB = _unbroadcast(db, b.shape);
    }

    return [gradA, gradB];
  }
}

/// Backward node for Negation: $y = -x$.
class NegBackward extends GradFn {
  final GpuArray input;

  NegBackward(this.input);

  @override
  String get name => 'NegBackward';

  @override
  List<GpuArray> get inputs => [input];

  @override
  List<GpuArray?> backward(GpuArray gradOutput) {
    if (!input.requiresGrad) return [null];
    return [gradOutput.negate()];
  }
}

/// Backward node for Square Root: $y = \sqrt{x}$.
class SqrtBackward extends GradFn {
  final GpuArray input;
  final GpuArray y; // cached output

  SqrtBackward(this.input, this.y);

  @override
  String get name => 'SqrtBackward';

  @override
  List<GpuArray> get inputs => [input];

  @override
  List<GpuArray?> backward(GpuArray gradOutput) {
    if (!input.requiresGrad) return [null];
    return [gradOutput / (y * 2.0)];
  }
}

/// Backward node for Exponential: $y = e^x$.
class ExpBackward extends GradFn {
  final GpuArray input;
  final GpuArray y; // cached output

  ExpBackward(this.input, this.y);

  @override
  String get name => 'ExpBackward';

  @override
  List<GpuArray> get inputs => [input];

  @override
  List<GpuArray?> backward(GpuArray gradOutput) {
    if (!input.requiresGrad) return [null];
    return [gradOutput * y];
  }
}

/// Backward node for Natural Logarithm: $y = \ln(x)$.
class LogBackward extends GradFn {
  final GpuArray input;

  LogBackward(this.input);

  @override
  String get name => 'LogBackward';

  @override
  List<GpuArray> get inputs => [input];

  @override
  List<GpuArray?> backward(GpuArray gradOutput) {
    if (!input.requiresGrad) return [null];
    return [gradOutput / input];
  }
}

/// Backward node for Matrix Multiplication: $Y = A B$.
class MatmulBackward extends GradFn {
  final GpuArray a;
  final GpuArray b;

  MatmulBackward(this.a, this.b);

  @override
  String get name => 'MatmulBackward';

  @override
  List<GpuArray> get inputs => [a, b];

  @override
  List<GpuArray?> backward(GpuArray gradOutput) {
    GpuArray? gradA;
    GpuArray? gradB;

    if (a.requiresGrad) {
      if (a.rank == 1 && b.rank == 1) {
        gradA = b * gradOutput;
      } else if (a.rank == 1) {
        final bT = b.swapaxes(-1, -2);
        gradA = gradOutput.matmul(bT);
      } else if (b.rank == 1) {
        final g = gradOutput.unsqueeze(-1);
        final bExp = b.unsqueeze(0);
        gradA = g.matmul(bExp);
      } else {
        final bT = b.swapaxes(-1, -2);
        gradA = gradOutput.matmul(bT);
      }
      gradA = _unbroadcast(gradA, a.shape);
    }

    if (b.requiresGrad) {
      if (a.rank == 1 && b.rank == 1) {
        gradB = a * gradOutput;
      } else if (a.rank == 1) {
        final aExp = a.unsqueeze(-1);
        final g = gradOutput.unsqueeze(0);
        gradB = aExp.matmul(g);
      } else if (b.rank == 1) {
        final aT = a.swapaxes(-1, -2);
        gradB = aT.matmul(gradOutput);
      } else {
        final aT = a.swapaxes(-1, -2);
        gradB = aT.matmul(gradOutput);
      }
      gradB = _unbroadcast(gradB, b.shape);
    }

    return [gradA, gradB];
  }
}

/// Backward node for Sum Reduction: $y = \sum a$.
class SumBackward extends GradFn {
  final GpuArray a;
  final int? axis;
  final bool keepDims;

  SumBackward(this.a, {this.axis, this.keepDims = false});

  @override
  String get name => 'SumBackward';

  @override
  List<GpuArray> get inputs => [a];

  @override
  List<GpuArray?> backward(GpuArray gradOutput) {
    if (!a.requiresGrad) return [null];

    var grad = gradOutput;
    if (axis != null && !keepDims) {
      final normAxis = axis! < 0 ? axis! + a.rank : axis!;
      grad = grad.unsqueeze(normAxis);
    }

    final ones = GpuArray.ones(a.shape, a.dtype, device: a.device);
    return [grad * ones];
  }
}

/// Backward node for Mean Reduction: $y = \text{mean}(a)$.
class MeanBackward extends GradFn {
  final GpuArray a;
  final int? axis;
  final bool keepDims;

  MeanBackward(this.a, {this.axis, this.keepDims = false});

  @override
  String get name => 'MeanBackward';

  @override
  List<GpuArray> get inputs => [a];

  @override
  List<GpuArray?> backward(GpuArray gradOutput) {
    if (!a.requiresGrad) return [null];

    final n = (axis == null)
        ? a.size
        : a.shape[axis! < 0 ? axis! + a.rank : axis!];
    final scale = 1.0 / n;

    var grad = gradOutput * scale;
    if (axis != null && !keepDims) {
      final normAxis = axis! < 0 ? axis! + a.rank : axis!;
      grad = grad.unsqueeze(normAxis);
    }

    final ones = GpuArray.ones(a.shape, a.dtype, device: a.device);
    return [grad * ones];
  }
}

/// Backward node for ReLU: $y = \max(0, x)$.
class ReluBackward extends GradFn {
  final GpuArray x;

  ReluBackward(this.x);

  @override
  String get name => 'ReluBackward';

  @override
  List<GpuArray> get inputs => [x];

  @override
  List<GpuArray?> backward(GpuArray gradOutput) {
    if (!x.requiresGrad) return [null];
    final mask = x.greater(0.0).astype(x.dtype);
    return [gradOutput * mask];
  }
}

/// Backward node for Sigmoid: $y = \sigma(x) = \frac{1}{1 + e^{-x}}$.
class SigmoidBackward extends GradFn {
  final GpuArray input;
  final GpuArray y; // cached forward output

  SigmoidBackward(this.input, this.y);

  @override
  String get name => 'SigmoidBackward';

  @override
  List<GpuArray> get inputs => [input];

  @override
  List<GpuArray?> backward(GpuArray gradOutput) {
    if (!input.requiresGrad) return [null];
    // dy/dx = y * (1 - y)
    final oneMinusY = y.negate() + 1.0;
    return [gradOutput * y * oneMinusY];
  }
}

/// Backward node for Tanh: $y = \tanh(x)$.
class TanhBackward extends GradFn {
  final GpuArray input;
  final GpuArray y; // cached forward output

  TanhBackward(this.input, this.y);

  @override
  String get name => 'TanhBackward';

  @override
  List<GpuArray> get inputs => [input];

  @override
  List<GpuArray?> backward(GpuArray gradOutput) {
    if (!input.requiresGrad) return [null];
    // dy/dx = 1 - y^2
    final oneMinusYSq = (y * y).negate() + 1.0;
    return [gradOutput * oneMinusYSq];
  }
}

/// Backward node for Softmax: $y = \text{softmax}(x, \text{axis})$.
class SoftmaxBackward extends GradFn {
  final GpuArray input;
  final GpuArray y; // cached forward output
  final int axis;

  SoftmaxBackward(this.input, this.y, {this.axis = -1});

  @override
  String get name => 'SoftmaxBackward';

  @override
  List<GpuArray> get inputs => [input];

  @override
  List<GpuArray?> backward(GpuArray gradOutput) {
    if (!input.requiresGrad) return [null];
    // dL/dx = y * (gradOutput - sum(gradOutput * y, axis, keepDims: true))
    final gy = gradOutput * y;
    final sumGy = gy.sum(axis: axis, keepDims: true);
    final gradInput = y * (gradOutput - sumGy);
    return [gradInput];
  }
}

/// Backward node for Log-Softmax: $y = \text{log\_softmax}(x, \text{axis})$.
class LogSoftmaxBackward extends GradFn {
  final GpuArray input;
  final GpuArray y; // cached forward output
  final int axis;

  LogSoftmaxBackward(this.input, this.y, {this.axis = -1});

  @override
  String get name => 'LogSoftmaxBackward';

  @override
  List<GpuArray> get inputs => [input];

  @override
  List<GpuArray?> backward(GpuArray gradOutput) {
    if (!input.requiresGrad) return [null];
    // dL/dx = gradOutput - exp(y) * sum(gradOutput, axis, keepDims: true)
    final sumG = gradOutput.sum(axis: axis, keepDims: true);
    final expY = y.exp();
    final gradInput = gradOutput - (expY * sumG);
    return [gradInput];
  }
}

/// Backward node for Transpose / Permute.
class TransposeBackward extends GradFn {
  final GpuArray input;
  final List<int> axes;

  TransposeBackward(this.input, this.axes);

  @override
  String get name => 'TransposeBackward';

  @override
  List<GpuArray> get inputs => [input];

  @override
  List<GpuArray?> backward(GpuArray gradOutput) {
    if (!input.requiresGrad) return [null];
    final invAxes = List<int>.filled(axes.length, 0);
    for (var i = 0; i < axes.length; i++) {
      invAxes[axes[i]] = i;
    }
    return [gradOutput.transpose(invAxes)];
  }
}

/// Backward node for Reshape.
class ReshapeBackward extends GradFn {
  final GpuArray input;
  final List<int> originalShape;

  ReshapeBackward(this.input, this.originalShape);

  @override
  String get name => 'ReshapeBackward';

  @override
  List<GpuArray> get inputs => [input];

  @override
  List<GpuArray?> backward(GpuArray gradOutput) {
    if (!input.requiresGrad) return [null];
    return [gradOutput.reshape(originalShape)];
  }
}

/// Backward node for Squeeze.
class SqueezeBackward extends GradFn {
  final GpuArray input;
  final List<int> originalShape;

  SqueezeBackward(this.input, this.originalShape);

  @override
  String get name => 'SqueezeBackward';

  @override
  List<GpuArray> get inputs => [input];

  @override
  List<GpuArray?> backward(GpuArray gradOutput) {
    if (!input.requiresGrad) return [null];
    return [gradOutput.reshape(originalShape)];
  }
}

/// Backward node for Unsqueeze.
class UnsqueezeBackward extends GradFn {
  final GpuArray input;
  final List<int> originalShape;

  UnsqueezeBackward(this.input, this.originalShape);

  @override
  String get name => 'UnsqueezeBackward';

  @override
  List<GpuArray> get inputs => [input];

  @override
  List<GpuArray?> backward(GpuArray gradOutput) {
    if (!input.requiresGrad) return [null];
    return [gradOutput.reshape(originalShape)];
  }
}

/// Backward node for Subview Slicing.
class SliceBackward extends GradFn {
  final GpuArray input;
  final List<dynamic> specs;

  SliceBackward(this.input, this.specs);

  @override
  String get name => 'SliceBackward';

  @override
  List<GpuArray> get inputs => [input];

  @override
  List<GpuArray?> backward(GpuArray gradOutput) {
    if (!input.requiresGrad) return [null];
    final grad = GpuArray.zeros(input.shape, input.dtype, device: input.device);
    final sliceView = grad.slice(specs);
    GpuKernels.copyStrided(
      src: gradOutput.buffer,
      shape: sliceView.shape,
      strides: gradOutput.strides,
      offsetSrc: gradOutput.offsetElements,
      dtypeSrc: gradOutput.dtype,
      dst: sliceView.buffer,
      outStrides: sliceView.strides,
      offsetDst: sliceView.offsetElements,
      dtypeDst: sliceView.dtype,
    );
    return [grad];
  }
}

/// Backward node for Concatenate: $y = \text{concat}([a, b, \dots], \text{axis})$.
class ConcatenateBackward extends GradFn {
  @override
  final List<GpuArray> inputs;
  final int axis;

  ConcatenateBackward(this.inputs, {required this.axis});

  @override
  String get name => 'ConcatenateBackward';

  @override
  List<GpuArray?> backward(GpuArray gradOutput) {
    final grads = <GpuArray?>[];
    var offset = 0;
    final rank = gradOutput.rank;
    final normAxis = axis < 0 ? axis + rank : axis;

    for (final inp in inputs) {
      final inpLen = inp.shape[normAxis];
      if (inp.requiresGrad) {
        final sliceSpecs = List<dynamic>.generate(rank, (d) {
          if (d == normAxis) {
            return Slice(offset, offset + inpLen);
          }
          return const All();
        });
        final sliceView = gradOutput.slice(sliceSpecs);
        grads.add(sliceView.copy());
      } else {
        grads.add(null);
      }
      offset += inpLen;
    }
    return grads;
  }
}

/// Backward node for Embedding lookup.
class EmbeddingBackward extends GradFn {
  final GpuArray weight;
  final GpuArray indices;
  final int numEmbeddings;
  final int embeddingDim;

  EmbeddingBackward(
    this.weight,
    this.indices,
    this.numEmbeddings,
    this.embeddingDim,
  );

  @override
  String get name => 'EmbeddingBackward';

  @override
  List<GpuArray> get inputs => [weight];

  @override
  List<GpuArray?> backward(GpuArray gradOutput) {
    if (!weight.requiresGrad) return [null];
    final gradW = GpuArray.zeros(
      weight.shape,
      weight.dtype,
      device: weight.device,
    );
    final idxList = indices.toList().cast<int>();
    final gradND = gradOutput.toNDArray();
    final gradList = gradND
        .toList()
        .cast<num>()
        .map((e) => e.toDouble())
        .toList();
    gradND.dispose();

    final wND = gradW.toNDArray();
    final wList = wND.toList().cast<num>().map((e) => e.toDouble()).toList();
    wND.dispose();

    for (var i = 0; i < idxList.length; i++) {
      final idx = idxList[i];
      if (idx < 0 || idx >= numEmbeddings) continue;
      for (var d = 0; d < embeddingDim; d++) {
        wList[idx * embeddingDim + d] += gradList[i * embeddingDim + d];
      }
    }

    final updated = GpuArray.fromList(
      wList,
      weight.shape,
      weight.dtype,
      device: weight.device,
    );
    updated.buffer.copyToBuffer(gradW.buffer, gradW.byteSize);
    updated.dispose();
    return [gradW];
  }
}

/// Backward node for Cross Entropy Loss.
class CrossEntropyBackward extends GradFn {
  final GpuArray logits;
  final GpuArray targets;
  final GpuArray probs;
  final String reduction;

  CrossEntropyBackward(
    this.logits,
    this.targets,
    this.probs, {
    this.reduction = 'mean',
  });

  @override
  String get name => 'CrossEntropyBackward';

  @override
  List<GpuArray> get inputs => [logits];

  @override
  List<GpuArray?> backward(GpuArray gradOutput) {
    if (!logits.requiresGrad) return [null];
    final numSamples = logits.shape[0];
    final numClasses = logits.shape[logits.rank - 1];
    final targetList = targets.toList().cast<int>();

    final gradND = probs.toNDArray();
    final gradList = gradND
        .toList()
        .cast<num>()
        .map((e) => e.toDouble())
        .toList();
    gradND.dispose();

    for (var i = 0; i < numSamples; i++) {
      final t = targetList[i];
      if (t >= 0 && t < numClasses) {
        gradList[i * numClasses + t] -= 1.0;
      }
    }

    final grad = GpuArray.fromList(
      gradList,
      logits.shape,
      logits.dtype,
      device: logits.device,
    );

    if (reduction == 'mean') {
      final scale = 1.0 / numSamples;
      final gVal = (gradOutput.rank == 0 || gradOutput.size == 1)
          ? (gradOutput.item() as num).toDouble()
          : 1.0;
      final scaled = grad * (scale * gVal);
      grad.dispose();
      return [scaled];
    } else if (reduction == 'sum') {
      final gVal = (gradOutput.rank == 0 || gradOutput.size == 1)
          ? (gradOutput.item() as num).toDouble()
          : 1.0;
      final scaled = grad * gVal;
      grad.dispose();
      return [scaled];
    } else {
      final gUnsq = gradOutput.unsqueeze(-1);
      final scaled = grad * gUnsq;
      grad.dispose();
      return [scaled];
    }
  }
}

/// Backward node for 2D Convolution.
class Conv2dBackward extends GradFn {
  final GpuArray input;
  final GpuArray weight;
  final GpuArray? bias;
  final int stride;
  final int padding;
  final int kernelSize;

  Conv2dBackward({
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
  List<GpuArray> get inputs => [input, weight, ?bias];

  @override
  List<GpuArray?> backward(GpuArray gradOutput) {
    final batchSize = input.shape[0];
    final inC = input.shape[1];
    final inH = input.shape[2];
    final inW = input.shape[3];
    final outC = weight.shape[0];
    final outH = gradOutput.shape[2];
    final outW = gradOutput.shape[3];

    final gradOutND = gradOutput.toNDArray();
    final gradOutList = gradOutND
        .toList()
        .cast<num>()
        .map((e) => e.toDouble())
        .toList();
    gradOutND.dispose();

    GpuArray? gradInput;
    GpuArray? gradWeight;
    GpuArray? gradBias;

    if (input.requiresGrad) {
      final weightND = weight.toNDArray();
      final weightList = weightND
          .toList()
          .cast<num>()
          .map((e) => e.toDouble())
          .toList();
      weightND.dispose();

      final inGradList = List<double>.filled(input.size, 0.0);
      for (var b = 0; b < batchSize; b++) {
        for (var oc = 0; oc < outC; oc++) {
          for (var oh = 0; oh < outH; oh++) {
            for (var ow = 0; ow < outW; ow++) {
              final gOut =
                  gradOutList[((b * outC + oc) * outH + oh) * outW + ow];
              final ihStart = oh * stride - padding;
              final iwStart = ow * stride - padding;

              for (var ic = 0; ic < inC; ic++) {
                for (var kh = 0; kh < kernelSize; kh++) {
                  final ih = ihStart + kh;
                  if (ih < 0 || ih >= inH) continue;
                  for (var kw = 0; kw < kernelSize; kw++) {
                    final iw = iwStart + kw;
                    if (iw < 0 || iw >= inW) continue;

                    final inIdx = ((b * inC + ic) * inH + ih) * inW + iw;
                    final wIdx =
                        ((oc * inC + ic) * kernelSize + kh) * kernelSize + kw;
                    inGradList[inIdx] += gOut * weightList[wIdx];
                  }
                }
              }
            }
          }
        }
      }
      gradInput = GpuArray.fromList(
        inGradList,
        input.shape,
        input.dtype,
        device: input.device,
      );
    }

    if (weight.requiresGrad) {
      final inputND = input.toNDArray();
      final inList = inputND
          .toList()
          .cast<num>()
          .map((e) => e.toDouble())
          .toList();
      inputND.dispose();

      final wGradList = List<double>.filled(weight.size, 0.0);
      for (var b = 0; b < batchSize; b++) {
        for (var oc = 0; oc < outC; oc++) {
          for (var oh = 0; oh < outH; oh++) {
            for (var ow = 0; ow < outW; ow++) {
              final gOut =
                  gradOutList[((b * outC + oc) * outH + oh) * outW + ow];
              final ihStart = oh * stride - padding;
              final iwStart = ow * stride - padding;

              for (var ic = 0; ic < inC; ic++) {
                for (var kh = 0; kh < kernelSize; kh++) {
                  final ih = ihStart + kh;
                  if (ih < 0 || ih >= inH) continue;
                  for (var kw = 0; kw < kernelSize; kw++) {
                    final iw = iwStart + kw;
                    if (iw < 0 || iw >= inW) continue;

                    final inIdx = ((b * inC + ic) * inH + ih) * inW + iw;
                    final wIdx =
                        ((oc * inC + ic) * kernelSize + kh) * kernelSize + kw;
                    wGradList[wIdx] += gOut * inList[inIdx];
                  }
                }
              }
            }
          }
        }
      }
      gradWeight = GpuArray.fromList(
        wGradList,
        weight.shape,
        weight.dtype,
        device: weight.device,
      );
    }

    if (bias != null && bias!.requiresGrad) {
      final bGradList = List<double>.filled(outC, 0.0);
      for (var b = 0; b < batchSize; b++) {
        for (var oc = 0; oc < outC; oc++) {
          for (var oh = 0; oh < outH; oh++) {
            for (var ow = 0; ow < outW; ow++) {
              final gOut =
                  gradOutList[((b * outC + oc) * outH + oh) * outW + ow];
              bGradList[oc] += gOut;
            }
          }
        }
      }
      gradBias = GpuArray.fromList(
        bGradList,
        bias!.shape,
        bias!.dtype,
        device: bias!.device,
      );
    }

    return [gradInput, gradWeight, ?gradBias];
  }
}

/// Executes reverse-mode automatic differentiation starting from [root].
void runBackward(
  GpuArray root, [
  GpuArray? gradient,
  bool retainGraph = false,
]) {
  if (!root.requiresGrad) {
    throw StateError(
      'Cannot call backward() on a tensor with requiresGrad = false.',
    );
  }

  no_grad(() {
    final seedGrad =
        gradient ??
        ((root.rank == 0 || root.size == 1)
            ? GpuArray.ones(root.shape, root.dtype, device: root.device)
            : throw ArgumentError(
                'grad can be implicitly created only for scalar outputs',
              ));

    root.grad = (root.grad == null) ? seedGrad : (root.grad! + seedGrad);

    // Topological sort of computational DAG
    final orderedNodes = <GpuArray>[];
    final visited = <GpuArray>{};

    void buildTopo(GpuArray node) {
      if (visited.contains(node)) return;
      visited.add(node);

      final fn = node.gradFn;
      if (fn != null) {
        for (final input in fn.inputs) {
          if (input.requiresGrad) {
            buildTopo(input);
          }
        }
      }
      orderedNodes.add(node);
    }

    buildTopo(root);

    // Traverse in reverse topological order
    for (var i = orderedNodes.length - 1; i >= 0; i--) {
      final node = orderedNodes[i];
      final fn = node.gradFn;
      final nodeGrad = node.grad;

      if (fn != null && nodeGrad != null) {
        final inputGrads = fn.backward(nodeGrad);
        final inputs = fn.inputs;

        for (var j = 0; j < inputs.length; j++) {
          final inp = inputs[j];
          final g = (j < inputGrads.length) ? inputGrads[j] : null;

          if (inp.requiresGrad && g != null) {
            inp.grad = (inp.grad == null) ? g : (inp.grad! + g);
          }
        }

        if (!retainGraph) {
          node.gradFn = null;
        }
      }
    }
  });
}
