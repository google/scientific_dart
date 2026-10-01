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
import 'autograd_wgsl.dart';

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

/// Tracks non-leaf tensors whose computation graphs have already been released.
final Expando<bool> _releasedGraphNodes = Expando<bool>('releasedGraphNodes');

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
GpuArray<DTypeTag> unbroadcast(
  GpuArray<DTypeTag> grad,
  List<int> targetShape, {
  bool copyIfUnchanged = true,
}) {
  if (areShapesIdentical(grad.shape, targetShape)) {
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
GpuArray<DTypeTag> unbroadcastOwned(
  GpuArray<DTypeTag> intermediate,
  List<int> targetShape,
) {
  final reduced = unbroadcast(
    intermediate,
    targetShape,
    copyIfUnchanged: false,
  );
  if (!identical(reduced, intermediate)) {
    intermediate.dispose();
  }
  return reduced;
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
  if (_releasedGraphNodes[root] == true) {
    throw StateError(
      'Cannot call backward() a second time after the computation graph has been released. '
      'Pass retainGraph: true to backward() to run multiple backward passes.',
    );
  }

  noGrad(() {
    final bool createdSeed;
    final GpuArray<DTypeTag> seedGrad;
    if (gradient != null) {
      if (!areShapesIdentical(gradient.shape, root.shape)) {
        throw ArgumentError.value(
          gradient.shape,
          'gradient',
          'Must match root tensor shape ${root.shape}.',
        );
      }
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

    if (root.isLeaf) {
      if (root.grad == null) {
        root.grad = seedGrad.copy();
      } else {
        final previousGrad = root.grad!;
        root.grad = previousGrad + seedGrad;
        previousGrad.dispose();
      }
    }

    final nodeGrads = <GpuArray<DTypeTag>, GpuArray<DTypeTag>>{root: seedGrad};
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
      final nodeGrad = nodeGrads[node];

      if (functionNode != null && nodeGrad != null) {
        final inputGrads = functionNode.backward(nodeGrad);
        final inputs = functionNode.inputs;

        for (var j = 0; j < inputs.length; j++) {
          final inputTensor = inputs[j];
          final incomingGrad = (j < inputGrads.length) ? inputGrads[j] : null;

          if (incomingGrad != null) {
            if (inputTensor.requiresGrad) {
              if (inputTensor.isLeaf) {
                if (inputTensor.grad == null) {
                  inputTensor.grad = identical(incomingGrad, nodeGrad)
                      ? incomingGrad.copy()
                      : incomingGrad;
                } else {
                  final previousGrad = inputTensor.grad!;
                  inputTensor.grad = previousGrad + incomingGrad;
                  previousGrad.dispose();
                  if (!identical(incomingGrad, nodeGrad)) {
                    incomingGrad.dispose();
                  }
                }
              } else {
                final existingGrad = nodeGrads[inputTensor];
                if (existingGrad == null) {
                  nodeGrads[inputTensor] = identical(incomingGrad, nodeGrad)
                      ? incomingGrad.copy()
                      : incomingGrad;
                } else {
                  nodeGrads[inputTensor] = existingGrad + incomingGrad;
                  existingGrad.dispose();
                  if (!identical(incomingGrad, nodeGrad)) {
                    incomingGrad.dispose();
                  }
                }
              }
            } else if (!identical(incomingGrad, nodeGrad)) {
              incomingGrad.dispose();
            }
          }
        }

        if (!retainGraph) {
          _releasedGraphNodes[node] = true;
          node.gradFn = null;
        }
      }

      if (nodeGrad != null) {
        if (!identical(node, root) || createdSeed) {
          nodeGrad.dispose();
        }
        nodeGrads.remove(node);
      }
    }
  });
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
