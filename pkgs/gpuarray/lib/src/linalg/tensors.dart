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

import '../dtype.dart';
import '../gpu_array.dart';
import 'linalg.dart';
import 'linalg_buffer_ops.dart';
import 'linalg_wgsl_df64.dart';
import 'tensor_kernels.dart';

void _checkPair(GpuArray a, GpuArray b, String functionName) {
  if (a.isDisposed) {
    throw StateError(
      'Cannot execute $functionName on a disposed GpuArray (a).',
    );
  }
  if (b.isDisposed) {
    throw StateError(
      'Cannot execute $functionName on a disposed GpuArray (b).',
    );
  }
  if (a.device != b.device) {
    throw ArgumentError.value(
      b,
      'b',
      'Must reside on the same GpuDevice as a.',
    );
  }
  if (a.dtype != b.dtype) {
    throw ArgumentError.value(
      b,
      'b',
      'Must have the same DType as a (${a.dtype}), got ${b.dtype}.',
    );
  }
}

int _normalizeAxis(int axis, int ndim, String paramName) {
  final normalized = axis < 0 ? ndim + axis : axis;
  if (normalized < 0 || normalized >= ndim) {
    throw ArgumentError.value(
      axis,
      paramName,
      'Must be in range [-$ndim, ${ndim - 1}].',
    );
  }
  return normalized;
}

List<int> _contiguousElementStrides(List<int> shape) {
  final strides = List<int>.filled(shape.length, 1);
  var running = 1;
  for (var i = shape.length - 1; i >= 0; i--) {
    strides[i] = running;
    running *= shape[i];
  }
  return strides;
}

bool _isIdentityPermutation(List<int> perm) {
  for (var i = 0; i < perm.length; i++) {
    if (perm[i] != i) return false;
  }
  return true;
}

/// Evaluates the Einstein summation convention [subscripts] on [operands].
///
/// Supports both explicit (`'ij,jk->ik'`) and implicit (`'ij,jk'`, `'ii'`)
/// Einstein summation notation across one or more [GpuArray] tensors.
///
/// The [operands] list must be non-empty, reside on the same [GpuDevice], and
/// share a common [DType]. If [out] is provided, it must match the output
/// shape, dtype, and device.
GpuArray<T> einsum<T extends DTypeTag>(
  String subscripts,
  List<GpuArray<T>> operands, {
  GpuArray<T>? out,
}) {
  if (operands.isEmpty) {
    throw ArgumentError.value(
      operands,
      'operands',
      'Must provide at least one operand for einsum.',
    );
  }
  final first = operands.first;
  for (var i = 0; i < operands.length; i++) {
    if (operands[i].isDisposed) {
      throw StateError('Cannot execute einsum on a disposed GpuArray.');
    }
    if (operands[i].device != first.device) {
      throw ArgumentError.value(
        operands[i],
        'operands[$i]',
        'Must reside on the same GpuDevice as operands[0].',
      );
    }
    if (operands[i].dtype != first.dtype) {
      throw ArgumentError.value(
        operands[i],
        'operands[$i]',
        'Must have the same DType as operands[0].',
      );
    }
  }

  final cleaned = subscripts.replaceAll(' ', '');
  final parts = cleaned.split('->');
  if (parts.length > 2) {
    throw ArgumentError.value(
      subscripts,
      'subscripts',
      'Must contain at most one "->" separator.',
    );
  }
  final inputSpecs = parts[0].split(',');
  if (inputSpecs.length != operands.length) {
    throw ArgumentError.value(
      subscripts,
      'subscripts',
      'Must specify ${operands.length} operand subscripts, got ${inputSpecs.length}.',
    );
  }

  final labelDims = <String, int>{};
  final labelCounts = <String, int>{};
  for (var i = 0; i < operands.length; i++) {
    final spec = inputSpecs[i];
    final operand = operands[i];
    if (spec.length != operand.ndim) {
      throw ArgumentError.value(
        subscripts,
        'subscripts',
        'Must match ndim of operand $i (${operand.ndim}), got "$spec".',
      );
    }
    for (var d = 0; d < spec.length; d++) {
      final ch = spec[d];
      final dimSize = operand.shape[d];
      if (labelDims[ch] case final existing?) {
        if (existing != dimSize) {
          throw ArgumentError.value(
            subscripts,
            'subscripts',
            'Must have consistent dimension size for index "$ch" ($existing != $dimSize).',
          );
        }
      } else {
        labelDims[ch] = dimSize;
      }
      labelCounts[ch] = (labelCounts[ch] ?? 0) + 1;
    }
  }

  final List<String> outputLabels;
  if (parts.length == 2) {
    outputLabels = parts[1].split('');
    final seen = <String>{};
    for (final ch in outputLabels) {
      if (!labelDims.containsKey(ch)) {
        throw ArgumentError.value(
          subscripts,
          'subscripts',
          'Must not reference unknown output label "$ch".',
        );
      }
      if (!seen.add(ch)) {
        throw ArgumentError.value(
          subscripts,
          'subscripts',
          'Must not contain duplicate output label "$ch".',
        );
      }
    }
  } else {
    final sorted = labelCounts.keys.toList()..sort();
    outputLabels = <String>[
      for (final ch in sorted)
        if (labelCounts[ch] == 1) ch,
    ];
  }

  // If more than 3 operands, reduce pairwise on GPU inside ResourceScope.
  if (operands.length > 3) {
    return ResourceScope.scope(() {
      final remainingLabels = <String>{
        ...outputLabels,
        for (var i = 2; i < inputSpecs.length; i++) ...inputSpecs[i].split(''),
      };
      final firstTwoLabels = <String>{
        ...inputSpecs[0].split(''),
        ...inputSpecs[1].split(''),
      };
      final midOut = <String>[
        for (final ch in firstTwoLabels)
          if (remainingLabels.contains(ch)) ch,
      ];
      final stepSubscripts =
          '${inputSpecs[0]},${inputSpecs[1]}->${midOut.join()}';
      final midArray = einsum<T>(stepSubscripts, <GpuArray<T>>[
        operands[0],
        operands[1],
      ]);
      final restSpecs = <String>[midOut.join(), ...inputSpecs.sublist(2)];
      final restSubscripts = '${restSpecs.join(',')}->${outputLabels.join()}';
      final finalResult = einsum<T>(restSubscripts, <GpuArray<T>>[
        midArray,
        ...operands.sublist(2),
      ], out: out);
      if (out == null) finalResult.detachToParentScope();
      return finalResult;
    });
  }

  final outSet = outputLabels.toSet();
  final contractedLabels = <String>[
    for (final ch in labelDims.keys)
      if (!outSet.contains(ch)) ch,
  ];
  final allLabels = <String>[...outputLabels, ...contractedLabels];
  final labelToId = <String, int>{
    for (var i = 0; i < allLabels.length; i++) allLabels[i]: i,
  };

  final outShape = <int>[for (final ch in outputLabels) labelDims[ch]!];
  validateLinalgOut(out, first.device, outShape, first.dtype);

  final outSize = outShape.isEmpty ? 1 : outShape.reduce((a, b) => a * b);
  var contractSize = 1;
  for (final ch in contractedLabels) {
    contractSize *= labelDims[ch]!;
  }
  final labelSizes = <int>[for (final ch in allLabels) labelDims[ch]!];
  final operandLabelStrides = <List<int>>[];
  for (var i = 0; i < operands.length; i++) {
    final elemStrides = _contiguousElementStrides(operands[i].shape);
    final stridesForLabels = List<int>.filled(allLabels.length, 0);
    final spec = inputSpecs[i];
    for (var axis = 0; axis < spec.length; axis++) {
      final id = labelToId[spec[axis]]!;
      stridesForLabels[id] += elemStrides[axis];
    }
    operandLabelStrides.add(stridesForLabels);
  }

  return ResourceScope.scope(() {
    if (contractSize == 0 && outSize > 0) {
      final zeroOut = GpuArray<T>.zeros(
        outShape,
        first.dtype,
        device: first.device,
      );
      if (out != null) {
        return copyGpuArray(zeroOut, out: out);
      }
      zeroOut.detachToParentScope();
      return zeroOut;
    }

    final operandBuffers = <dynamic>[
      for (final op in operands) toContiguousFloat64Buffer(op),
    ];
    final outBuffer = dispatchEinsumF64Gpu(
      first.device,
      operandBuffers.cast(),
      outSize: outSize,
      contractSize: contractSize,
      numOutLabels: outputLabels.length,
      numTotalLabels: allLabels.length,
      labelSizes: labelSizes,
      operandLabelStrides: operandLabelStrides,
    );
    final output = writeFloat64BufferToArray<T>(
      first.device,
      outBuffer,
      outShape,
      first.dtype,
      out: out,
    );
    if (out == null) output.detachToParentScope();
    return output;
  });
}

/// Tensor dot product of [a] and [b] along specified [axes].
///
/// The [axes] parameter may be:
/// - A non-negative [int] `N`: sums over the last `N` axes of [a] and the first
///   `N` axes of [b] in order.
/// - A 2-element record `(List<int>, List<int>)` or 2-element [List] specifying
///   the exact axes of [a] and [b] to contract.
///
/// Both [a] and [b] must reside on the same [GpuDevice], have matching
/// [DType]s, and have matching dimension sizes along contracted axes.
GpuArray<T> tensordot<T extends DTypeTag>(
  GpuArray<T> a,
  GpuArray<T> b, {
  Object axes = 2,
  GpuArray<T>? out,
}) {
  _checkPair(a, b, 'tensordot');

  List<int> axesA;
  List<int> axesB;
  if (axes is int) {
    if (axes < 0) {
      throw ArgumentError.value(axes, 'axes', 'Must be non-negative.');
    }
    if (axes > a.ndim || axes > b.ndim) {
      throw ArgumentError.value(
        axes,
        'axes',
        'Must not exceed input dimensions (${a.ndim}, ${b.ndim}).',
      );
    }
    axesA = <int>[for (var i = a.ndim - axes; i < a.ndim; i++) i];
    axesB = <int>[for (var i = 0; i < axes; i++) i];
  } else if (axes is (List<int>, List<int>)) {
    axesA = List<int>.of(axes.$1);
    axesB = List<int>.of(axes.$2);
  } else if (axes is List && axes.length == 2) {
    final firstAxes = axes[0];
    final secondAxes = axes[1];
    if (firstAxes is List<int> && secondAxes is List<int>) {
      axesA = List<int>.of(firstAxes);
      axesB = List<int>.of(secondAxes);
    } else if (firstAxes is int && secondAxes is int) {
      axesA = <int>[firstAxes];
      axesB = <int>[secondAxes];
    } else {
      throw ArgumentError.value(
        axes,
        'axes',
        'Must be an int or a pair of axis lists.',
      );
    }
  } else {
    throw ArgumentError.value(
      axes,
      'axes',
      'Must be an int or a pair of axis lists.',
    );
  }

  if (axesA.length != axesB.length) {
    throw ArgumentError.value(
      axes,
      'axes',
      'Must have the same number of axes for a and b.',
    );
  }

  final normAxesA = <int>[];
  final seenA = <int>{};
  for (final ax in axesA) {
    final norm = _normalizeAxis(ax, a.ndim, 'axes');
    if (!seenA.add(norm)) {
      throw ArgumentError.value(
        axes,
        'axes',
        'Must not contain duplicate axes for a.',
      );
    }
    normAxesA.add(norm);
  }

  final normAxesB = <int>[];
  final seenB = <int>{};
  for (final ax in axesB) {
    final norm = _normalizeAxis(ax, b.ndim, 'axes');
    if (!seenB.add(norm)) {
      throw ArgumentError.value(
        axes,
        'axes',
        'Must not contain duplicate axes for b.',
      );
    }
    normAxesB.add(norm);
  }

  var contractCount = 1;
  for (var i = 0; i < normAxesA.length; i++) {
    final dimA = a.shape[normAxesA[i]];
    final dimB = b.shape[normAxesB[i]];
    if (dimA != dimB) {
      throw ArgumentError.value(
        axes,
        'axes',
        'Must have matching dimension sizes along contracted axes ($dimA != $dimB).',
      );
    }
    contractCount *= dimA;
  }

  final freeA = <int>[
    for (var i = 0; i < a.ndim; i++)
      if (!seenA.contains(i)) i,
  ];
  final freeB = <int>[
    for (var i = 0; i < b.ndim; i++)
      if (!seenB.contains(i)) i,
  ];

  final outShape = <int>[
    for (final i in freeA) a.shape[i],
    for (final i in freeB) b.shape[i],
  ];
  validateLinalgOut(out, a.device, outShape, a.dtype);

  var m = 1;
  for (final i in freeA) {
    m *= a.shape[i];
  }
  var n = 1;
  for (final i in freeB) {
    n *= b.shape[i];
  }

  final permA = <int>[...freeA, ...normAxesA];
  final permB = <int>[...normAxesB, ...freeB];

  return ResourceScope.scope(() {
    final aView = _isIdentityPermutation(permA) ? a : a.transpose(permA);
    final bView = _isIdentityPermutation(permB) ? b : b.transpose(permB);
    if (isComplexDType(a.dtype)) {
      final bufferA = toContiguousComplex128Buffer(aView);
      final bufferB = toContiguousComplex128Buffer(bView);
      final bufferC = dispatchBatchedMatmulC128Gpu(
        a.device,
        bufferA,
        bufferB,
        batchCount: 1,
        m: m,
        k: contractCount,
        n: n,
      );
      final output = writeComplex128BufferToArray<T>(
        a.device,
        bufferC,
        outShape,
        a.dtype,
        out: out,
      );
      if (out == null) output.detachToParentScope();
      return output;
    } else {
      final bufferA = toContiguousFloat64Buffer(aView);
      final bufferB = toContiguousFloat64Buffer(bView);
      final bufferC = dispatchBatchedMatmulF64Gpu(
        a.device,
        bufferA,
        bufferB,
        batchCount: 1,
        m: m,
        k: contractCount,
        n: n,
      );
      final output = writeFloat64BufferToArray<T>(
        a.device,
        bufferC,
        outShape,
        a.dtype,
        out: out,
      );
      if (out == null) output.detachToParentScope();
      return output;
    }
  });
}

/// Kronecker product of two [GpuArray] tensors [a] and [b].
///
/// Produces a block tensor whose shape along each dimension `d` is
/// `a.shape[d] * b.shape[d]` (after prepending `1`s to the shorter shape).
///
/// Both [a] and [b] must reside on the same [GpuDevice] and share a common
/// [DType].
GpuArray<T> kron<T extends DTypeTag>(
  GpuArray<T> a,
  GpuArray<T> b, {
  GpuArray<T>? out,
}) {
  _checkPair(a, b, 'kron');
  final maxRank = math.max(a.ndim, b.ndim);
  final aPadded = <int>[
    for (var i = 0; i < maxRank - a.ndim; i++) 1,
    ...a.shape,
  ];
  final bPadded = <int>[
    for (var i = 0; i < maxRank - b.ndim; i++) 1,
    ...b.shape,
  ];
  final outShape = <int>[
    for (var i = 0; i < maxRank; i++) aPadded[i] * bPadded[i],
  ];
  validateLinalgOut(out, a.device, outShape, a.dtype);

  return ResourceScope.scope(() {
    final bufferA = toContiguousFloat64Buffer(a);
    final bufferB = toContiguousFloat64Buffer(b);
    final bufferC = dispatchKronF64Gpu(
      a.device,
      bufferA,
      bufferB,
      outShape: outShape,
      aShapePadded: aPadded,
      bShapePadded: bPadded,
    );
    final output = writeFloat64BufferToArray<T>(
      a.device,
      bufferC,
      outShape,
      a.dtype,
      out: out,
    );
    if (out == null) output.detachToParentScope();
    return output;
  });
}

/// Inner product of two [GpuArray] tensors [a] and [b].
///
/// Computes the ordinary dot product for 1-D vectors (without complex
/// conjugation), or sums the product over the last axes of N-D tensors [a] and
/// [b].
///
/// Both [a] and [b] must reside on the same [GpuDevice], share a common
/// [DType], and have matching last dimensions (`a.shape.last == b.shape.last`).
GpuArray<T> inner<T extends DTypeTag>(
  GpuArray<T> a,
  GpuArray<T> b, {
  GpuArray<T>? out,
}) {
  _checkPair(a, b, 'inner');
  if (a.ndim == 0 || b.ndim == 0) {
    return dot(a, b, out: out);
  }
  final kA = a.shape.last;
  final kB = b.shape.last;
  if (kA != kB) {
    throw ArgumentError.value(
      b.shape,
      'b',
      'Must have last dimension matching a ($kA != $kB).',
    );
  }

  final outShape = <int>[
    ...a.shape.sublist(0, a.ndim - 1),
    ...b.shape.sublist(0, b.ndim - 1),
  ];
  validateLinalgOut(out, a.device, outShape, a.dtype);

  final m = kA == 0 ? 0 : a.size ~/ kA;
  final n = kB == 0 ? 0 : b.size ~/ kB;
  return ResourceScope.scope(() {
    if (isComplexDType(a.dtype)) {
      final bufferA = toContiguousComplex128Buffer(a);
      final bufferB = toContiguousComplex128Buffer(b);
      final bufferC = dispatchBatchedMatmulC128Gpu(
        a.device,
        bufferA,
        bufferB,
        batchCount: 1,
        m: m,
        k: kA,
        n: n,
        transposeB: true,
      );
      final output = writeComplex128BufferToArray<T>(
        a.device,
        bufferC,
        outShape,
        a.dtype,
        out: out,
      );
      if (out == null) output.detachToParentScope();
      return output;
    } else {
      final bufferA = toContiguousFloat64Buffer(a);
      final bufferB = toContiguousFloat64Buffer(b);
      final bufferC = dispatchBatchedMatmulF64Gpu(
        a.device,
        bufferA,
        bufferB,
        batchCount: 1,
        m: m,
        k: kA,
        n: n,
        transposeB: true,
      );
      final output = writeFloat64BufferToArray<T>(
        a.device,
        bufferC,
        outShape,
        a.dtype,
        out: out,
      );
      if (out == null) output.detachToParentScope();
      return output;
    }
  });
}

/// Outer product of two [GpuArray] tensors [a] and [b].
///
/// Flattens [a] (`M = a.size`) and [b] (`N = b.size`) and produces a 2-D
/// matrix of shape `[M, N]` where `result[i, j] = a_flat[i] * b_flat[j]`.
///
/// Both [a] and [b] must reside on the same [GpuDevice] and share a common
/// [DType].
GpuArray<T> outer<T extends DTypeTag>(
  GpuArray<T> a,
  GpuArray<T> b, {
  GpuArray<T>? out,
}) {
  _checkPair(a, b, 'outer');
  final outShape = <int>[a.size, b.size];
  validateLinalgOut(out, a.device, outShape, a.dtype);

  return ResourceScope.scope(() {
    if (isComplexDType(a.dtype)) {
      final bufferA = toContiguousComplex128Buffer(a);
      final bufferB = toContiguousComplex128Buffer(b);
      final bufferC = dispatchBatchedMatmulC128Gpu(
        a.device,
        bufferA,
        bufferB,
        batchCount: 1,
        m: a.size,
        k: 1,
        n: b.size,
      );
      final output = writeComplex128BufferToArray<T>(
        a.device,
        bufferC,
        outShape,
        a.dtype,
        out: out,
      );
      if (out == null) output.detachToParentScope();
      return output;
    } else {
      final bufferA = toContiguousFloat64Buffer(a);
      final bufferB = toContiguousFloat64Buffer(b);
      final bufferC = dispatchBatchedMatmulF64Gpu(
        a.device,
        bufferA,
        bufferB,
        batchCount: 1,
        m: a.size,
        k: 1,
        n: b.size,
      );
      final output = writeFloat64BufferToArray<T>(
        a.device,
        bufferC,
        outShape,
        a.dtype,
        out: out,
      );
      if (out == null) output.detachToParentScope();
      return output;
    }
  });
}

/// Vector cross product of two [GpuArray] tensors [a] and [b].
///
/// The vector dimensions (length 2 or 3) are selected by [axisa] and [axisb]
/// (or [axis] if specified), and the resulting cross-product vectors are placed
/// along [axisc] (when at least one input has vector dimension 3). When both
/// inputs have vector dimension 2, the scalar z-component of the cross product
/// is produced.
///
/// Both [a] and [b] must be at least 1-D, reside on the same [GpuDevice], share
/// a common [DType], and have dimension 2 or 3 along their respective vector
/// axes.
GpuArray<T> cross<T extends DTypeTag>(
  GpuArray<T> a,
  GpuArray<T> b, {
  int axisa = -1,
  int axisb = -1,
  int axisc = -1,
  int? axis,
  GpuArray<T>? out,
}) {
  _checkPair(a, b, 'cross');
  if (a.ndim == 0) {
    throw ArgumentError.value(a.shape, 'a', 'Must be at least 1-D for cross.');
  }
  if (b.ndim == 0) {
    throw ArgumentError.value(b.shape, 'b', 'Must be at least 1-D for cross.');
  }

  final effAxisA = axis ?? axisa;
  final effAxisB = axis ?? axisb;
  final effAxisC = axis ?? axisc;

  final normA = _normalizeAxis(effAxisA, a.ndim, 'axisa');
  final normB = _normalizeAxis(effAxisB, b.ndim, 'axisb');
  final dimA = a.shape[normA];
  final dimB = b.shape[normB];
  if (dimA != 2 && dimA != 3) {
    throw ArgumentError.value(
      dimA,
      'axisa',
      'Must have dimension 2 or 3 along axisa, got $dimA.',
    );
  }
  if (dimB != 2 && dimB != 3) {
    throw ArgumentError.value(
      dimB,
      'axisb',
      'Must have dimension 2 or 3 along axisb, got $dimB.',
    );
  }

  final permA = <int>[
    for (var i = 0; i < a.ndim; i++)
      if (i != normA) i,
    normA,
  ];
  final permB = <int>[
    for (var i = 0; i < b.ndim; i++)
      if (i != normB) i,
    normB,
  ];

  final aBatchRaw = <int>[
    for (var i = 0; i < a.ndim; i++)
      if (i != normA) a.shape[i],
  ];
  final bBatchRaw = <int>[
    for (var i = 0; i < b.ndim; i++)
      if (i != normB) b.shape[i],
  ];

  final maxRank = math.max(aBatchRaw.length, bBatchRaw.length);
  final batchShape = List<int>.filled(maxRank, 1);
  final aBatchPadded = List<int>.filled(maxRank, 1);
  final bBatchPadded = List<int>.filled(maxRank, 1);
  for (var i = 0; i < maxRank; i++) {
    final da = i >= maxRank - aBatchRaw.length
        ? aBatchRaw[i - (maxRank - aBatchRaw.length)]
        : 1;
    final db = i >= maxRank - bBatchRaw.length
        ? bBatchRaw[i - (maxRank - bBatchRaw.length)]
        : 1;
    if (da != db && da != 1 && db != 1) {
      throw ArgumentError.value(
        b.shape,
        'b',
        'Must have broadcast-compatible non-vector dimensions with a.',
      );
    }
    aBatchPadded[i] = da;
    bBatchPadded[i] = db;
    batchShape[i] = math.max(da, db);
  }

  final both2d = dimA == 2 && dimB == 2;
  final outNdim = both2d ? batchShape.length : batchShape.length + 1;
  final normC = both2d ? 0 : _normalizeAxis(effAxisC, outNdim, 'axisc');
  final outShape = both2d
      ? batchShape
      : <int>[...batchShape.sublist(0, normC), 3, ...batchShape.sublist(normC)];
  validateLinalgOut(out, a.device, outShape, a.dtype);

  final batchCount = batchShape.isEmpty
      ? 1
      : batchShape.reduce((x, y) => x * y);

  return ResourceScope.scope(() {
    final aView = _isIdentityPermutation(permA) ? a : a.transpose(permA);
    final bView = _isIdentityPermutation(permB) ? b : b.transpose(permB);
    final bufferA = toContiguousFloat64Buffer(aView);
    final bufferB = toContiguousFloat64Buffer(bView);
    final bufferC = dispatchCrossF64Gpu(
      a.device,
      bufferA,
      bufferB,
      batchCount: batchCount,
      dimA: dimA,
      dimB: dimB,
      batchShape: batchShape,
      aBatchShape: aBatchPadded,
      bBatchShape: bBatchPadded,
    );

    if (both2d || normC == outNdim - 1) {
      final output = writeFloat64BufferToArray<T>(
        a.device,
        bufferC,
        outShape,
        a.dtype,
        out: out,
      );
      if (out == null) output.detachToParentScope();
      return output;
    }

    final unpermutedShape = <int>[...batchShape, 3];
    final tempArray = writeFloat64BufferToArray<T>(
      a.device,
      bufferC,
      unpermutedShape,
      a.dtype,
    );
    final outPerm = <int>[
      for (var i = 0; i < normC; i++) i,
      outNdim - 1,
      for (var i = normC; i < outNdim - 1; i++) i,
    ];
    final permutedView = tempArray.transpose(outPerm);
    final output = copyGpuArray(permutedView, out: out);
    if (out == null) output.detachToParentScope();
    return output;
  });
}
