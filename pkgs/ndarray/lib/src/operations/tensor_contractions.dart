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

import "dart:ffi" as ffi;
import "package:openblas/openblas.dart";
import "dart:math" as math;
import "../ndarray.dart";
import "../scratch_arena.dart";
import "../ndarray_bindings.dart";
import "math.dart";
import "linalg.dart";
import "stats.dart";
import "helpers.dart";

NDArray _createRootArray(List<int> shape, DType dtype) => switch (dtype) {
  DType.float64 => NDArray<Float64>.create(shape, DType.float64),
  DType.float32 => NDArray<Float32>.create(shape, DType.float32),
  DType.float16 => NDArray<Float16>.create(shape, DType.float16),
  DType.bfloat16 => NDArray<BFloat16>.create(shape, DType.bfloat16),
  DType.int64 => NDArray<Int64>.create(shape, DType.int64),
  DType.int32 => NDArray<Int32>.create(shape, DType.int32),
  DType.int16 => NDArray<Int16>.create(shape, DType.int16),
  DType.int8 => NDArray<Int8>.create(shape, DType.int8),
  DType.uint64 => NDArray<Uint64>.create(shape, DType.uint64),
  DType.uint32 => NDArray<Uint32>.create(shape, DType.uint32),
  DType.uint16 => NDArray<Uint16>.create(shape, DType.uint16),
  DType.uint8 => NDArray<Uint8>.create(shape, DType.uint8),
  DType.complex128 => NDArray<Complex128>.create(shape, DType.complex128),
  DType.complex64 => NDArray<Complex64>.create(shape, DType.complex64),
  DType.boolean => NDArray<Boolean>.create(shape, DType.boolean),
};

NDArray _createView(
  NDArray parent,
  List<int> shape,
  List<int> strides,
) => switch (parent.dtype) {
  DType.float64 => NDArray<Float64>.view(
    parent,
    shape: shape,
    strides: strides,
  ),
  DType.float32 => NDArray<Float32>.view(
    parent,
    shape: shape,
    strides: strides,
  ),
  DType.float16 => NDArray<Float16>.view(
    parent,
    shape: shape,
    strides: strides,
  ),
  DType.bfloat16 => NDArray<BFloat16>.view(
    parent,
    shape: shape,
    strides: strides,
  ),
  DType.int64 => NDArray<Int64>.view(parent, shape: shape, strides: strides),
  DType.int32 => NDArray<Int32>.view(parent, shape: shape, strides: strides),
  DType.int16 => NDArray<Int16>.view(parent, shape: shape, strides: strides),
  DType.int8 => NDArray<Int8>.view(parent, shape: shape, strides: strides),
  DType.uint64 => NDArray<Uint64>.view(parent, shape: shape, strides: strides),
  DType.uint32 => NDArray<Uint32>.view(parent, shape: shape, strides: strides),
  DType.uint16 => NDArray<Uint16>.view(parent, shape: shape, strides: strides),
  DType.uint8 => NDArray<Uint8>.view(parent, shape: shape, strides: strides),
  DType.complex128 => NDArray<Complex128>.view(
    parent,
    shape: shape,
    strides: strides,
  ),
  DType.complex64 => NDArray<Complex64>.view(
    parent,
    shape: shape,
    strides: strides,
  ),
  DType.boolean => NDArray<Boolean>.view(
    parent,
    shape: shape,
    strides: strides,
  ),
};

NDArray<R> _asTyped<R extends DTypeTag>(NDArray arr) {
  if (arr is NDArray<R>) return arr;
  return _createView(arr, arr.shape, arr.strides) as NDArray<R>;
}

(int, int) _physicalByteSpan(NDArray x) {
  if (x.size == 0) {
    return (x.pointer.address, x.pointer.address);
  }
  var minElemOffset = 0;
  var maxElemOffset = 0;
  for (var d = 0; d < x.shape.length; d++) {
    final stride = x.strides[d];
    final size = x.shape[d];
    if (stride > 0) {
      maxElemOffset += (size - 1) * stride;
    } else if (stride < 0) {
      minElemOffset += (size - 1) * stride;
    }
  }
  final byteWidth = x.dtype.byteWidth;
  final startAddr = x.pointer.address + minElemOffset * byteWidth;
  final endAddr = x.pointer.address + (maxElemOffset + 1) * byteWidth;
  return (startAddr, endAddr);
}

bool _arraysOverlap(NDArray x, NDArray y) {
  if (identical(x, y) || x.pointer.address == y.pointer.address) return true;
  if (x.size == 0 || y.size == 0) return false;
  final (startX, endX) = _physicalByteSpan(x);
  final (startY, endY) = _physicalByteSpan(y);
  return startX < endY && startY < endX;
}

NDArray<R> _returnFromScope<R extends DTypeTag>(
  NDArray res,
  List<NDArray> callerInputs, {
  NDArray<R>? out,
}) {
  if (out != null) {
    if (!identical(res, out)) {
      res.copy(out: out);
    }
    return out;
  }
  final isCallerView = callerInputs.any((input) => _arraysOverlap(res, input));
  if (isCallerView) {
    if (res is NDArray<R> && res.isView) {
      return res;
    }
    return _createView(res, res.shape, res.strides) as NDArray<R>;
  }
  if (!res.isView && res is NDArray<R>) {
    return res.detachToParentScope();
  }
  final rootCopy = _createRootArray(res.shape, res.dtype) as NDArray<R>;
  res.copy(out: rootCopy);
  return rootCopy.detachToParentScope();
}

NDArray<T> _diagonalView<T extends DTypeTag>(NDArray<T> arr, int ax1, int ax2) {
  if (arr.shape[ax1] != arr.shape[ax2]) {
    throw ArgumentError(
      "Dimension mismatch for diagonal extraction: axis $ax1 size (${arr.shape[ax1]}) != axis $ax2 size (${arr.shape[ax2]}).",
    );
  }
  final minAx = math.min(ax1, ax2);
  final maxAx = math.max(ax1, ax2);

  final newShape = <int>[];
  final newStrides = <int>[];

  for (var i = 0; i < arr.shape.length; i++) {
    if (i == minAx) {
      newShape.add(arr.shape[minAx]);
      newStrides.add(arr.strides[minAx] + arr.strides[maxAx]);
    } else if (i != maxAx) {
      newShape.add(arr.shape[i]);
      newStrides.add(arr.strides[i]);
    }
  }

  return NDArray<T>.view(arr, shape: newShape, strides: newStrides);
}

/// Computes tensor dot product along specified axes for arrays [a] and [b].
///
/// Given two tensors [a] and [b] and an array of axes (or integer number of axes),
/// sums products of elements over the specified axes.
///
/// **Axes specification:**
/// Represents contracted axis specifications for tensor dot products ([tensordot]).
///
/// Use one of the three primary constructors to create a [TensordotAxes] instance:
/// - [TensordotAxes.count]: Contracts the last `N` axes of array A with the first `N` axes of array B.
/// - [TensordotAxes.explicit]: Contracts explicit axis lists of array A with corresponding axes of array B.
/// - [TensordotAxes.pair]: Contracts a single pair of axes between array A and array B.
final class TensordotAxes {
  /// The number of contracted axes, or `null` if explicit axis lists are provided.
  final int? count;

  /// Explicit contracted axis indices for array A, or `null` if [count] is used.
  final List<int>? explicitAxesA;

  /// Explicit contracted axis indices for array B, or `null` if [count] is used.
  final List<int>? explicitAxesB;

  /// Creates a [TensordotAxes] instance from a flexible [axes] parameter representation.
  ///
  /// Supported types/formats for [axes]:
  /// - An [int] `N`: contracts the last `N` axes of array A with the first `N` axes of array B.
  ///   `0` computes the tensor outer product.
  /// - A `(int, int)` record pair (e.g. `(1, 0)`): contracts a single axis pair.
  /// - A `(List<int>, List<int>)` record pair or `List<List<int>>` (e.g. `([1], [0])`):
  ///   contracts explicit axis index lists.
  /// - A [TensordotAxes] object: returned as-is.
  ///
  /// Throws [ArgumentError] if [axes] is of an unsupported type or format.
  factory TensordotAxes.from(Object? axes) {
    if (axes is TensordotAxes) {
      return axes;
    }
    if (axes is int) {
      return TensordotAxes.count(axes);
    }
    if (axes is (int, int)) {
      return TensordotAxes.pair(axes.$1, axes.$2);
    }
    if (axes is (List<int>, List<int>)) {
      return TensordotAxes.explicit(axes.$1, axes.$2);
    }
    if (axes is List) {
      if (axes.length != 2) {
        throw ArgumentError(
          "List axes specification must contain exactly 2 elements, got length ${axes.length}.",
        );
      }
      if (axes[0] is int && axes[1] is int) {
        return TensordotAxes.pair(axes[0] as int, axes[1] as int);
      }
      if (axes[0] is Iterable && axes[1] is Iterable) {
        return TensordotAxes.explicit(
          (axes[0] as Iterable).cast<int>().toList(),
          (axes[1] as Iterable).cast<int>().toList(),
        );
      }
    }
    throw ArgumentError("Unsupported axes parameter type or format: $axes.");
  }

  /// 1. Contracts the last [n] axes of array A with the first [n] axes of array B.
  ///
  /// It is an error if [n] is negative.
  const TensordotAxes.count(int n)
    : count = n,
      explicitAxesA = null,
      explicitAxesB = null;

  /// 2. Contracts specified axis lists [axesA] of array A with corresponding axes [axesB] of array B.
  ///
  /// It is an error if [axesA] and [axesB] have different lengths.
  TensordotAxes.explicit(List<int> axesA, List<int> axesB)
    : count = null,
      explicitAxesA = axesA,
      explicitAxesB = axesB {
    if (axesA.length != axesB.length) {
      throw ArgumentError(
        "Axes length mismatch: ${axesA.length} vs ${axesB.length}.",
      );
    }
  }

  /// 3. Contracts a single axis pair [axisA] of array A with [axisB] of array B.
  TensordotAxes.pair(int axisA, int axisB)
    : count = null,
      explicitAxesA = [axisA],
      explicitAxesB = [axisB];

  /// Resolves contracted axis index lists for tensors of rank [rankA] and [rankB].
  (List<int>, List<int>) resolve(int rankA, int rankB) {
    if (count != null) {
      final n = count!;
      if (n < 0 || n > rankA || n > rankB) {
        throw ArgumentError(
          "Invalid number of contracted axes $n for tensor ranks $rankA and $rankB.",
        );
      }
      final axesA = List.generate(n, (i) => rankA - n + i);
      final axesB = List.generate(n, (i) => i);
      return (axesA, axesB);
    }

    final axesA = explicitAxesA!;
    final axesB = explicitAxesB!;
    return (List<int>.from(axesA), List<int>.from(axesB));
  }
}

/// Computes tensor dot product along specified contracted axes for arrays [a] and [b].
///
/// Given two tensors [a] and [b], sums products of elements over specified contracted axes.
///
/// ### Axes Specification
/// [axes] can be passed as:
/// - An [int] `N`: Contracts the last `N` axes of [a] and first `N` axes of [b].
///   Passing `0` computes the tensor outer product.
/// - A pair `(int, int)` record (e.g., `(1, 0)`): Contracts a single axis pair.
/// - A `(List<int>, List<int>)` record or `List<List<int>>` (e.g., `([1], [0])`): Contracts explicit lists of axes.
/// - A [TensordotAxes] object: Pre-constructed axes specification.
///
/// **Preconditions:**
/// - It is an error if [a], [b], or [out] is disposed.
/// - It is an error if axes specifications are invalid or out of bounds.
/// - It is an error if contracted axis dimensions mismatch.
/// - It is an error if [out] has incompatible shape or dtype.
///
/// **Performance considerations:**
/// - Converted to an optimized matrix multiplication (`matmul` / BLAS GEMM) via axis transposition and reshaping.
/// - Outer product (`axes == 0`) uses optimized elementwise broadcasting.
///
/// **Example:**
/// ```dart
/// final a = NDArray.arange(60.0).reshape([3, 4, 5]);
/// final b = NDArray.arange(24.0).reshape([4, 3, 2]);
/// final c = tensordot(a, b, axes: ([1, 0], [0, 1]));
/// print(c.shape); // [5, 2]
/// ```
///
/// Reference: [NumPy tensordot](https://numpy.org/doc/stable/reference/generated/numpy.tensordot.html)
NDArray<T> tensordot<T extends DTypeTag>(
  NDArray<T> a,
  NDArray<T> b, {
  Object axes = const TensordotAxes.count(2),
  NDArray<T>? out,
}) {
  if (a.isDisposed || b.isDisposed) {
    throw StateError("Cannot execute tensordot() on a disposed array.");
  }
  if (out != null && out.isDisposed) {
    throw StateError(
      "Cannot write tensordot result to a disposed output array.",
    );
  }
  if (a.dtype != b.dtype) {
    throw ArgumentError.value(
      b.dtype,
      'b',
      'Must have the same dtype as a (${a.dtype})',
    );
  }

  final resolvedAxes = TensordotAxes.from(axes);
  final (axesA, axesB) = resolvedAxes.resolve(a.shape.length, b.shape.length);

  final normAxesA = axesA
      .map((ax) => ax < 0 ? a.shape.length + ax : ax)
      .toList();
  final normAxesB = axesB
      .map((ax) => ax < 0 ? b.shape.length + ax : ax)
      .toList();

  for (var i = 0; i < normAxesA.length; i++) {
    final axA = normAxesA[i];
    final axB = normAxesB[i];
    if (axA < 0 || axA >= a.shape.length) {
      throw RangeError.range(axA, 0, a.shape.length - 1, "axisA");
    }
    if (axB < 0 || axB >= b.shape.length) {
      throw RangeError.range(axB, 0, b.shape.length - 1, "axisB");
    }
    if (a.shape[axA] != b.shape[axB]) {
      throw ArgumentError(
        "Dimension mismatch at contracted axis: ${a.shape[axA]} != ${b.shape[axB]}",
      );
    }
  }

  if (normAxesA.isEmpty && normAxesB.isEmpty) {
    final targetShape = [...a.shape, ...b.shape];
    final targetDType = resolveDType(a.dtype, b.dtype);
    if (out != null) {
      if (!listEquals(out.shape, targetShape) || out.dtype != targetDType) {
        throw ArgumentError(
          "Provided out buffer has incompatible shape or dtype (expected shape $targetShape and dtype $targetDType, got shape ${out.shape} and dtype ${out.dtype}).",
        );
      }
    }

    return NDArray.scope(() {
      final aShapeExpanded = [...a.shape, ...List.filled(b.shape.length, 1)];
      final bShapeExpanded = [...List.filled(a.shape.length, 1), ...b.shape];

      final aView = a.reshape(aShapeExpanded);
      final bView = b.reshape(bShapeExpanded);

      final res = multiply<DTypeTag>(aView, bView, out: out);
      return _returnFromScope<T>(res, [a, b], out: out);
    });
  }

  final freeA = List.generate(
    a.shape.length,
    (i) => i,
  ).where((i) => !normAxesA.contains(i)).toList();
  final freeB = List.generate(
    b.shape.length,
    (i) => i,
  ).where((i) => !normAxesB.contains(i)).toList();

  final targetShape = [
    ...freeA.map((i) => a.shape[i]),
    ...freeB.map((i) => b.shape[i]),
  ];

  final targetDType = resolveDType(a.dtype, b.dtype);
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != targetDType) {
      throw ArgumentError(
        "Provided out buffer has incompatible shape or dtype (expected shape $targetShape and dtype $targetDType, got shape ${out.shape} and dtype ${out.dtype}).",
      );
    }
  }

  return NDArray.scope(() {
    if (freeA.isEmpty && freeB.isEmpty) {
      bool isSeq(List<int> axes) {
        for (var i = 0; i < axes.length; i++) {
          if (axes[i] != i) return false;
        }
        return true;
      }

      final aToUse = a.dtype == targetDType ? a : castNDArray(a, targetDType);
      final bToUse = b.dtype == targetDType ? b : castNDArray(b, targetDType);

      if (aToUse.isContiguous &&
          bToUse.isContiguous &&
          isSeq(normAxesA) &&
          isSeq(normAxesB)) {
        final n = aToUse.size;
        if (targetDType == DType.float64) {
          final val = cblas_ddot(
            n,
            aToUse.pointer.cast<ffi.Double>(),
            1,
            bToUse.pointer.cast<ffi.Double>(),
            1,
          );
          return _returnFromScope<T>(
            NDArray.scalar(val, dtype: DType.float64),
            [a, b],
            out: out,
          );
        } else if (targetDType == DType.float32) {
          final val = cblas_sdot(
            n,
            aToUse.pointer.cast<ffi.Float>(),
            1,
            bToUse.pointer.cast<ffi.Float>(),
            1,
          );
          return _returnFromScope<T>(
            NDArray.scalar(val, dtype: DType.float32),
            [a, b],
            out: out,
          );
        }
      }
    }

    final aPerm = a.transpose([...freeA, ...normAxesA]);
    final bPerm = b.transpose([...normAxesB, ...freeB]);

    final m = freeA.map((i) => a.shape[i]).fold(1, (x, y) => x * y);
    final k = normAxesA.map((i) => a.shape[i]).fold(1, (x, y) => x * y);
    final n = freeB.map((i) => b.shape[i]).fold(1, (x, y) => x * y);

    final a2D = aPerm.reshape([m, k]);
    final b2D = bPerm.reshape([k, n]);

    final res2D = matmul<DTypeTag>(a2D, b2D);
    final res = listEquals(res2D.shape, targetShape)
        ? res2D
        : res2D.reshape(targetShape);

    return _returnFromScope<T>(res, [a, b], out: out);
  });
}

/// Represents index subscript specifications for Einstein summation ([einsum]).
///
/// Subscripts define how input tensor axes map to contracted or output dimensions using numeric index identifiers.
///
/// Use one of the three primary constructors to create an [EinsumSubscripts] instance:
/// - [EinsumSubscripts.fromIndices]: Construct directly using integer axis IDs. Ellipsis `...` is represented by `-1`.
/// - [EinsumSubscripts.parse]: Parses standard Einstein summation notation strings (e.g. `'ij,jk->ik'`).
/// - [EinsumSubscripts.fromLabels]: Construct using string label lists for input operands and optional output.
final class EinsumSubscripts {
  /// The numeric index IDs for each operand tensor axis. `-1` represents an ellipsis (`...`).
  final List<List<int>> operandIndices;

  /// The numeric index IDs for output tensor axes, or `null` if implicit output.
  final List<int>? outputIndices;

  /// Whether any operand or output subscript contains an ellipsis (`-1`).
  final bool hasEllipsis;

  const EinsumSubscripts._(
    this.operandIndices,
    this.outputIndices,
    this.hasEllipsis,
  );

  /// 1. Creates an [EinsumSubscripts] from explicit integer index lists for operands and optional output.
  ///
  /// Each operand's axes are assigned integer IDs. Contracted axes share matching integer IDs across operands.
  /// Use `-1` to represent an ellipsis (`...`) for batch broadcasting.
  ///
  /// It is an error if [inputIndices] is empty.
  factory EinsumSubscripts.fromIndices(
    List<List<int>> inputIndices, [
    List<int>? outputIndices,
  ]) {
    if (inputIndices.isEmpty) {
      throw ArgumentError('inputIndices cannot be empty.');
    }

    var containsEllipsis = false;
    for (final op in inputIndices) {
      if (op.where((idx) => idx == -1).length > 1) {
        throw ArgumentError(
          'Operand indices cannot contain more than one ellipsis (-1).',
        );
      }
      if (op.contains(-1)) containsEllipsis = true;
    }
    if (outputIndices != null) {
      if (outputIndices.where((idx) => idx == -1).length > 1) {
        throw ArgumentError(
          'Output indices cannot contain more than one ellipsis (-1).',
        );
      }
      if (outputIndices.contains(-1)) {
        containsEllipsis = true;
      }
    }

    final copyInputs = inputIndices
        .map((l) => List<int>.unmodifiable(l))
        .toList();
    final copyOutput = outputIndices != null
        ? List<int>.unmodifiable(outputIndices)
        : null;

    return EinsumSubscripts._(
      List<List<int>>.unmodifiable(copyInputs),
      copyOutput,
      containsEllipsis,
    );
  }

  static final Map<String, EinsumSubscripts> _parseCache = {};

  /// 2. Parses a subscript string in Einstein summation notation (e.g. `'ij,jk->ik'`, `'...ij,...jk->...ik'`, or `'ii->'`).
  ///
  /// Maps character labels (e.g. `'i'`, `'j'`) to numeric index IDs internally.
  ///
  /// It is an error if subscript syntax is invalid.
  factory EinsumSubscripts.parse(String subscripts) {
    final cleanSub = subscripts.replaceAll(' ', '');
    return _parseCache.putIfAbsent(cleanSub, () => _parseImpl(cleanSub));
  }

  static EinsumSubscripts _parseImpl(String cleanSub) {
    final parts = cleanSub.split('->');
    if (parts.length > 2) {
      throw ArgumentError(
        'Invalid einsum subscript: multiple "->" delimiters found.',
      );
    }

    final inStr = parts[0];
    final outStr = parts.length == 2 ? parts[1] : null;

    final rawOperandTokens = inStr.isEmpty
        ? <List<String>>[]
        : inStr.split(',').map((s) => _tokenizeTerm(s)).toList();

    final rawOutTokens = outStr != null ? _tokenizeTerm(outStr) : null;

    return EinsumSubscripts.fromLabels(rawOperandTokens, rawOutTokens);
  }

  /// 3. Creates an [EinsumSubscripts] from string label lists for input operands and optional output.
  ///
  /// String labels (e.g. `'i'`, `'j'`, `'k'`) are automatically assigned unique numeric index IDs.
  /// Use `'...'` to represent an ellipsis for batch broadcasting.
  ///
  /// It is an error if [inputLabels] is empty.
  factory EinsumSubscripts.fromLabels(
    List<List<String>> inputLabels, [
    List<String>? outputLabels,
  ]) {
    if (inputLabels.isEmpty) {
      throw ArgumentError('inputLabels cannot be empty.');
    }

    final distinctLabelsSet = <String>{};
    for (final list in inputLabels) {
      if (list.where((lbl) => lbl == '...').length > 1) {
        throw ArgumentError(
          'Operand subscript cannot contain more than one ellipsis ("...").',
        );
      }
      for (final lbl in list) {
        if (lbl != '...') distinctLabelsSet.add(lbl);
      }
    }
    if (outputLabels != null) {
      if (outputLabels.where((lbl) => lbl == '...').length > 1) {
        throw ArgumentError(
          'Output subscript cannot contain more than one ellipsis ("...").',
        );
      }
      for (final lbl in outputLabels) {
        if (lbl != '...') distinctLabelsSet.add(lbl);
      }
    }

    final sortedLabels = distinctLabelsSet.toList()..sort();
    final labelToId = <String, int>{};
    for (var i = 0; i < sortedLabels.length; i++) {
      labelToId[sortedLabels[i]] = i;
    }

    int getId(String label) {
      if (label == '...') return -1;
      return labelToId[label]!;
    }

    final numericInputs = inputLabels
        .map((list) => list.map((lbl) => getId(lbl)).toList())
        .toList();

    final numericOutput = outputLabels?.map((lbl) => getId(lbl)).toList();

    return EinsumSubscripts.fromIndices(numericInputs, numericOutput);
  }

  static List<String> _tokenizeTerm(String term) {
    final tokens = <String>[];
    var ellipsisCount = 0;
    for (var i = 0; i < term.length; i++) {
      if (i + 2 < term.length && term.substring(i, i + 3) == '...') {
        ellipsisCount++;
        if (ellipsisCount > 1) {
          throw ArgumentError(
            'Invalid einsum subscript: operand or output term cannot contain more than one ellipsis ("..."). Term: "$term"',
          );
        }
        tokens.add('...');
        i += 2;
      } else {
        tokens.add(term[i]);
      }
    }
    return tokens;
  }
}

/// Evaluates Einstein summation convention over multiple multi-dimensional array operands.
///
/// Einstein summation provides a concise notation for expressing complex tensor contractions,
/// matrix multiplications, transpositions, traces, and array operations by specifying index labels
/// for each input operand and the resulting output.
///
/// Subscripts are specified using an [EinsumSubscripts] object created via one of its constructors:
/// - [EinsumSubscripts.fromIndices]: Construct directly using integer axis IDs.
/// - [EinsumSubscripts.parse]: Parses a standard notation string (e.g. `EinsumSubscripts.parse('ij,jk->ik')`).
/// - [EinsumSubscripts.fromLabels]: Construct using string label lists.
///
/// **Preconditions:**
/// - It is an error if [operands] is empty.
/// - It is an error if any operand or [out] is disposed.
/// - It is an error if subscript terms count does not match operands count.
/// - It is an error if subscript format or dimension lengths are invalid or mismatched.
/// - It is an error if [out] has incompatible shape or dtype.
///
/// **Performance considerations:**
/// - Optimized contraction paths (matrix multiplications, inner products, traces, diagonal reductions)
///   are automatically selected to leverage BLAS GEMM where possible.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
/// final b = NDArray.fromList([5.0, 6.0, 7.0, 8.0], [2, 2], DType.float64);
/// final c = einsum(EinsumSubscripts.parse('ij,jk->ik'), [a, b]);
/// print(c.toList()); // [19.0, 22.0, 43.0, 50.0]
/// ```
///
/// Reference: [NumPy einsum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html)
NDArray<T> einsum<T extends DTypeTag>(
  EinsumSubscripts subscripts,
  List<NDArray<T>> operands, {
  NDArray<T>? out,
}) {
  if (operands.isEmpty) {
    throw ArgumentError("einsum requires at least one operand.");
  }
  for (var i = 0; i < operands.length; i++) {
    if (operands[i].isDisposed) {
      throw StateError(
        "Cannot execute einsum on disposed operand at index $i.",
      );
    }
  }
  if (out != null && out.isDisposed) {
    throw StateError("Cannot write einsum result to a disposed output array.");
  }

  if (subscripts.operandIndices.length != operands.length) {
    throw ArgumentError(
      "Number of subscript terms (${subscripts.operandIndices.length}) does not match number of operands (${operands.length}).",
    );
  }

  var operandSubs = subscripts.operandIndices;
  List<int>? outSub = subscripts.outputIndices;

  if (subscripts.hasEllipsis) {
    int maxEllipsisDims = 0;
    for (var i = 0; i < operands.length; i++) {
      final sub = operandSubs[i];
      final explicitCount = sub.where((id) => id != -1).length;
      final ellipsisDims = operands[i].shape.length - explicitCount;
      if (ellipsisDims < 0) {
        throw ArgumentError(
          "Operand $i shape ${operands[i].shape} has fewer dimensions than explicit labels in $sub.",
        );
      }
      if (ellipsisDims > maxEllipsisDims) {
        maxEllipsisDims = ellipsisDims;
      }
    }

    final ellipsisIds = List<int>.generate(maxEllipsisDims, (i) => 10000 + i);

    final resolvedOperandSubs = <List<int>>[];
    for (var i = 0; i < operands.length; i++) {
      final sub = operandSubs[i];
      if (sub.contains(-1)) {
        final explicitCount = sub.where((id) => id != -1).length;
        final count = operands[i].shape.length - explicitCount;
        final activeEllipsis = ellipsisIds.sublist(maxEllipsisDims - count);
        final newSub = <int>[];
        for (final id in sub) {
          if (id == -1) {
            newSub.addAll(activeEllipsis);
          } else {
            newSub.add(id);
          }
        }
        resolvedOperandSubs.add(newSub);
      } else {
        resolvedOperandSubs.add(sub);
      }
    }

    List<int> resolvedOutSub;
    if (outSub != null) {
      if (outSub.contains(-1)) {
        final newOut = <int>[];
        for (final id in outSub) {
          if (id == -1) {
            newOut.addAll(ellipsisIds);
          } else {
            newOut.add(id);
          }
        }
        resolvedOutSub = newOut;
      } else {
        resolvedOutSub = outSub;
      }
    } else {
      final labelCounts = <int, int>{};
      for (final sub in resolvedOperandSubs) {
        for (final id in sub) {
          labelCounts[id] = (labelCounts[id] ?? 0) + 1;
        }
      }
      final singleLabels =
          labelCounts.entries
              .where((e) => e.value == 1 && !ellipsisIds.contains(e.key))
              .map((e) => e.key)
              .toList()
            ..sort();
      resolvedOutSub = [...ellipsisIds, ...singleLabels];
    }

    operandSubs = resolvedOperandSubs;
    outSub = resolvedOutSub;
  } else {
    if (outSub == null) {
      final labelCounts = <int, int>{};
      for (final sub in operandSubs) {
        for (final id in sub) {
          labelCounts[id] = (labelCounts[id] ?? 0) + 1;
        }
      }
      final singleLabels =
          labelCounts.entries
              .where((e) => e.value == 1)
              .map((e) => e.key)
              .toList()
            ..sort();
      outSub = singleLabels;
    }
  }

  final finalOutSub = outSub;

  for (var i = 0; i < operands.length; i++) {
    if (operandSubs[i].length != operands[i].shape.length) {
      throw ArgumentError(
        "Operand $i shape ${operands[i].shape} rank does not match subscript ${operandSubs[i]} length (${operandSubs[i].length}).",
      );
    }
  }

  return NDArray.scope(() {
    final labelSizes = <int, int>{};
    for (var i = 0; i < operands.length; i++) {
      final sub = operandSubs[i];
      final shape = operands[i].shape;
      for (var j = 0; j < sub.length; j++) {
        final id = sub[j];
        final size = shape[j];
        if (labelSizes.containsKey(id)) {
          if (labelSizes[id] != size) {
            throw ArgumentError(
              "Dimension mismatch for label ID $id: ${labelSizes[id]} vs $size",
            );
          }
        } else {
          labelSizes[id] = size;
        }
      }
    }

    if (operands.length == 1) {
      var op = _asTyped<T>(operands[0]);
      var sub = operandSubs[0];

      final seenInOp = <int, int>{};
      for (var j = 0; j < sub.length; j++) {
        final id = sub[j];
        if (seenInOp.containsKey(id)) {
          final firstAx = seenInOp[id]!;
          final secondAx = j;
          op = _diagonalView(op, firstAx, secondAx);
          final newSub = List<int>.from(sub)..removeAt(secondAx);
          sub = newSub;
          j--;
        } else {
          seenInOp[id] = j;
        }
      }

      final remainingIds = sub;
      final axesToSum = <int>[];
      final keptIds = <int>[];

      for (var j = 0; j < remainingIds.length; j++) {
        final id = remainingIds[j];
        if (!finalOutSub.contains(id)) {
          axesToSum.add(j);
        } else {
          keptIds.add(id);
        }
      }

      NDArray<T> res = op;
      axesToSum.sort((a, b) => b.compareTo(a));
      for (final ax in axesToSum) {
        res = sumAs<T, T>(res, res.dtype, axis: ax);
      }

      if (keptIds.length > 1) {
        final perm = <int>[];
        for (final id in finalOutSub) {
          perm.add(keptIds.indexOf(id));
        }
        res = res.transpose(perm);
      }

      final targetDType = res.dtype;
      if (out != null) {
        if (!listEquals(out.shape, res.shape) || out.dtype != targetDType) {
          throw ArgumentError(
            "Provided out buffer has incompatible shape or dtype (expected shape ${res.shape} and dtype $targetDType, got shape ${out.shape} and dtype ${out.dtype}).",
          );
        }
      }
      return _returnFromScope<T>(res, operands, out: out);
    }

    if (operands.length == 2) {
      final subA = operandSubs[0];
      final subB = operandSubs[1];

      bool hasRepeated(List<int> s) {
        final seen = <int>{};
        for (final id in s) {
          if (seen.contains(id)) return true;
          seen.add(id);
        }
        return false;
      }

      if (!hasRepeated(subA) && !hasRepeated(subB)) {
        if (subA.length == 2 &&
            subB.length == 2 &&
            finalOutSub.length == 2 &&
            subA[0] != subA[1] &&
            subB[0] != subB[1] &&
            subA[1] == subB[0] &&
            subA[0] == finalOutSub[0] &&
            subB[1] == finalOutSub[1]) {
          final res = matmul<T>(operands[0], operands[1], out: out);
          return _returnFromScope<T>(res, operands, out: out);
        }

        if (subA.length == 3 &&
            subB.length == 3 &&
            finalOutSub.length == 3 &&
            subA[0] == subB[0] &&
            subA[0] == finalOutSub[0] &&
            subA[2] == subB[1] &&
            subA[1] == finalOutSub[1] &&
            subB[2] == finalOutSub[2]) {
          final res = matmul<T>(operands[0], operands[1], out: out);
          return _returnFromScope<T>(res, operands, out: out);
        }

        if (subA.length == 1 &&
            subB.length == 1 &&
            finalOutSub.isEmpty &&
            subA[0] == subB[0]) {
          final res = matmul<T>(operands[0], operands[1], out: out);
          return _returnFromScope<T>(res, operands, out: out);
        }

        if (subA.length == 1 &&
            subB.length == 1 &&
            finalOutSub.length == 2 &&
            subA[0] == finalOutSub[0] &&
            subB[0] == finalOutSub[1]) {
          final aCol = operands[0].reshape([operands[0].shape[0], 1]);
          final bRow = operands[1].reshape([1, operands[1].shape[0]]);
          final res = multiply<T>(aCol, bRow, out: out);
          return _returnFromScope<T>(res, operands, out: out);
        }

        if (subA.length >= 3 &&
            subB.length == subA.length &&
            finalOutSub.length == subA.length) {
          final kBatch = subA.length - 2;
          bool isBatchMatmul = true;
          for (var b = 0; b < kBatch; b++) {
            if (subA[b] != subB[b] || subA[b] != finalOutSub[b]) {
              isBatchMatmul = false;
              break;
            }
          }
          if (isBatchMatmul &&
              subA[kBatch + 1] == subB[kBatch] &&
              subA[kBatch] == finalOutSub[kBatch] &&
              subB[kBatch + 1] == finalOutSub[kBatch + 1]) {
            final res = matmul<T>(operands[0], operands[1], out: out);
            return _returnFromScope<T>(res, operands, out: out);
          }
        }

        final shared = subA.where((id) => subB.contains(id)).toList();
        final contracted = shared
            .where((id) => !finalOutSub.contains(id))
            .toList();
        final batch = shared.where((id) => finalOutSub.contains(id)).toList();

        NDArray? fast2OpRes;
        List<int>? fast2OpResIds;

        if (batch.isEmpty) {
          final axesA = contracted.map((id) => subA.indexOf(id)).toList();
          final axesB = contracted.map((id) => subB.indexOf(id)).toList();

          final freeA = subA.where((id) => !contracted.contains(id)).toList();
          final freeB = subB.where((id) => !contracted.contains(id)).toList();

          final allFreeInOut =
              freeA.every((id) => finalOutSub.contains(id)) &&
              freeB.every((id) => finalOutSub.contains(id));

          if (allFreeInOut) {
            fast2OpResIds = [...freeA, ...freeB];
            fast2OpRes = tensordot<T>(
              operands[0],
              operands[1],
              axes: TensordotAxes.explicit(axesA, axesB),
            );
          }
        } else {
          final freeA = subA.where((id) => !shared.contains(id)).toList();
          final freeB = subB.where((id) => !shared.contains(id)).toList();

          final allFreeInOut =
              freeA.every((id) => finalOutSub.contains(id)) &&
              freeB.every((id) => finalOutSub.contains(id));

          if (allFreeInOut && contracted.isNotEmpty) {
            final batchShape = batch.map((id) => labelSizes[id]!).toList();
            final m = freeA
                .map((id) => labelSizes[id]!)
                .fold(1, (x, y) => x * y);
            final k = contracted
                .map((id) => labelSizes[id]!)
                .fold(1, (x, y) => x * y);
            final n = freeB
                .map((id) => labelSizes[id]!)
                .fold(1, (x, y) => x * y);

            final permA = [
              ...batch,
              ...freeA,
              ...contracted,
            ].map((id) => subA.indexOf(id)).toList();
            final permB = [
              ...batch,
              ...contracted,
              ...freeB,
            ].map((id) => subB.indexOf(id)).toList();

            final numBatch = batchShape.fold(1, (x, y) => x * y);

            final isIdentityA = listEquals(
              permA,
              List.generate(permA.length, (i) => i),
            );
            final isIdentityB = listEquals(
              permB,
              List.generate(permB.length, (i) => i),
            );

            final NDArray a3D;
            if (isIdentityA && operands[0].isContiguous) {
              a3D = operands[0].reshape([numBatch, m, k]);
            } else {
              a3D = operands[0].transpose(permA).reshape([numBatch, m, k]);
            }

            final NDArray b3D;
            if (isIdentityB && operands[1].isContiguous) {
              b3D = operands[1].reshape([numBatch, k, n]);
            } else {
              b3D = operands[1].transpose(permB).reshape([numBatch, k, n]);
            }

            final res3D = matmul<DTypeTag>(a3D, b3D);

            final freeAShapes = freeA.map((id) => labelSizes[id]!);
            final freeBShapes = freeB.map((id) => labelSizes[id]!);
            final targetUnpermutedShape = [
              ...batchShape,
              ...freeAShapes,
              ...freeBShapes,
            ];
            final resBatch = listEquals(res3D.shape, targetUnpermutedShape)
                ? res3D
                : res3D.reshape(targetUnpermutedShape);

            fast2OpResIds = [...batch, ...freeA, ...freeB];
            fast2OpRes = resBatch;
          }
        }

        if (fast2OpRes != null && fast2OpResIds != null) {
          var finalRes = fast2OpRes;
          if (!listEquals(fast2OpResIds, finalOutSub)) {
            final perm = finalOutSub
                .map((id) => fast2OpResIds!.indexOf(id))
                .toList();
            finalRes = fast2OpRes.transpose(perm);
          }

          final targetDType = finalRes.dtype;
          if (out != null) {
            if (!listEquals(out.shape, finalRes.shape) ||
                out.dtype != targetDType) {
              throw ArgumentError(
                "Provided out buffer has incompatible shape or dtype (expected shape ${finalRes.shape} and dtype $targetDType, got shape ${out.shape} and dtype ${out.dtype}).",
              );
            }
          }
          return _returnFromScope<T>(finalRes, operands, out: out);
        }
      }
    }

    if (operands.length > 2) {
      final currentOps = List<NDArray>.from(operands);
      final currentSubs = List<List<int>>.from(operandSubs);
      final toDispose = <NDArray>[];

      try {
        bool progress = true;
        while (currentOps.length > 2 && progress) {
          progress = false;
          int bestI = -1;
          int bestJ = -1;
          num minCost = double.infinity;
          List<int>? bestInterOut;

          for (var i = 0; i < currentOps.length; i++) {
            for (var j = i + 1; j < currentOps.length; j++) {
              final subI = currentSubs[i];
              final subJ = currentSubs[j];
              final union = <int>{...subI, ...subJ};

              final neededByOthers = <int>{...finalOutSub};
              for (var k = 0; k < currentOps.length; k++) {
                if (k != i && k != j) {
                  neededByOthers.addAll(currentSubs[k]);
                }
              }

              final interOut = union
                  .where((id) => neededByOthers.contains(id))
                  .toList();
              final contracted = union
                  .where((id) => !neededByOthers.contains(id))
                  .toList();

              if (contracted.isNotEmpty ||
                  union.length < subI.length + subJ.length) {
                num cost = 1;
                for (final id in union) {
                  cost *= labelSizes[id]!;
                }
                if (cost < minCost) {
                  minCost = cost;
                  bestI = i;
                  bestJ = j;
                  bestInterOut = interOut;
                }
              }
            }
          }

          if (bestI != -1 && bestJ != -1 && bestInterOut != null) {
            final opI = currentOps[bestI];
            final opJ = currentOps[bestJ];
            final subI = currentSubs[bestI];
            final subJ = currentSubs[bestJ];

            final specInter = EinsumSubscripts.fromIndices([
              subI,
              subJ,
            ], bestInterOut);

            final interRes = NDArray.unmanaged(
              () => einsum<DTypeTag>(specInter, [opI, opJ]),
            );
            toDispose.add(interRes);

            currentOps[bestI] = interRes;
            currentSubs[bestI] = bestInterOut;
            currentOps.removeAt(bestJ);
            currentSubs.removeAt(bestJ);
            progress = true;
          }
        }

        if (currentOps.length == 2) {
          final specFinal = EinsumSubscripts.fromIndices([
            currentSubs[0],
            currentSubs[1],
          ], finalOutSub);
          final finalRes = einsum<DTypeTag>(specFinal, [
            currentOps[0],
            currentOps[1],
          ], out: out);
          return _returnFromScope<T>(finalRes, operands, out: out);
        }
      } finally {
        for (final temp in toDispose) {
          if (!temp.isDisposed) {
            temp.dispose();
          }
        }
      }
    }

    final allIdsSet = <int>{};
    for (final id in finalOutSub) {
      allIdsSet.add(id);
    }

    for (final sub in operandSubs) {
      for (final id in sub) {
        allIdsSet.add(id);
      }
    }
    final allIds = allIdsSet.toList();

    final expandedOperands = <NDArray>[];
    for (var i = 0; i < operands.length; i++) {
      var op = _asTyped<DTypeTag>(operands[i]);
      var sub = operandSubs[i];

      final seenInOp = <int, int>{};
      for (var j = 0; j < sub.length; j++) {
        final id = sub[j];
        if (seenInOp.containsKey(id)) {
          final firstAx = seenInOp[id]!;
          final secondAx = j;
          op = _diagonalView(op, firstAx, secondAx);
          final newSub = List<int>.from(sub)..removeAt(secondAx);
          sub = newSub;
          j--;
        } else {
          seenInOp[id] = j;
        }
      }

      final expShape = <int>[];
      final perm = <int>[];

      for (final id in allIds) {
        if (sub.contains(id)) {
          expShape.add(labelSizes[id]!);
          perm.add(sub.indexOf(id));
        } else {
          expShape.add(1);
        }
      }

      final opTransposed = op.transpose(perm);
      final opExpanded = opTransposed.reshape(expShape);
      expandedOperands.add(opExpanded);
    }

    NDArray combined = expandedOperands[0];
    for (var i = 1; i < expandedOperands.length; i++) {
      combined = multiply<DTypeTag>(combined, expandedOperands[i]);
    }

    for (var j = allIds.length - 1; j >= 0; j--) {
      final id = allIds[j];
      if (!finalOutSub.contains(id)) {
        combined = sumAs<DTypeTag, DTypeTag>(combined, combined.dtype, axis: j);
      }
    }

    final targetDType = combined.dtype;
    if (out != null) {
      if (!listEquals(out.shape, combined.shape) || out.dtype != targetDType) {
        throw ArgumentError(
          "Provided out buffer has incompatible shape or dtype (expected shape ${combined.shape} and dtype $targetDType, got shape ${out.shape} and dtype ${out.dtype}).",
        );
      }
    }
    return _returnFromScope<T>(combined, operands, out: out);
  });
}

/// Computes the inner product of two arrays.
///
/// Ordinary inner product of vectors for 1-D arrays (without complex conjugation),
/// in higher dimensions a sum product over the last axes.
///
/// **Preconditions:**
/// - Neither [a] nor [b] may be disposed.
/// - If both [a] and [b] have rank $\ge 1$, the size of their last dimension must match: `a.shape.last == b.shape.last`.
/// - If [out] is provided, it must not be disposed, and its shape and dtype must match the expected result.
///
/// **Throws:**
/// - [StateError] if [a], [b], or [out] is disposed.
/// - [ArgumentError] if the last dimensions of [a] and [b] do not match (for rank $\ge 1$).
/// - [ArgumentError] if [out] has incompatible shape or dtype.
///
/// **Performance considerations:**
/// - Internally delegates to [matmul] with reshaped/transposed views.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
/// final b = NDArray.fromList([0.0, 1.0, 0.0], [3], DType.float64);
/// final res = inner(a, b);
/// print(res.scalar); // 2.0
/// ```
///
/// Reference: [NumPy inner](https://numpy.org/doc/stable/reference/generated/numpy.inner.html)
NDArray<T> inner<T extends DTypeTag>(
  NDArray<T> a,
  NDArray<T> b, {
  NDArray<T>? out,
}) {
  if (a.isDisposed || b.isDisposed) {
    throw StateError("Cannot execute inner() on a disposed array.");
  }
  if (out != null && out.isDisposed) {
    throw StateError("Cannot write inner result to a disposed output array.");
  }
  if (a.dtype != b.dtype) {
    throw ArgumentError.value(
      b.dtype,
      'b',
      'Must have the same dtype as a (${a.dtype})',
    );
  }

  if (a.rank == 0 || b.rank == 0) {
    return NDArray.scope(() {
      final res = multiply<T>(a, b, out: out);
      return _returnFromScope<T>(res, [a, b], out: out);
    });
  }

  if (a.shape.last != b.shape.last) {
    throw ArgumentError(
      "Dimension mismatch for inner: last dimension of a (${a.shape.last}) != last dimension of b (${b.shape.last}).",
    );
  }

  return tensordot<T>(
    a,
    b,
    axes: TensordotAxes.pair(a.rank - 1, b.rank - 1),
    out: out,
  );
}

/// Return the dot product of two vectors, flattening higher-dimensional arrays.
///
/// The [vdot] function handles complex numbers differently than `dot`: if the first argument
/// [a] is complex, it is replaced by its complex conjugate before the dot product is computed.
/// Higher-dimensional arrays are flattened to 1-D vectors before computing the dot product.
///
/// **Preconditions:**
/// - Neither [a] nor [b] may be disposed.
/// - The total number of elements in [a] and [b] (`a.size` and `b.size`) must be equal.
/// - If [out] is provided, it must not be disposed, and its shape must be `[]` (0-D scalar) with matching dtype.
///
/// **Throws:**
/// - [StateError] if [a], [b], or [out] is disposed.
/// - [ArgumentError] if `a.size != b.size`.
/// - [ArgumentError] if [out] has incompatible shape or dtype.
///
/// **Performance considerations:**
/// - Flattening is zero-copy when operands are contiguous.
/// - Delegates to BLAS dot product kernels via [matmul].
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([Complex(1, 2), Complex(3, 4)], [2], DType.complex128);
/// final b = NDArray.fromList([Complex(5, 6), Complex(7, 8)], [2], DType.complex128);
/// final res = vdot(a, b);
/// // (1 - 2i)*(5 + 6i) + (3 - 4i)*(7 + 8i) = (17 - 4i) + (53 - 4i) = 70 - 8i
/// print(res.scalar); // 70.0 - 8.0i
/// ```
///
/// Reference: [NumPy vdot](https://numpy.org/doc/stable/reference/generated/numpy.vdot.html)
NDArray<T> vdot<T extends DTypeTag>(
  NDArray<T> a,
  NDArray<T> b, {
  NDArray<T>? out,
}) {
  if (a.isDisposed || b.isDisposed) {
    throw StateError("Cannot execute vdot() on a disposed array.");
  }
  if (out != null && out.isDisposed) {
    throw StateError("Cannot write vdot result to a disposed output array.");
  }
  if (a.dtype != b.dtype) {
    throw ArgumentError.value(
      b.dtype,
      'b',
      'Must have the same dtype as a (${a.dtype})',
    );
  }
  if (a.size != b.size) {
    throw ArgumentError(
      "Cannot compute vdot: operands must have the same total number of elements (${a.size} != ${b.size}).",
    );
  }

  return NDArray.scope(() {
    final flatA = a.reshape([a.size]);
    final flatB = b.reshape([b.size]);
    final conjA = a.dtype.isComplex ? conjugate(flatA) : flatA;
    final res = matmul<T>(conjA, flatB, out: out);
    return _returnFromScope<T>(res, [a, b], out: out);
  });
}

/// Computes the Kronecker product of two arrays.
///
/// The Kronecker product is a composite tensor operation formed by multiplying every element
/// of array [a] by the entire array [b]. If [a] has shape $(m, n)$ and [b] has shape $(p, q)$,
/// the result has shape $(mp, nq)$.
///
/// For arrays with different numbers of dimensions, 1s are prepended to the shape of the
/// array with fewer dimensions until both shapes have equal rank.
///
/// **Preconditions:**
/// - It is an error if [a], [b], or [out] is disposed.
/// - It is an error if [out] has incompatible shape or dtype.
///
/// **Performance considerations:**
/// - 2D matrix Kronecker product uses dedicated AVX2 SIMD vectorization kernels for optimal row scaling throughput.
/// - Higher-dimensional Kronecker products leverage odometer-based slice sweeps with vectorized 2D inner tiles.
///
/// **Example:**
/// {@example /example/linalg_advanced_example.dart lang=dart}
///
/// Reference: [NumPy kron](https://numpy.org/doc/stable/reference/generated/numpy.kron.html)
NDArray<T> kron<T extends DTypeTag>(
  NDArray<T> a,
  NDArray<T> b, {
  NDArray<T>? out,
}) {
  if (a.isDisposed || b.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute kron() on a disposed array.');
  }
  if (a.dtype != b.dtype) {
    throw ArgumentError.value(
      b.dtype,
      'b',
      'Must have the same dtype as a (${a.dtype})',
    );
  }

  final rankA = a.rank;
  final rankB = b.rank;
  final maxRank = math.max(rankA, rankB);

  final paddedShapeA = List<int>.filled(maxRank, 1);
  for (var i = 0; i < rankA; i++) {
    paddedShapeA[maxRank - rankA + i] = a.shape[i];
  }

  final paddedShapeB = List<int>.filled(maxRank, 1);
  for (var i = 0; i < rankB; i++) {
    paddedShapeB[maxRank - rankB + i] = b.shape[i];
  }

  final expectedShape = List<int>.filled(maxRank, 0);
  for (var i = 0; i < maxRank; i++) {
    expectedShape[i] = paddedShapeA[i] * paddedShapeB[i];
  }

  final targetDType = a.dtype;
  if (out != null &&
      (!listEquals(out.shape, expectedShape) || out.dtype != targetDType)) {
    throw ArgumentError(
      'Provided out buffer has incompatible shape or dtype (expected shape $expectedShape and dtype $targetDType).',
    );
  }

  return NDArray.scope(() {
    var aCast = a.dtype == targetDType ? a : castNDArray(a, targetDType);
    if (aCast.strides.any((s) => s < 0)) {
      aCast = aCast.copy();
    }

    var bCast = b.dtype == targetDType ? b : castNDArray(b, targetDType);
    if (bCast.strides.any((s) => s < 0)) {
      bCast = bCast.copy();
    }

    final paddedStridesA = List<int>.filled(maxRank, 0);
    for (var i = 0; i < rankA; i++) {
      paddedStridesA[maxRank - rankA + i] = aCast.strides[i];
    }

    final paddedStridesB = List<int>.filled(maxRank, 0);
    for (var i = 0; i < rankB; i++) {
      paddedStridesB[maxRank - rankB + i] = bCast.strides[i];
    }

    final bool needTempOut =
        out != null &&
        (out.strides.any((s) => s < 0) ||
            _arraysOverlap(out, a) ||
            _arraysOverlap(out, b) ||
            _arraysOverlap(out, aCast) ||
            _arraysOverlap(out, bCast));
    final result = (out == null || needTempOut)
        ? _createRootArray(expectedShape, targetDType) as NDArray<T>
        : out;

    if (maxRank <= 2) {
      final m = maxRank >= 2 ? paddedShapeA[0] : 1;
      final n = maxRank >= 1 ? paddedShapeA[maxRank - 1] : 1;
      final p = maxRank >= 2 ? paddedShapeB[0] : 1;
      final q = maxRank >= 1 ? paddedShapeB[maxRank - 1] : 1;

      final strideA_0 = maxRank >= 2 ? paddedStridesA[0] : 0;
      final strideA_1 = maxRank >= 1 ? paddedStridesA[maxRank - 1] : 0;
      final strideB_0 = maxRank >= 2 ? paddedStridesB[0] : 0;
      final strideB_1 = maxRank >= 1 ? paddedStridesB[maxRank - 1] : 0;

      final strideRes_0 = maxRank >= 2 ? result.strides[0] : 0;
      final strideRes_1 = maxRank >= 1 ? result.strides[maxRank - 1] : 0;

      native_kron_2d(
        encodeDType(targetDType),
        aCast.pointer.cast(),
        strideA_0,
        strideA_1,
        m,
        n,
        bCast.pointer.cast(),
        strideB_0,
        strideB_1,
        p,
        q,
        result.pointer.cast(),
        strideRes_0,
        strideRes_1,
      );
    } else {
      final marker = ScratchArena.marker;
      try {
        final cStridesA = ScratchArena.allocate<ffi.Int>(
          maxRank * ffi.sizeOf<ffi.Int>(),
        );
        final cShapeA = ScratchArena.allocate<ffi.Int>(
          maxRank * ffi.sizeOf<ffi.Int>(),
        );
        final cStridesB = ScratchArena.allocate<ffi.Int>(
          maxRank * ffi.sizeOf<ffi.Int>(),
        );
        final cShapeB = ScratchArena.allocate<ffi.Int>(
          maxRank * ffi.sizeOf<ffi.Int>(),
        );
        final cStridesRes = ScratchArena.allocate<ffi.Int>(
          maxRank * ffi.sizeOf<ffi.Int>(),
        );
        final cShapeRes = ScratchArena.allocate<ffi.Int>(
          maxRank * ffi.sizeOf<ffi.Int>(),
        );

        for (var i = 0; i < maxRank; i++) {
          cStridesA[i] = paddedStridesA[i];
          cShapeA[i] = paddedShapeA[i];
          cStridesB[i] = paddedStridesB[i];
          cShapeB[i] = paddedShapeB[i];
          cStridesRes[i] = result.strides[i];
          cShapeRes[i] = result.shape[i];
        }

        native_kron_nd(
          encodeDType(targetDType),
          aCast.pointer.cast(),
          cStridesA,
          cShapeA,
          bCast.pointer.cast(),
          cStridesB,
          cShapeB,
          result.pointer.cast(),
          cStridesRes,
          cShapeRes,
          maxRank,
        );
      } finally {
        ScratchArena.reset(marker);
      }
    }

    if (needTempOut) {
      result.copy(out: out);
      return out;
    }

    if (out == null) {
      result.detachToParentScope();
    }
    return result;
  });
}
