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
import 'dart:math' as math;
import '../dtype.dart';
import '../gpu_array.dart';
import '../exceptions.dart';
import '../backend/compute_engine.dart';
import '../operations/manipulation.dart' as manip;

/// Evaluates the Einstein summation convention on the operands.
GpuArray<T> einsum<T extends DTypeTag>(
  String subscripts,
  List<GpuArray> operands,
) {
  if (operands.isEmpty) {
    throw ArgumentError('einsum requires at least one operand.');
  }

  final cleanSub = subscripts.replaceAll(' ', '');
  final parts = cleanSub.split('->');
  final inPart = parts[0];
  final inputTerms = inPart.split(',');

  if (inputTerms.length != operands.length) {
    throw ArgumentError(
      'Number of einsum subscript input terms (${inputTerms.length}) does not match operands count (${operands.length}).',
    );
  }

  // Determine output indices
  String outTerm;
  if (parts.length > 1) {
    outTerm = parts[1];
  } else {
    // Implicit mode: indices appearing exactly once sorted alphabetically
    final counts = <String, int>{};
    for (final term in inputTerms) {
      for (var i = 0; i < term.length; i++) {
        final ch = term[i];
        counts[ch] = (counts[ch] ?? 0) + 1;
      }
    }
    final singleList =
        counts.entries.where((e) => e.value == 1).map((e) => e.key).toList()
          ..sort();
    outTerm = singleList.join();
  }

  // Map each unique index character to its dimension length
  final indexDims = <String, int>{};
  for (var opIdx = 0; opIdx < operands.length; opIdx++) {
    final term = inputTerms[opIdx];
    final op = operands[opIdx];
    if (term.length != op.rank) {
      throw ArgumentError(
        'Subscript term "$term" length (${term.length}) does not match operand rank (${op.rank}).',
      );
    }
    for (var i = 0; i < term.length; i++) {
      final ch = term[i];
      final dim = op.shape[i];
      if (indexDims.containsKey(ch) && indexDims[ch] != dim) {
        throw ArgumentError(
          'Dimension mismatch for index "$ch": expected ${indexDims[ch]} but got $dim.',
        );
      }
      indexDims[ch] = dim;
    }
  }

  if (parts.length > 1) {
    for (final ch in outTerm.split('')) {
      if (!indexDims.containsKey(ch)) {
        throw ArgumentError(
          'Output subscript "$ch" not found in input subscripts.',
        );
      }
    }
  }

  // Determine promoted dtype
  var outDType = operands[0].dtype;
  for (var i = 1; i < operands.length; i++) {
    outDType = GpuArray.promoteDTypes(outDType, operands[i].dtype);
  }

  final outShape = outTerm.split('').map((ch) => indexDims[ch]!).toList();
  final result = GpuArray<T>.zeros(
    outShape,
    outDType as DType<T>,
    device: operands[0].device,
  );

  // All unique indices
  final allIndices = indexDims.keys.toList();
  final allSizes = allIndices.map((ch) => indexDims[ch]!).toList();
  final totalIterations = allSizes.fold(1, (a, b) => a * b);

  final currIndexVals = <String, int>{};
  final coords = List<int>.filled(allIndices.length, 0);

  final outIndexList = outTerm.split('');
  final outStrides = ShapeUtils.computeCStrides(outShape);

  // Flattened data views
  final opFlats = operands.map((op) => op.toNDArray()).toList();
  final opDataList = opFlats
      .map((nd) => nd.toList().cast<num>().map((e) => e.toDouble()).toList())
      .toList();
  final opStridesList = operands
      .map((op) => ShapeUtils.computeCStrides(op.shape))
      .toList();
  final outData = List<double>.filled(ShapeUtils.computeSize(outShape), 0.0);

  for (var iter = 0; iter < totalIterations; iter++) {
    for (var i = 0; i < allIndices.length; i++) {
      currIndexVals[allIndices[i]] = coords[i];
    }

    var prod = 1.0;
    for (var opIdx = 0; opIdx < operands.length; opIdx++) {
      final term = inputTerms[opIdx];
      final opStrides = opStridesList[opIdx];
      var elemIdx = 0;
      for (var d = 0; d < term.length; d++) {
        elemIdx += currIndexVals[term[d]]! * opStrides[d];
      }
      prod *= opDataList[opIdx][elemIdx];
    }

    var outElemIdx = 0;
    for (var d = 0; d < outIndexList.length; d++) {
      outElemIdx += currIndexVals[outIndexList[d]]! * outStrides[d];
    }
    outData[outElemIdx] += prod;

    for (var d = allIndices.length - 1; d >= 0; d--) {
      coords[d]++;
      if (coords[d] < allSizes[d]) break;
      coords[d] = 0;
    }
  }

  for (final nd in opFlats) {
    nd.dispose();
  }

  for (var i = 0; i < outData.length; i++) {
    ComputeEngine.writeAny(result.buffer, outDType, i, outData[i]);
  }

  return result;
}

/// Compute tensor dot product along specified axes.
GpuArray<T> tensordot<T extends DTypeTag>(
  GpuArray a,
  GpuArray b, {
  dynamic axes = 2,
}) {
  List<int> aAxes;
  List<int> bAxes;

  if (axes is int) {
    aAxes = List.generate(axes, (i) => a.rank - axes + i);
    bAxes = List.generate(axes, (i) => i);
  } else if (axes is List &&
      axes.length == 2 &&
      axes[0] is List &&
      axes[1] is List) {
    aAxes = (axes[0] as List).cast<int>();
    bAxes = (axes[1] as List).cast<int>();
  } else {
    throw ArgumentError('Invalid axes parameter for tensordot: $axes');
  }

  final normAAxes = aAxes.map((ax) => ax < 0 ? ax + a.rank : ax).toList();
  final normBAxes = bAxes.map((ax) => ax < 0 ? ax + b.rank : ax).toList();

  final aFree = <int>[];
  for (var i = 0; i < a.rank; i++) {
    if (!normAAxes.contains(i)) aFree.add(i);
  }
  final bFree = <int>[];
  for (var i = 0; i < b.rank; i++) {
    if (!normBAxes.contains(i)) bFree.add(i);
  }

  final aPerm = [...aFree, ...normAAxes];
  final bPerm = [...normBAxes, ...bFree];

  final aTransposed = a.transpose(aPerm);
  final bTransposed = b.transpose(bPerm);

  final aFreeSize = aFree.fold(1, (res, i) => res * a.shape[i]);
  final kSize = normAAxes.fold(1, (res, i) => res * a.shape[i]);
  final bFreeSize = bFree.fold(1, (res, i) => res * b.shape[i]);

  final aMat = aTransposed.reshape([aFreeSize, kSize]);
  final bMat = bTransposed.reshape([kSize, bFreeSize]);

  final cMat = aMat.matmul(bMat);

  final outShape = [
    ...aFree.map((i) => a.shape[i]),
    ...bFree.map((i) => b.shape[i]),
  ];

  return cMat.reshape(outShape) as GpuArray<T>;
}

/// Computes the Kronecker product of two arrays.
GpuArray<T> kron<T extends DTypeTag>(GpuArray a, GpuArray b) {
  final rank = math.max(a.rank, b.rank);
  final padRankA = rank - a.rank;
  final padRankB = rank - b.rank;

  final shapeA = List<int>.filled(padRankA, 1, growable: true)..addAll(a.shape);
  final shapeB = List<int>.filled(padRankB, 1, growable: true)..addAll(b.shape);

  final outShape = List.generate(rank, (i) => shapeA[i] * shapeB[i]);
  final outDType = GpuArray.promoteDTypes(a.dtype, b.dtype) as DType<T>;
  final result = GpuArray<T>.empty(outShape, outDType, device: a.device);

  final aArr = a.reshape(shapeA);
  final bArr = b.reshape(shapeB);

  final totalA = ShapeUtils.computeSize(shapeA);
  final totalB = ShapeUtils.computeSize(shapeB);

  final aFlat = aArr.toNDArray();
  final aData = aFlat.toList().cast<num>().map((e) => e.toDouble()).toList();
  final bFlat = bArr.toNDArray();
  final bData = bFlat.toList().cast<num>().map((e) => e.toDouble()).toList();

  final outData = List<double>.filled(ShapeUtils.computeSize(outShape), 0.0);
  final coordsA = List<int>.filled(rank, 0);
  final coordsB = List<int>.filled(rank, 0);

  for (var i = 0; i < totalA; i++) {
    final aVal = aData[i];

    for (var j = 0; j < totalB; j++) {
      final bVal = bData[j];
      var outIdx = 0;
      var stride = 1;
      for (var d = rank - 1; d >= 0; d--) {
        final outCoord = coordsA[d] * shapeB[d] + coordsB[d];
        outIdx += outCoord * stride;
        stride *= outShape[d];
      }

      outData[outIdx] = aVal * bVal;

      for (var d = rank - 1; d >= 0; d--) {
        coordsB[d]++;
        if (coordsB[d] < shapeB[d]) break;
        coordsB[d] = 0;
      }
    }

    for (var d = rank - 1; d >= 0; d--) {
      coordsA[d]++;
      if (coordsA[d] < shapeA[d]) break;
      coordsA[d] = 0;
    }
  }

  aFlat.dispose();
  bFlat.dispose();

  for (var i = 0; i < outData.length; i++) {
    ComputeEngine.writeAny(result.buffer, outDType, i, outData[i]);
  }

  return result;
}

/// Computes inner product of two arrays.
GpuArray<T> inner<T extends DTypeTag>(GpuArray a, GpuArray b) {
  if (a.rank == 1 && b.rank == 1) {
    return a.matmul(b) as GpuArray<T>;
  }
  return tensordot<T>(
    a,
    b,
    axes: [
      [a.rank - 1],
      [b.rank - 1],
    ],
  );
}

/// Computes the outer product of two 1D vectors.
GpuArray<T> outer<T extends DTypeTag>(GpuArray a, GpuArray b) {
  final aFlat = a.flatten();
  final bFlat = b.flatten();
  final aCol = aFlat.reshape([aFlat.shape[0], 1]);
  final bRow = bFlat.reshape([1, bFlat.shape[0]]);
  return aCol.matmul(bRow) as GpuArray<T>;
}

/// Computes vector cross product of two 3D vectors or batched 3D vectors.
GpuArray<T> cross<T extends DTypeTag>(
  GpuArray a,
  GpuArray b, {
  int? axisa,
  int? axisb,
  int? axisc,
  int? axis,
}) {
  if (a.rank < 1 || b.rank < 1) {
    throw ArgumentError('cross() requires arrays of at least 1 dimension.');
  }

  final axA = axis ?? axisa ?? -1;
  final axB = axis ?? axisb ?? -1;
  final axC = axis ?? axisc ?? -1;

  final normAxA = axA < 0 ? axA + a.rank : axA;
  final normAxB = axB < 0 ? axB + b.rank : axB;

  if (normAxA < 0 || normAxA >= a.rank) {
    throw GpuAxisOutOfBoundsException(axA, a.rank);
  }
  if (normAxB < 0 || normAxB >= b.rank) {
    throw GpuAxisOutOfBoundsException(axB, b.rank);
  }

  if (a.shape[normAxA] != 3 || b.shape[normAxB] != 3) {
    throw ArgumentError(
      'cross() requires vectors of length 3 along target axis.',
    );
  }

  // Move the vector axes to the trailing dimension (-1)
  final aMoved = (normAxA == a.rank - 1) ? a : manip.moveaxis(a, normAxA, -1);
  final bMoved = (normAxB == b.rank - 1) ? b : manip.moveaxis(b, normAxB, -1);

  final batchShapeA = aMoved.shape.sublist(0, aMoved.rank - 1);
  final batchShapeB = bMoved.shape.sublist(0, bMoved.rank - 1);
  final batchShape = ShapeUtils.broadcastShapes(batchShapeA, batchShapeB);

  final targetShape = [...batchShape, 3];
  final aBroad = manip.broadcast_to(aMoved, targetShape);
  final bBroad = manip.broadcast_to(bMoved, targetShape);

  final outDType = GpuArray.promoteDTypes(a.dtype, b.dtype) as DType<T>;
  final interResult = GpuArray<T>.empty(
    targetShape,
    outDType,
    device: a.device,
  );

  final aFlat = aBroad.toNDArray();
  final bFlat = bBroad.toNDArray();
  final aData = aFlat.toList().cast<num>().map((e) => e.toDouble()).toList();
  final bData = bFlat.toList().cast<num>().map((e) => e.toDouble()).toList();
  final outData = List<double>.filled(ShapeUtils.computeSize(targetShape), 0.0);

  final total = ShapeUtils.computeSize(targetShape) ~/ 3;
  for (var i = 0; i < total; i++) {
    final a0 = aData[i * 3];
    final a1 = aData[i * 3 + 1];
    final a2 = aData[i * 3 + 2];

    final b0 = bData[i * 3];
    final b1 = bData[i * 3 + 1];
    final b2 = bData[i * 3 + 2];

    outData[i * 3] = a1 * b2 - a2 * b1;
    outData[i * 3 + 1] = a2 * b0 - a0 * b2;
    outData[i * 3 + 2] = a0 * b1 - a1 * b0;
  }

  aFlat.dispose();
  bFlat.dispose();

  for (var i = 0; i < outData.length; i++) {
    ComputeEngine.writeAny(interResult.buffer, outDType, i, outData[i]);
  }

  final finalRank = targetShape.length;
  final normAxC = axC < 0 ? axC + finalRank : axC;
  if (normAxC < 0 || normAxC >= finalRank) {
    throw GpuAxisOutOfBoundsException(axC, finalRank);
  }

  if (normAxC == finalRank - 1) {
    return interResult;
  }
  return manip.moveaxis(interResult, -1, normAxC);
}

/// Efficiently chains matrix multiplications with optimized parenthesization.
GpuArray<T> multi_dot<T extends DTypeTag>(List<GpuArray> arrays) {
  if (arrays.isEmpty) {
    throw ArgumentError('multi_dot requires at least one array.');
  }
  if (arrays.length == 1) return arrays[0] as GpuArray<T>;
  if (arrays.length == 2) {
    final a0 = arrays[0];
    final a1 = arrays[1];
    if (a0.rank == 1 && a1.rank == 1) {
      return a0.matmul<T>(a1);
    } else if (a0.rank == 1 && a1.rank == 2) {
      if (a0.shape[0] != a1.shape[0]) {
        throw GpuShapeMismatchException('multi_dot', a0.shape, a1.shape);
      }
      final r0 = a0.reshape([1, a0.shape[0]]);
      final res = r0.matmul(a1);
      return res.reshape([a1.shape[1]]) as GpuArray<T>;
    } else if (a0.rank == 2 && a1.rank == 1) {
      if (a0.shape[1] != a1.shape[0]) {
        throw GpuShapeMismatchException('multi_dot', a0.shape, a1.shape);
      }
      final r1 = a1.reshape([a1.shape[0], 1]);
      final res = a0.matmul(r1);
      return res.reshape([a0.shape[0]]) as GpuArray<T>;
    } else if (a0.rank == 2 && a1.rank == 2) {
      return a0.matmul<T>(a1);
    } else {
      throw ArgumentError(
        'multi_dot only supports 1D vectors and 2D matrices.',
      );
    }
  }

  final isFirst1D = arrays[0].rank == 1;
  final isLast1D = arrays.last.rank == 1;

  if (arrays[0].rank != 1 && arrays[0].rank != 2) {
    throw ArgumentError('First array in multi_dot must be 1D or 2D.');
  }
  if (arrays.last.rank != 1 && arrays.last.rank != 2) {
    throw ArgumentError('Last array in multi_dot must be 1D or 2D.');
  }
  for (var i = 1; i < arrays.length - 1; i++) {
    if (arrays[i].rank != 2) {
      throw ArgumentError(
        'Intermediate arrays in multi_dot must be 2D matrices.',
      );
    }
  }

  final ops = <GpuArray>[];
  ops.add(isFirst1D ? arrays[0].reshape([1, arrays[0].shape[0]]) : arrays[0]);
  for (var i = 1; i < arrays.length - 1; i++) {
    ops.add(arrays[i]);
  }
  ops.add(
    isLast1D ? arrays.last.reshape([arrays.last.shape[0], 1]) : arrays.last,
  );

  final n = ops.length;
  final p = List<int>.filled(n + 1, 0);
  p[0] = ops[0].shape[0];
  for (var i = 0; i < n; i++) {
    if (ops[i].shape[0] != p[i]) {
      throw GpuShapeMismatchException(
        'multi_dot',
        ops[i - 1].shape,
        ops[i].shape,
      );
    }
    p[i + 1] = ops[i].shape[1];
  }

  final m = List.generate(n, (_) => List.filled(n, 0));
  final s = List.generate(n, (_) => List.filled(n, 0));

  for (var l = 2; l <= n; l++) {
    for (var i = 0; i <= n - l; i++) {
      final j = i + l - 1;
      m[i][j] = 1 << 60;
      for (var k = i; k < j; k++) {
        final q = m[i][k] + m[k + 1][j] + p[i] * p[k + 1] * p[j + 1];
        if (q < m[i][j]) {
          m[i][j] = q;
          s[i][j] = k;
        }
      }
    }
  }

  GpuArray<T> multiplyChain(int i, int j) {
    if (i == j) return ops[i] as GpuArray<T>;
    final k = s[i][j];
    final left = multiplyChain(i, k);
    final right = multiplyChain(k + 1, j);
    return left.matmul<T>(right);
  }

  final full2DResult = multiplyChain(0, n - 1);

  if (isFirst1D && isLast1D) {
    return full2DResult.reshape([]);
  } else if (isFirst1D) {
    return full2DResult.reshape([p[n]]);
  } else if (isLast1D) {
    return full2DResult.reshape([p[0]]);
  } else {
    return full2DResult;
  }
}
