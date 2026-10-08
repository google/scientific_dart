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
import '../ndarray.dart';
import 'dart:ffi' as ffi;
import '../ndarray_bindings.dart';
import '../nditer.dart';
import '../scratch_arena.dart';

// Standalone operational relative cross-imports
import 'math.dart';
import 'sorting.dart';
import 'broadcasting.dart';
import 'helpers.dart';
import 'stats.dart';

/// Supported sorting algorithms.
///
/// {@example /example/sorting_searching_example.dart lang=dart}
enum SortKind {
  /// Unstable, fast QuickSort.
  quicksort,

  /// Stable, mergesort-based TimSort.
  mergesort,

  /// Unstable, HeapSort.
  heapsort,

  /// Alias for mergesort (guaranteed stable).
  stable,
}

/// Search boundary behavior selector for binary search insertion points in [searchsorted].
///
/// This determines the index returned when a query value matches existing elements
/// in the sorted target array:
/// - [SearchSide.left] returns the index of the **first** suitable location found (the leftmost match).
/// - [SearchSide.right] returns the index of the **last** suitable location found (the rightmost match).
///
/// **Example:**
/// {@example /example/shaping_example.dart lang=dart}
enum SearchSide {
  /// Finds the first suitable index to insert to maintain sorted order.
  left,

  /// Finds the last suitable index to insert to maintain sorted order.
  right,
}

/// Returns [numSamples] evenly spaced samples, calculated over the interval `[start, stop]`.
///
/// The endpoint of the interval can optionally be excluded.
/// Supports [Complex] bounds for path generation in the complex plane.
/// If [endpoint] is true, `stop` is the last sample. Otherwise, it is not included.
///
/// **Preconditions:**
/// - [numSamples] must be non-negative.
///
/// - It is an error if [numSamples] is negative.
///
/// **Memory Ownership & Lifetime:**
/// - Allocates a new array on the unmanaged C heap. The caller takes full ownership of this memory and must explicitly call [dispose] to prevent native leaks, unless executing inside a managed [NDArray.scope].
///
/// **Example:**
/// {@example /example/shaping_example.dart lang=dart}
NDArray<T> linspace<T extends DTypeTag>(
  Object? start,
  Object? stop,
  int numSamples, {
  bool endpoint = true,
  required DType<T> dtype,
  NDArray<T>? out,
}) {
  return linspaceInternal<T>(
    start,
    stop,
    numSamples,
    endpoint: endpoint,
    dtype: dtype,
    out: out,
  ).samples;
}

/// Computes evenly spaced samples over a specified interval and returns them along with the step size.
///
/// Returns a record `(samples, step)`.
/// If [endpoint] is true, `stop` is the last sample. Otherwise, it is not included.
///
/// **Preconditions:**
/// - [numSamples] must be non-negative.
///
/// - It is an error if [numSamples] is negative.
///
/// **Memory Ownership & Lifetime:**
/// - Allocates a new array on the unmanaged C heap. The caller takes full ownership of this memory and must explicitly call [dispose] to prevent native leaks, unless executing inside a managed [NDArray.scope].
({NDArray<T> samples, Object step}) linspaceWithStep<T extends DTypeTag>(
  Object? start,
  Object? stop,
  int numSamples, {
  bool endpoint = true,
  required DType<T> dtype,
  NDArray<T>? out,
}) {
  final res = linspaceInternal<T>(
    start,
    stop,
    numSamples,
    endpoint: endpoint,
    dtype: dtype,
    out: out,
  );
  return (samples: res.samples, step: res.step as Object);
}

/// Generalized [linspace] that supports broadcasting when [start] or [stop] are [NDArray]s.
///
/// Returns an array of shape `(..., numSamples, ...)` depending on the [axis].
///
/// **Example:**
/// {@example /example/shaping_example.dart lang=dart}
///
/// **Preconditions:**
/// - [start] and [stop] must not be disposed.
/// - [numSamples] must be non-negative.
///
/// - It is an error if [start] or [stop] is disposed.
/// - It is an error if [numSamples] is negative.
/// - It is an error if [axis] is out of bounds.
///
/// **Memory Ownership & Lifetime:**
/// - Allocates a new array on the unmanaged C heap. The caller takes full ownership of this memory and must explicitly call [dispose] to prevent native leaks, unless executing inside a managed [NDArray.scope].
NDArray<T> linspaceGrid<T extends DTypeTag>(
  NDArray<T> start,
  NDArray<T> stop,
  int numSamples, {
  bool endpoint = true,
  int axis = 0,
  DType<T>? dtype,
  NDArray<T>? out,
}) {
  if (start.isDisposed || stop.isDisposed) {
    throw StateError('Cannot execute linspaceGrid() on a disposed array.');
  }
  final res = _linspaceGridInternal<T>(
    start,
    stop,
    numSamples,
    endpoint: endpoint,
    axis: axis,
    dtype: dtype,
    out: out,
  );
  res.step.dispose();
  return res.samples;
}

/// Similar to [linspaceGrid], but also returns the calculated step size as an [NDArray].
///
/// Returns a Record `(samples, step)`.
///
/// **Example:**
/// {@example /example/shaping_example.dart lang=dart}
///
/// **Preconditions:**
/// - [start] and [stop] must not be disposed.
/// - [numSamples] must be non-negative.
///
/// - It is an error if [start] or [stop] is disposed.
/// - It is an error if [numSamples] is negative.
/// - It is an error if [axis] is out of bounds.
///
/// **Memory Ownership & Lifetime:**
/// - Allocates new arrays on the unmanaged C heap. The caller takes full ownership of this memory and must explicitly call [dispose] to prevent native leaks, unless executing inside a managed [NDArray.scope].
({NDArray<T> samples, NDArray<T> step})
linspaceGridWithStep<T extends DTypeTag>(
  NDArray<T> start,
  NDArray<T> stop,
  int numSamples, {
  bool endpoint = true,
  int axis = 0,
  DType<T>? dtype,
  NDArray<T>? out,
}) {
  if (start.isDisposed || stop.isDisposed) {
    throw StateError(
      'Cannot execute linspaceGridWithStep() on a disposed array.',
    );
  }
  return _linspaceGridInternal<T>(
    start,
    stop,
    numSamples,
    endpoint: endpoint,
    axis: axis,
    dtype: dtype,
    out: out,
  );
}

({NDArray<T> samples, NDArray<T> step})
_linspaceGridInternal<T extends DTypeTag>(
  NDArray<T> start,
  NDArray<T> stop,
  int numSamples, {
  bool endpoint = true,
  int axis = 0,
  DType<T>? dtype,
  NDArray<T>? out,
}) {
  if (numSamples < 0) {
    throw ArgumentError.value(numSamples, 'numSamples', 'Must be non-negative');
  }
  if (dtype == DType.boolean ||
      start.dtype == DType.boolean ||
      stop.dtype == DType.boolean) {
    throw UnsupportedError('linspaceGrid not supported for boolean arrays');
  }

  final resolvedDType =
      dtype ?? (resolveDType(start.dtype, stop.dtype) as DType<T>);

  return NDArray.scope(() {
    final startArr = toNDArray(start, resolvedDType);
    final stopArr = toNDArray(stop, resolvedDType);

    final commonShape = broadcastShapes(startArr.shape, stopArr.shape);
    final outRank = commonShape.length + 1;
    final actualAxis = axis < 0 ? outRank + axis : axis;
    if (actualAxis < 0 || actualAxis >= outRank) {
      throw RangeError.range(
        axis,
        -outRank,
        outRank - 1,
        'axis',
        'Must be within valid rank range',
      );
    }

    final resultShape = List<int>.from(commonShape);
    resultShape.insert(actualAxis, numSamples);

    if (out != null) {
      validateOutBuffer(out);
      if (!listEquals(out.shape, resultShape) || out.dtype != resolvedDType) {
        throw ArgumentError.value(
          out,
          'out',
          'Incompatible out buffer shape or dtype',
        );
      }
    }
    if (numSamples == 0) {
      final res = out ?? NDArray<T>.create(resultShape, resolvedDType);
      final step = NDArray<T>.create(commonShape, resolvedDType);
      final nanVal = (resolvedDType.isInteger || resolvedDType == DType.boolean)
          ? normalizeScalar(0, resolvedDType)
          : normalizeScalar(double.nan, resolvedDType);
      step.fill(nanVal);
      if (out == null) res.detachToParentScope();
      step.detachToParentScope();
      return (samples: res, step: step);
    }

    final startBroadcasted = broadcastTo(startArr, commonShape);
    final stopBroadcasted = broadcastTo(stopArr, commonShape);

    final stridesStart = List<int>.from(startBroadcasted.strides);
    stridesStart.insert(actualAxis, 0);

    final stridesStop = List<int>.from(stopBroadcasted.strides);
    stridesStop.insert(actualAxis, 0);

    final bool useTempOut =
        out != null && (sharesMemory(start, out) || sharesMemory(stop, out));
    final res = (out != null && !useTempOut)
        ? out
        : NDArray<T>.create(resultShape, resolvedDType);
    final stridesRes = res.strides;

    final step = NDArray<T>.create(commonShape, resolvedDType);
    final stridesStepOdo = List<int>.from(step.strides);
    stridesStepOdo.insert(actualAxis, 0);

    final rank = resultShape.length;
    final marker = ScratchArena.marker;
    try {
      final cShape = ScratchArena.copyInts(resultShape);
      final cStridesStart = ScratchArena.copyInts(stridesStart);
      final cStridesStop = ScratchArena.copyInts(stridesStop);
      final cStridesRes = ScratchArena.copyInts(stridesRes);
      final cStridesStep = ScratchArena.copyInts(stridesStepOdo);

      switch (resolvedDType) {
        case DType.float64:
          s_linspace_grid_double(
            startBroadcasted.pointer.cast<ffi.Double>(),
            cStridesStart,
            stopBroadcasted.pointer.cast<ffi.Double>(),
            cStridesStop,
            res.pointer.cast<ffi.Double>(),
            cStridesRes,
            step.pointer.cast<ffi.Double>(),
            cStridesStep,
            cShape,
            rank,
            actualAxis,
            numSamples,
            endpoint ? 1 : 0,
          );
        case DType.float32:
          s_linspace_grid_float(
            startBroadcasted.pointer.cast<ffi.Float>(),
            cStridesStart,
            stopBroadcasted.pointer.cast<ffi.Float>(),
            cStridesStop,
            res.pointer.cast<ffi.Float>(),
            cStridesRes,
            step.pointer.cast<ffi.Float>(),
            cStridesStep,
            cShape,
            rank,
            actualAxis,
            numSamples,
            endpoint ? 1 : 0,
          );
        case DType.complex128:
          s_linspace_grid_complex128(
            startBroadcasted.pointer.cast<cpx_t>(),
            cStridesStart,
            stopBroadcasted.pointer.cast<cpx_t>(),
            cStridesStop,
            res.pointer.cast<cpx_t>(),
            cStridesRes,
            step.pointer.cast<cpx_t>(),
            cStridesStep,
            cShape,
            rank,
            actualAxis,
            numSamples,
            endpoint ? 1 : 0,
          );
        case DType.complex64:
          s_linspace_grid_complex64(
            startBroadcasted.pointer.cast<cpx_f_t>(),
            cStridesStart,
            stopBroadcasted.pointer.cast<cpx_f_t>(),
            cStridesStop,
            res.pointer.cast<cpx_f_t>(),
            cStridesRes,
            step.pointer.cast<cpx_f_t>(),
            cStridesStep,
            cShape,
            rank,
            actualAxis,
            numSamples,
            endpoint ? 1 : 0,
          );
        case DType.int64:
          s_linspace_grid_int64(
            startBroadcasted.pointer.cast<ffi.Int64>(),
            cStridesStart,
            stopBroadcasted.pointer.cast<ffi.Int64>(),
            cStridesStop,
            res.pointer.cast<ffi.Int64>(),
            cStridesRes,
            step.pointer.cast<ffi.Int64>(),
            cStridesStep,
            cShape,
            rank,
            actualAxis,
            numSamples,
            endpoint ? 1 : 0,
          );
        case DType.int32:
          s_linspace_grid_int32(
            startBroadcasted.pointer.cast<ffi.Int32>(),
            cStridesStart,
            stopBroadcasted.pointer.cast<ffi.Int32>(),
            cStridesStop,
            res.pointer.cast<ffi.Int32>(),
            cStridesRes,
            step.pointer.cast<ffi.Int32>(),
            cStridesStep,
            cShape,
            rank,
            actualAxis,
            numSamples,
            endpoint ? 1 : 0,
          );
        case DType.int16:
          s_linspace_grid_int16(
            startBroadcasted.pointer.cast<ffi.Int16>(),
            cStridesStart,
            stopBroadcasted.pointer.cast<ffi.Int16>(),
            cStridesStop,
            res.pointer.cast<ffi.Int16>(),
            cStridesRes,
            step.pointer.cast<ffi.Int16>(),
            cStridesStep,
            cShape,
            rank,
            actualAxis,
            numSamples,
            endpoint ? 1 : 0,
          );
        case DType.uint8:
          s_linspace_grid_uint8(
            startBroadcasted.pointer.cast<ffi.Uint8>(),
            cStridesStart,
            stopBroadcasted.pointer.cast<ffi.Uint8>(),
            cStridesStop,
            res.pointer.cast<ffi.Uint8>(),
            cStridesRes,
            step.pointer.cast<ffi.Uint8>(),
            cStridesStep,
            cShape,
            rank,
            actualAxis,
            numSamples,
            endpoint ? 1 : 0,
          );
        case DType.float16:
        case DType.bfloat16:
          final startF64 = castNDArray<Float64>(
            startBroadcasted,
            DType.float64,
          );
          final stopF64 = castNDArray<Float64>(stopBroadcasted, DType.float64);
          final resF64 = NDArray<Float64>.create(resultShape, DType.float64);
          final stepF64 = NDArray<Float64>.create(commonShape, DType.float64);
          final stridesStartF64 = List<int>.from(startF64.strides)
            ..insert(actualAxis, 0);
          final stridesStopF64 = List<int>.from(stopF64.strides)
            ..insert(actualAxis, 0);
          final stridesStepF64 = List<int>.from(stepF64.strides)
            ..insert(actualAxis, 0);
          final cStridesStartF64 = ScratchArena.copyInts(stridesStartF64);
          final cStridesStopF64 = ScratchArena.copyInts(stridesStopF64);
          final cStridesResF64 = ScratchArena.copyInts(resF64.strides);
          final cStridesStepF64 = ScratchArena.copyInts(stridesStepF64);
          s_linspace_grid_double(
            startF64.pointer.cast<ffi.Double>(),
            cStridesStartF64,
            stopF64.pointer.cast<ffi.Double>(),
            cStridesStopF64,
            resF64.pointer.cast<ffi.Double>(),
            cStridesResF64,
            stepF64.pointer.cast<ffi.Double>(),
            cStridesStepF64,
            cShape,
            rank,
            actualAxis,
            numSamples,
            endpoint ? 1 : 0,
          );
          castNDArray<T>(resF64, resolvedDType).copy(out: res);
          castNDArray<T>(stepF64, resolvedDType).copy(out: step);
        case DType.int8:
        case DType.uint32:
        case DType.uint16:
          final startF64 = castNDArray<Float64>(
            startBroadcasted,
            DType.float64,
          );
          final stopF64 = castNDArray<Float64>(stopBroadcasted, DType.float64);
          final resF64 = NDArray<Float64>.create(resultShape, DType.float64);
          final stepF64 = NDArray<Float64>.create(commonShape, DType.float64);
          final stridesStartF64 = List<int>.from(startF64.strides)
            ..insert(actualAxis, 0);
          final stridesStopF64 = List<int>.from(stopF64.strides)
            ..insert(actualAxis, 0);
          final stridesStepF64 = List<int>.from(stepF64.strides)
            ..insert(actualAxis, 0);
          final cStridesStartF64 = ScratchArena.copyInts(stridesStartF64);
          final cStridesStopF64 = ScratchArena.copyInts(stridesStopF64);
          final cStridesResF64 = ScratchArena.copyInts(resF64.strides);
          final cStridesStepF64 = ScratchArena.copyInts(stridesStepF64);
          s_linspace_grid_double(
            startF64.pointer.cast<ffi.Double>(),
            cStridesStartF64,
            stopF64.pointer.cast<ffi.Double>(),
            cStridesStopF64,
            resF64.pointer.cast<ffi.Double>(),
            cStridesResF64,
            stepF64.pointer.cast<ffi.Double>(),
            cStridesStepF64,
            cShape,
            rank,
            actualAxis,
            numSamples,
            endpoint ? 1 : 0,
          );
          final resPtr = resF64.pointer.cast<ffi.Double>();
          for (var i = 0; i < resF64.size; i++) {
            resPtr[i] = resPtr[i].floorToDouble();
          }
          final stepPtr = stepF64.pointer.cast<ffi.Double>();
          for (var i = 0; i < stepF64.size; i++) {
            stepPtr[i] = stepPtr[i].floorToDouble();
          }
          castNDArray<T>(resF64, resolvedDType).copy(out: res);
          castNDArray<T>(stepF64, resolvedDType).copy(out: step);
        case DType.uint64:
          final div = endpoint ? (numSamples - 1) : numSamples;
          final iter = NDIter.broadcast3(
            startBroadcasted,
            stopBroadcasted,
            step,
          );
          while (iter.moveNext()) {
            final sInt =
                startBroadcasted.getCellRawUntyped(iter.getIndex(0)) as int;
            final eInt =
                stopBroadcasted.getCellRawUntyped(iter.getIndex(1)) as int;
            final sD = sInt < 0
                ? BigInt.from(sInt).toUnsigned(64).toDouble()
                : sInt.toDouble();
            final eD = eInt < 0
                ? BigInt.from(eInt).toUnsigned(64).toDouble()
                : eInt.toDouble();
            final stp = numSamples <= 1 ? 0.0 : (eD - sD) / div;
            step.setCellRaw(
              iter.getIndex(2),
              saturatingDoubleToInt(stp.floorToDouble(), DType.uint64),
            );
            var baseOffset = res.offsetElements;
            final coords = iter.coords;
            for (var d = 0; d < commonShape.length; d++) {
              final resDim = d < actualAxis ? d : d + 1;
              baseOffset += coords[d] * res.strides[resDim];
            }
            final axisStride = res.strides[actualAxis];
            for (var k = 0; k < numSamples; k++) {
              final int cellVal;
              if (endpoint && numSamples > 1 && k == numSamples - 1) {
                cellVal = eInt;
              } else {
                cellVal = saturatingDoubleToInt(
                  (sD + k * stp).floorToDouble(),
                  DType.uint64,
                );
              }
              res.setCellRaw(baseOffset + k * axisStride, cellVal);
            }
          }
        case DType.boolean:
          throw UnsupportedError(
            'linspaceGrid not supported for type $resolvedDType',
          );
      }
      checkNativeOom();
    } finally {
      ScratchArena.reset(marker);
    }

    if (useTempOut) {
      res.copy(out: out);
      step.detachToParentScope();
      return (samples: out, step: step);
    }

    if (out == null) res.detachToParentScope();
    step.detachToParentScope();
    return (samples: res, step: step);
  });
}

/// Returns numbers spaced evenly on a log scale.
///
/// In linear space, the sequence starts at `base ** start` and ends with `base ** stop`.
///
/// **Preconditions:**
/// - [numSamples] must be non-negative.
///
/// - It is an error if [numSamples] is negative.
///
/// **Memory Ownership & Lifetime:**
/// - Allocates a new array on the unmanaged C heap. The caller takes full ownership of this memory and must explicitly call [dispose] to prevent native leaks, unless executing inside a managed [NDArray.scope].
NDArray<T> logspace<T extends DTypeTag>(
  Object? start,
  Object? stop,
  int numSamples, {
  double base = 10.0,
  bool endpoint = true,
  required DType<T> dtype,
  NDArray<T>? out,
}) {
  if (numSamples < 0) {
    throw ArgumentError.value(
      numSamples,
      'numSamples',
      'Must be non-negative (was $numSamples)',
    );
  }
  final resolvedDType = dtype;
  if (resolvedDType == DType.boolean) {
    throw UnsupportedError('logspace not supported for type $resolvedDType');
  }
  if (out != null) {
    if (out.isDisposed) {
      throw StateError(
        'Cannot write logspace result to a disposed output array.',
      );
    }
    validateOutBuffer(out);
    if (!listEquals(out.shape, [numSamples]) || out.dtype != resolvedDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Incompatible out buffer shape or dtype',
      );
    }
  }
  if (numSamples == 0) {
    return out ?? NDArray<T>.create([0], resolvedDType);
  }

  final bool useTempOut = out != null && !out.isContiguous;
  final div = endpoint ? (numSamples - 1) : numSamples;

  return NDArray.scope(() {
    final arr = (out != null && !useTempOut)
        ? out
        : NDArray<T>.create([numSamples], resolvedDType);

    switch (resolvedDType) {
      case DType.float64:
        final s = (start as num).toDouble();
        final e = (stop as num).toDouble();
        final stp = numSamples <= 1 ? 0.0 : (e - s) / div;
        v_logspace_double(arr.pointer.cast(), s, stp, base, numSamples);
      case DType.float32:
        final s = (start as num).toDouble();
        final e = (stop as num).toDouble();
        final stp = numSamples <= 1 ? 0.0 : (e - s) / div;
        v_logspace_float(arr.pointer.cast(), s, stp, base, numSamples);
      case DType.complex128:
        final s = normalizeScalar(start as Object, DType.complex128) as Complex;
        final e = normalizeScalar(stop as Object, DType.complex128) as Complex;
        final stp = numSamples <= 1 ? Complex(0.0, 0.0) : (e - s) / div;
        v_logspace_complex128(
          arr.pointer.cast(),
          s.real,
          s.imag,
          stp.real,
          stp.imag,
          base,
          0.0,
          numSamples,
        );
      case DType.complex64:
        final s = normalizeScalar(start as Object, DType.complex128) as Complex;
        final e = normalizeScalar(stop as Object, DType.complex128) as Complex;
        final stp = numSamples <= 1 ? Complex(0.0, 0.0) : (e - s) / div;
        v_logspace_complex64(
          arr.pointer.cast(),
          s.real,
          s.imag,
          stp.real,
          stp.imag,
          base,
          0.0,
          numSamples,
        );
      case DType.float16:
      case DType.bfloat16:
      case DType.int8:
      case DType.uint64:
      case DType.uint32:
      case DType.uint16:
      case DType.int64:
      case DType.int32:
      case DType.int16:
      case DType.uint8:
      case DType.boolean:
        throw UnsupportedError(
          'logspace not supported for type $resolvedDType',
        );
    }

    if (useTempOut) {
      arr.copy(out: out);
      return out;
    }
    if (out == null) {
      arr.detachToParentScope();
    }
    return arr;
  });
}

/// Generalized [logspace] that supports broadcasting.
///
/// **Example:**
/// {@example /example/shaping_example.dart lang=dart}
///
/// **Parameters:**
/// - [start]: The starting value(s) as an [NDArray].
/// - [stop]: The end value(s) as an [NDArray].
/// - [numSamples]: Number of samples to generate. Must be non-negative.
/// - [base]: The base of the log space as an [NDArray]. Defaults to 10.0.
/// - [endpoint]: If true, `stop` is the last sample. Otherwise, it is not included.
/// - [axis]: The axis in the result to store the samples. Defaults to 0.
/// - [dtype]: The type of the output array. If not provided, it defaults to
///   the resolved dtype between [start] and [stop].
NDArray<T> logspaceGrid<T extends DTypeTag>(
  NDArray<T> start,
  NDArray<T> stop,
  int numSamples, {
  NDArray<T>? base,
  bool endpoint = true,
  int axis = 0,
  DType<T>? dtype,
  NDArray<T>? out,
}) {
  if (start.isDisposed || stop.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute logspaceGrid() on a disposed array.');
  }
  if (base != null && base.isDisposed) {
    throw StateError(
      'Cannot execute logspaceGrid() with a disposed base array.',
    );
  }
  if (dtype == DType.boolean ||
      start.dtype == DType.boolean ||
      stop.dtype == DType.boolean ||
      (base != null && base.dtype == DType.boolean)) {
    throw UnsupportedError('logspaceGrid not supported for boolean arrays');
  }
  final resolvedDType =
      dtype ?? (resolveDType(start.dtype, stop.dtype) as DType<T>);

  return NDArray.scope(() {
    final startArr = toNDArray<T>(start, resolvedDType);
    final stopArr = toNDArray<T>(stop, resolvedDType);
    final actualBase = base != null
        ? toNDArray<T>(base, resolvedDType)
        : toNDArray<T>(10.0, resolvedDType);

    final commonShape = broadcastShapes(
      broadcastShapes(startArr.shape, stopArr.shape),
      actualBase.shape,
    );
    final outRank = commonShape.length + 1;
    final actualAxis = axis < 0 ? outRank + axis : axis;
    if (actualAxis < 0 || actualAxis >= outRank) {
      throw RangeError.range(
        axis,
        -outRank,
        outRank - 1,
        'axis',
        'Must be within valid rank range',
      );
    }

    final startBroad = broadcastTo(startArr, commonShape);
    final stopBroad = broadcastTo(stopArr, commonShape);
    final baseBroad = broadcastTo(actualBase, commonShape);

    final y = linspaceGrid<T>(
      startBroad,
      stopBroad,
      numSamples,
      endpoint: endpoint,
      axis: actualAxis,
      dtype: resolvedDType,
    );

    final expandedBaseShape = List<int>.from(commonShape)
      ..insert(actualAxis, 1);
    final baseExpanded = baseBroad.reshape(expandedBaseShape);
    final res = power<T>(baseExpanded, y);
    if (out != null) {
      validateOutBuffer(out);
      if (!listEquals(out.shape, res.shape) || out.dtype != resolvedDType) {
        throw ArgumentError.value(
          out,
          'out',
          'Incompatible out buffer shape or dtype',
        );
      }
      res.copy(out: out);
      return out;
    }
    res.detachToParentScope();
    return res;
  });
}

/// Returns numbers spaced evenly on a log scale (geometric progression).
///
/// This is similar to [logspace], but with the start and end points specified directly.
///
/// **Preconditions:**
/// - [numSamples] must be non-negative.
/// - [start] and [stop] must be non-zero and have the same sign.
///
/// - It is an error if [numSamples] is negative.
/// - It is an error if [start] or [stop] is zero.
/// - It is an error if [start] and [stop] have different signs.
///
/// **Memory Ownership & Lifetime:**
/// - Allocates a new array on the unmanaged C heap. The caller takes full ownership of this memory and must explicitly call [dispose] to prevent native leaks, unless executing inside a managed [NDArray.scope].
NDArray<T> geomspace<T extends DTypeTag>(
  Object? start,
  Object? stop,
  int numSamples, {
  bool endpoint = true,
  required DType<T> dtype,
  NDArray<T>? out,
}) {
  if (numSamples < 0) {
    throw ArgumentError.value(
      numSamples,
      'numSamples',
      'Must be non-negative (was $numSamples)',
    );
  }
  final resolvedDType = dtype;
  if (out != null) {
    if (out.isDisposed) {
      throw StateError(
        'Cannot write geomspace result to a disposed output array.',
      );
    }
    validateOutBuffer(out);
    if (!listEquals(out.shape, [numSamples]) || out.dtype != resolvedDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Incompatible out buffer shape or dtype',
      );
    }
  }
  if (numSamples == 0) {
    return out ?? NDArray<T>.create([0], resolvedDType);
  }

  final bool useTempOut = out != null && !out.isContiguous;

  return NDArray.scope(() {
    switch (resolvedDType) {
      case DType.float64:
      case DType.float32:
        final s = (start as num).toDouble();
        final e = (stop as num).toDouble();
        if (s == 0.0 || e == 0.0) {
          throw ArgumentError.value(
            start,
            'start',
            'Geometric sequence cannot include zero',
          );
        }
        if ((s > 0.0) != (e > 0.0)) {
          throw ArgumentError.value(
            stop,
            'stop',
            'Geometric sequence start and stop must have same sign',
          );
        }

        final sign = s > 0.0 ? 1.0 : -1.0;
        final logStart = math.log(s.abs()) / math.ln10;
        final logStop = math.log(e.abs()) / math.ln10;
        final div = endpoint ? (numSamples - 1) : numSamples;
        final stp = numSamples <= 1 ? 0.0 : (logStop - logStart) / div;

        final arr = (out != null && !useTempOut)
            ? out
            : NDArray<T>.create([numSamples], resolvedDType);
        if (resolvedDType == DType.float64) {
          v_geomspace_double(
            arr.pointer.cast(),
            logStart,
            stp,
            sign,
            numSamples,
          );
        } else {
          v_geomspace_float(
            arr.pointer.cast(),
            logStart,
            stp,
            sign,
            numSamples,
          );
        }
        if (useTempOut) {
          arr.copy(out: out);
          return out;
        }
        if (out == null) {
          arr.detachToParentScope();
        }
        return arr;
      case DType.complex128:
      case DType.complex64:
        final s = normalizeScalar(start as Object, DType.complex128) as Complex;
        final e = normalizeScalar(stop as Object, DType.complex128) as Complex;
        if (s.abs == 0.0 || e.abs == 0.0) {
          throw ArgumentError.value(
            start,
            'start',
            'Geometric sequence cannot include zero',
          );
        }

        final logStart = s.log() / math.ln10;
        final logStop = e.log() / math.ln10;
        final div = endpoint ? (numSamples - 1) : numSamples;
        final stp = numSamples <= 1
            ? Complex(0.0, 0.0)
            : (logStop - logStart) / div;

        final arr = (out != null && !useTempOut)
            ? out
            : NDArray<T>.create([numSamples], resolvedDType);
        if (resolvedDType == DType.complex128) {
          v_geomspace_complex128(
            arr.pointer.cast(),
            logStart.real,
            logStart.imag,
            stp.real,
            stp.imag,
            numSamples,
          );
        } else {
          v_geomspace_complex64(
            arr.pointer.cast(),
            logStart.real,
            logStart.imag,
            stp.real,
            stp.imag,
            numSamples,
          );
        }
        if (useTempOut) {
          arr.copy(out: out);
          return out;
        }
        if (out == null) {
          arr.detachToParentScope();
        }
        return arr;
      case DType.float16:
      case DType.bfloat16:
      case DType.int8:
      case DType.uint64:
      case DType.uint32:
      case DType.uint16:
      case DType.int64:
      case DType.int32:
      case DType.int16:
      case DType.uint8:
      case DType.boolean:
        throw UnsupportedError(
          'geomspace not supported for type $resolvedDType',
        );
    }
  });
}

/// Generalized [geomspace] that supports broadcasting.
///
/// **Example:**
/// {@example /example/shaping_example.dart lang=dart}
///
/// **Parameters:**
/// - [start]: The starting value(s) as an [NDArray].
/// - [stop]: The end value(s) as an [NDArray].
/// - [numSamples]: Number of samples to generate. Must be non-negative.
/// - [endpoint]: If true, `stop` is the last sample. Otherwise, it is not included.
/// - [axis]: The axis in the result to store the samples. Defaults to 0.
/// - [dtype]: The type of the output array. If not provided, it defaults to
///   the resolved dtype between [start] and [stop].
NDArray<T> geomspaceGrid<T extends DTypeTag>(
  NDArray<T> start,
  NDArray<T> stop,
  int numSamples, {
  bool endpoint = true,
  int axis = 0,
  DType<T>? dtype,
  NDArray<T>? out,
}) {
  if (start.isDisposed || stop.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute geomspaceGrid() on a disposed array.');
  }
  if ((dtype != null && (dtype.isInteger || dtype == DType.boolean)) ||
      start.dtype.isInteger ||
      start.dtype == DType.boolean ||
      stop.dtype.isInteger ||
      stop.dtype == DType.boolean) {
    throw UnsupportedError(
      'geomspaceGrid not supported for integer or boolean types',
    );
  }

  final resolvedDType =
      dtype ?? (resolveDType(start.dtype, stop.dtype) as DType<T>);

  return NDArray.scope(() {
    final startArr = toNDArray(start, resolvedDType);
    final stopArr = toNDArray(stop, resolvedDType);

    final zero = toNDArray<T>(0.0, resolvedDType);
    final startZero = equal(startArr, zero);
    final stopZero = equal(stopArr, zero);
    if (any(startZero).scalar || any(stopZero).scalar) {
      throw ArgumentError.value(
        start,
        'start',
        'Geometric sequence cannot include zero',
      );
    }

    if (resolvedDType.isFloating) {
      final startNeg = less(startArr, zero);
      final stopNeg = less(stopArr, zero);
      final diffSign = notEqual(startNeg, stopNeg);
      if (any(diffSign).scalar) {
        throw ArgumentError.value(
          stop,
          'stop',
          'Geometric sequence start and stop must have same sign',
        );
      }
    }

    final commonShape = broadcastShapes(startArr.shape, stopArr.shape);
    final outRank = commonShape.length + 1;
    final actualAxis = axis < 0 ? outRank + axis : axis;
    if (actualAxis < 0 || actualAxis >= outRank) {
      throw RangeError.range(
        axis,
        -outRank,
        outRank - 1,
        'axis',
        'Must be within valid rank range',
      );
    }
    final startBroad = broadcastTo(startArr, commonShape);
    final stopBroad = broadcastTo(stopArr, commonShape);

    if (resolvedDType.isFloating) {
      final signs = sign<T>(startBroad);
      final absStart = abs(startBroad as NDArray<AnySpec>) as NDArray<T>;
      final absStop = abs(stopBroad as NDArray<AnySpec>) as NDArray<T>;
      final compDType =
          (resolvedDType == DType.float16 || resolvedDType == DType.bfloat16)
          ? DType.float64
          : resolvedDType;
      final ln10Arr = toNDArray<DTypeTag>(math.ln10, compDType);
      final logStart = divideUntyped<DTypeTag, DTypeTag>(
        log(absStart as NDArray<AnySpec>),
        ln10Arr,
      );
      final logStop = divideUntyped<DTypeTag, DTypeTag>(
        log(absStop as NDArray<AnySpec>),
        ln10Arr,
      );
      final y = linspaceGrid<DTypeTag>(
        logStart,
        logStop,
        numSamples,
        endpoint: endpoint,
        axis: actualAxis,
        dtype: compDType,
      );
      final powResComp = power<DTypeTag>(
        toNDArray<DTypeTag>(10.0, compDType),
        y,
      );
      final powRes = powResComp.dtype == resolvedDType
          ? powResComp as NDArray<T>
          : castNDArray<T>(powResComp, resolvedDType);
      final expandedSignShape = List<int>.from(commonShape)
        ..insert(actualAxis, 1);
      final signsExpanded = signs.reshape(expandedSignShape);
      final res = multiply<T>(signsExpanded, powRes);
      if (out != null) {
        validateOutBuffer(out);
        if (!listEquals(out.shape, res.shape) || out.dtype != resolvedDType) {
          throw ArgumentError.value(
            out,
            'out',
            'Incompatible out buffer shape or dtype',
          );
        }
        res.copy(out: out);
        return out;
      }
      res.detachToParentScope();
      return res;
    }

    final logStart = divideUntyped<T, T>(
      log(startBroad as NDArray<AnySpec>) as NDArray<T>,
      toNDArray<T>(math.ln10, resolvedDType),
    );
    final logStop = divideUntyped<T, T>(
      log(stopBroad as NDArray<AnySpec>) as NDArray<T>,
      toNDArray<T>(math.ln10, resolvedDType),
    );

    final y = linspaceGrid<T>(
      logStart,
      logStop,
      numSamples,
      endpoint: endpoint,
      axis: actualAxis,
      dtype: resolvedDType,
    );
    final res = power<T>(toNDArray<T>(10.0, resolvedDType), y);
    if (out != null) {
      validateOutBuffer(out);
      if (!listEquals(out.shape, res.shape) || out.dtype != resolvedDType) {
        throw ArgumentError.value(
          out,
          'out',
          'Incompatible out buffer shape or dtype',
        );
      }
      res.copy(out: out);
      return out;
    }
    res.detachToParentScope();
    return res;
  });
}
