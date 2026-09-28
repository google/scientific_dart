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

import 'dart:ffi' as ffi;
import '../ndarray.dart';
import '../ndarray_bindings.dart';
import '../scratch_arena.dart';
import 'helpers.dart';

/// Supported interpolation methods for one-dimensional interpolation.
enum InterpolationMethod {
  /// Piecewise linear interpolation.
  linear,

  /// Nearest-neighbor interpolation.
  nearest,
}

/// Validates that [xp] is strictly increasing.
///
/// It is an error if [xp] is not strictly increasing.
void _validateSorted(NDArray<Float64> xp) {
  final size = xp.shape[0];
  if (size <= 1) return;

  final res = is_strictly_increasing_double(
    xp.pointer.cast(),
    size,
    xp.strides[0],
  );
  if (res == 0) {
    throw ArgumentError('xp must be strictly increasing.');
  }
}

/// Computes one-dimensional interpolation.
///
/// Returns the one-dimensional piecewise interpolant to a function with
/// given discrete data points ([xp], [fp]), evaluated at [x].
/// The [xp] array must be strictly increasing and have the same length as [fp].
/// Optional [left] and [right] specify values to return for `x < xp[0]` and `x > xp[xp.length-1]` respectively, defaulting to `fp[0]` and `fp[fp.length-1]`.
///
/// **Preconditions:**
/// - [x], [xp], [fp] must not be disposed.
/// - [x] and [xp] must be real-valued (not complex).
/// - [xp] and [fp] must be 1D arrays.
/// - [xp] and [fp] must have the same length.
/// - [xp] must be strictly increasing.
///
/// - It is an error if any input array is disposed.
/// - It is an error if [x] or [xp] is complex.
/// - It is an error if [xp] or [fp] is not 1-dimensional, or if their lengths mismatch.
/// - It is an error if [xp] is empty.
/// - It is an error if [xp] is not strictly increasing.
///
/// **Example:**
/// {@example /example/interpolation_example.dart}
NDArray<R> interp<R extends DTypeTag>(
  NDArray<DTypeTag> x,
  NDArray<DTypeTag> xp,
  NDArray<DTypeTag> fp, {
  Object? left,
  Object? right,
  InterpolationMethod method = InterpolationMethod.linear,
  NDArray<R>? out,
}) {
  if (x.isDisposed ||
      xp.isDisposed ||
      fp.isDisposed ||
      (out != null && out.isDisposed)) {
    throw StateError('Cannot execute interp() with disposed arrays.');
  }

  if (x.dtype.isComplex) {
    throw ArgumentError.value(x.dtype, 'x', 'Must not be complex.');
  }
  if (xp.dtype.isComplex) {
    throw ArgumentError.value(xp.dtype, 'xp', 'Must not be complex.');
  }

  final isComplexFp = fp.dtype.isComplex;
  final expectedDType = isComplexFp ? DType.complex128 : DType.float64;

  if (out != null) {
    if (!listEquals(out.shape, x.shape) || out.dtype != expectedDType) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }

  if (xp.shape.length != 1 || fp.shape.length != 1) {
    throw ArgumentError('xp and fp must be 1-dimensional arrays.');
  }

  if (xp.shape[0] != fp.shape[0]) {
    throw ArgumentError('xp and fp must have the same length.');
  }

  if (xp.shape[0] == 0) {
    throw ArgumentError('xp must not be empty.');
  }

  if (isComplexFp) {
    Complex? leftC;
    if (left != null) {
      if (left is Complex) {
        leftC = left;
      } else if (left is num) {
        leftC = Complex(left.toDouble(), 0.0);
      } else {
        throw ArgumentError.value(left, 'left', 'Must be a num or Complex.');
      }
    }
    Complex? rightC;
    if (right != null) {
      if (right is Complex) {
        rightC = right;
      } else if (right is num) {
        rightC = Complex(right.toDouble(), 0.0);
      } else {
        throw ArgumentError.value(right, 'right', 'Must be a num or Complex.');
      }
    }

    return NDArray.scope(() {
      final fpLen = fp.shape[0];
      final fpReal = NDArray<Float64>.create([fpLen], DType.float64);
      final fpImag = NDArray<Float64>.create([fpLen], DType.float64);
      final fpRealPtr = fpReal.pointer.cast<ffi.Double>();
      final fpImagPtr = fpImag.pointer.cast<ffi.Double>();
      for (var i = 0; i < fpLen; i++) {
        final c = fp.getCell([i]) as Complex;
        fpRealPtr[i] = c.real;
        fpImagPtr[i] = c.imag;
      }

      final resReal = interp<Float64>(
        x,
        xp,
        fpReal,
        left: leftC?.real,
        right: rightC?.real,
        method: method,
      );
      final resImag = interp<Float64>(
        x,
        xp,
        fpImag,
        left: leftC?.imag,
        right: rightC?.imag,
        method: method,
      );

      final target =
          out ??
          (NDArray<Complex128>.create(x.shape, DType.complex128) as NDArray<R>);
      final tempTarget =
          (out != null &&
              (!out.isContiguous ||
                  sharesMemory(x, out) ||
                  sharesMemory(xp, out) ||
                  sharesMemory(fp, out)))
          ? (NDArray<Complex128>.create(x.shape, DType.complex128)
                as NDArray<R>)
          : target;

      final size = resReal.size;
      final rPtr = resReal.pointer.cast<ffi.Double>();
      final iPtr = resImag.pointer.cast<ffi.Double>();
      if (tempTarget.isContiguous) {
        final outPtr = tempTarget.pointer.cast<ffi.Double>();
        for (var i = 0; i < size; i++) {
          outPtr[2 * i] = rPtr[i];
          outPtr[2 * i + 1] = iPtr[i];
        }
      } else {
        for (var i = 0; i < size; i++) {
          tempTarget.setCellFlat(i, Complex(rPtr[i], iPtr[i]));
        }
      }

      if (!identical(tempTarget, target)) {
        tempTarget.copy(out: target);
      }
      if (out == null) {
        target.detachToParentScope();
      }
      return target;
    });
  }

  double? leftD;
  if (left != null) {
    if (left is num) {
      leftD = left.toDouble();
    } else {
      throw ArgumentError.value(
        left,
        'left',
        'Must be a real number when fp is real.',
      );
    }
  }
  double? rightD;
  if (right != null) {
    if (right is num) {
      rightD = right.toDouble();
    } else {
      throw ArgumentError.value(
        right,
        'right',
        'Must be a real number when fp is real.',
      );
    }
  }

  final xDouble = x.dtype == DType.float64
      ? x as NDArray<Float64>
      : promoteToDouble(x);
  NDArray<Float64>? xpDouble;
  NDArray<Float64>? fpDouble;

  try {
    xpDouble = xp.dtype == DType.float64
        ? xp as NDArray<Float64>
        : promoteToDouble(xp);
    fpDouble = fp.dtype == DType.float64
        ? fp as NDArray<Float64>
        : promoteToDouble(fp);

    _validateSorted(xpDouble);

    if (out != null &&
        (sharesMemory(x, out) ||
            sharesMemory(xp, out) ||
            sharesMemory(fp, out))) {
      return NDArray.scope(() {
        final temp = NDArray<Float64>.create(x.shape, DType.float64);
        interp<Float64>(
          xDouble,
          xpDouble!,
          fpDouble!,
          left: leftD,
          right: rightD,
          method: method,
          out: temp,
        );
        temp.copy(out: out as NDArray<Float64>);
        return out;
      });
    }

    final res =
        (out as NDArray<Float64>?) ??
        NDArray<Float64>.create(x.shape, DType.float64);

    if (method == InterpolationMethod.nearest) {
      final size = xDouble.size;
      final xpSize = xpDouble.shape[0];
      final xpContig = xpDouble.isContiguous ? xpDouble : xpDouble.copy();
      try {
        final fpContig = fpDouble.isContiguous ? fpDouble : fpDouble.copy();
        try {
          final xContig = xDouble.isContiguous ? xDouble : xDouble.copy();
          try {
            final xpPtr = xpContig.pointer.cast<ffi.Double>();
            final fpPtr = fpContig.pointer.cast<ffi.Double>();
            final xPtr = xContig.pointer.cast<ffi.Double>();

            final xpMin = xpPtr[0];
            final xpMax = xpPtr[xpSize - 1];
            final defaultLeft = leftD ?? fpPtr[0];
            final defaultRight = rightD ?? fpPtr[xpSize - 1];

            final tempRes = res.isContiguous
                ? res
                : NDArray<Float64>.create(x.shape, DType.float64);
            try {
              final tempResPtr = tempRes.pointer.cast<ffi.Double>();
              for (var i = 0; i < size; i++) {
                final xv = xPtr[i];
                if (xv.isNaN) {
                  tempResPtr[i] = double.nan;
                } else if (xv < xpMin) {
                  tempResPtr[i] = defaultLeft;
                } else if (xv > xpMax) {
                  tempResPtr[i] = defaultRight;
                } else if (xpSize == 1) {
                  tempResPtr[i] = fpPtr[0];
                } else {
                  var low = 0;
                  var high = xpSize - 1;
                  while (low < high - 1) {
                    final mid = (low + high) ~/ 2;
                    if (xpPtr[mid] <= xv) {
                      low = mid;
                    } else {
                      high = mid;
                    }
                  }
                  final x0 = xpPtr[low];
                  final x1 = xpPtr[low + 1];
                  final y0 = fpPtr[low];
                  final y1 = fpPtr[low + 1];
                  if ((xv - x0).abs() <= (x1 - xv).abs()) {
                    tempResPtr[i] = y0;
                  } else {
                    tempResPtr[i] = y1;
                  }
                }
              }
              if (!identical(tempRes, res)) {
                tempRes.copy(out: res);
              }
            } finally {
              if (!identical(tempRes, res)) {
                tempRes.dispose();
              }
            }
          } finally {
            if (!identical(xContig, xDouble)) xContig.dispose();
          }
        } finally {
          if (!identical(fpContig, fpDouble)) fpContig.dispose();
        }
      } finally {
        if (!identical(xpContig, xpDouble)) xpContig.dispose();
      }
    } else {
      final marker = ScratchArena.marker;
      try {
        // Prepare left/right pointers.
        ffi.Pointer<ffi.Double> pLeft = ffi.nullptr;
        if (leftD != null) {
          pLeft = ScratchArena.allocate<ffi.Double>(ffi.sizeOf<ffi.Double>());
          pLeft.value = leftD;
        }
        ffi.Pointer<ffi.Double> pRight = ffi.nullptr;
        if (rightD != null) {
          pRight = ScratchArena.allocate<ffi.Double>(ffi.sizeOf<ffi.Double>());
          pRight.value = rightD;
        }

        final isContiguous =
            xDouble.isContiguous &&
            xpDouble.isContiguous &&
            fpDouble.isContiguous &&
            res.isContiguous;

        if (isContiguous) {
          native_interp_double(
            xDouble.pointer.cast(),
            xDouble.shape.isEmpty ? 1 : xDouble.shape.reduce((a, b) => a * b),
            xpDouble.pointer.cast(),
            xpDouble.shape[0],
            fpDouble.pointer.cast(),
            res.pointer.cast(),
            pLeft,
            pRight,
          );
        } else {
          // Strided version.
          var ndim = xDouble.shape.length;
          final cBuffer = ScratchArena.getStridedBuffer(ndim == 0 ? 1 : ndim);
          final cShape = cBuffer;
          final cStridesX = ScratchArena.copyInts(
            ndim == 0 ? [0] : xDouble.strides,
          );
          final cStridesRes = ScratchArena.copyInts(
            ndim == 0 ? [0] : res.strides,
          );

          if (ndim == 0) {
            cShape[0] = 1;
            ndim = 1;
          } else {
            for (var i = 0; i < ndim; i++) {
              cShape[i] = xDouble.shape[i];
            }
          }

          s_interp_double(
            xDouble.pointer.cast(),
            cStridesX,
            xpDouble.pointer.cast(),
            xpDouble.strides.isEmpty ? 1 : xpDouble.strides[0],
            xpDouble.shape[0],
            fpDouble.pointer.cast(),
            fpDouble.strides.isEmpty ? 1 : fpDouble.strides[0],
            res.pointer.cast(),
            cStridesRes,
            cShape,
            ndim,
            pLeft,
            pRight,
          );
        }
      } finally {
        ScratchArena.reset(marker);
      }
    }

    return res as NDArray<R>;
  } finally {
    // Dispose promoted arrays if they were created.
    if (!identical(xDouble, x)) xDouble.dispose();
    if (xpDouble != null && !identical(xpDouble, xp)) xpDouble.dispose();
    if (fpDouble != null && !identical(fpDouble, fp)) fpDouble.dispose();
  }
}

/// Computes one-dimensional interpolation.
///
/// Alias for [interp].
NDArray<R> interpolate<R extends DTypeTag>(
  NDArray<DTypeTag> x,
  NDArray<DTypeTag> xp,
  NDArray<DTypeTag> fp, {
  Object? left,
  Object? right,
  InterpolationMethod method = InterpolationMethod.linear,
  NDArray<R>? out,
}) => interp<R>(x, xp, fp, left: left, right: right, method: method, out: out);
