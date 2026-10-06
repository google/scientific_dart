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

/// Quantitative Financial Operations.
///
/// This library provides vectorized financial functions
/// (`fv`, `pv`, `npv`, `irr`) designed for quantitative simulation and modeling.
///
/// **Intended Use Cases:**
/// - **Quantitative Modeling**: Large-scale parameter sweeps, grid searches, and portfolio valuation.
/// - **Monte Carlo Simulations**: Simulating thousands of randomized cash flow scenarios.
/// - **Backtesting**: Backtesting quantitative strategies that involve periodic cash flows.
///
/// **Limitations & Non-Goals:**
/// - **Not for Ledger Accounting**: These functions use double-precision floats (`Float64`) for vectorization. They are not intended for commercial banking or exact ledger accounting where arbitrary-precision decimals are mandatory to prevent rounding errors.
/// - **Not for Arbitrary Date Cash Flows**: These functions assume flat, periodic intervals. They do not support date-aware discounting (e.g., `XNPV`/`XIRR`).
/// - **Constant Rates**: `npv` assumes a constant discount rate across all periods, rather than a yield curve.
library;

import 'dart:ffi' as ffi;
import 'dart:math' as math;

import '../exceptions.dart';
import '../ndarray.dart';
import 'broadcasting.dart' show broadcastTo;
import 'helpers.dart' show sharesMemory, validateOutBuffer;
import 'linalg.dart';
import 'math.dart';
import 'sorting.dart';
import 'stats.dart';

/// Specifies when payments are due within a period in financial calculations.
enum PaymentDue {
  /// Payments are due at the end of each period (default).
  end,

  /// Payments are due at the beginning of each period.
  begin,
}

enum _TVMMode { fv, pv, pmt }

/// Future Value function.
///
/// Replicates the behavior of `numpy_financial.fv` exactly.
///
/// **Decimal Support:**
/// `Decimal` is not supported because `ndarray` uses double-precision floats
/// for quantitative simulations, which allow vectorization.
///
/// **Preconditions:**
/// - All input arrays must not be disposed.
/// - It is an error if any input array is disposed.
/// - It is an error if [out] has incompatible shape or dtype.
///
/// {@example /example/financial_example.dart lang=dart}
NDArray<Float64> fv(
  NDArray<Float64> rate,
  NDArray<Float64> nper,
  NDArray<Float64> pmt,
  NDArray<Float64> pv, {
  PaymentDue when = PaymentDue.end,
  NDArray<Float64>? out,
}) {
  if (rate.isDisposed ||
      nper.isDisposed ||
      pmt.isDisposed ||
      pv.isDisposed ||
      (out != null && out.isDisposed)) {
    throw StateError('Cannot perform operation on a disposed array.');
  }
  return _computeTVM(
    rate: rate,
    nper: nper,
    pmt: pmt,
    pv: pv,
    mode: _TVMMode.fv,
    when: when,
    out: out,
  );
}

/// Present Value function.
///
/// Replicates the behavior of `numpy_financial.pv` exactly.
///
/// **Decimal Support:**
/// `Decimal` is not supported because `ndarray` uses double-precision floats
/// for quantitative simulations, which allow vectorization.
///
/// **Preconditions:**
/// - All input arrays must not be disposed.
/// - It is an error if any input array is disposed.
/// - It is an error if [out] has incompatible shape or dtype.
///
/// {@example /example/financial_example.dart lang=dart}
NDArray<Float64> pv(
  NDArray<Float64> rate,
  NDArray<Float64> nper,
  NDArray<Float64> pmt,
  NDArray<Float64> fv, {
  PaymentDue when = PaymentDue.end,
  NDArray<Float64>? out,
}) {
  if (rate.isDisposed ||
      nper.isDisposed ||
      pmt.isDisposed ||
      fv.isDisposed ||
      (out != null && out.isDisposed)) {
    throw StateError('Cannot perform operation on a disposed array.');
  }
  return _computeTVM(
    rate: rate,
    nper: nper,
    pmt: pmt,
    fv: fv,
    mode: _TVMMode.pv,
    when: when,
    out: out,
  );
}

/// Payment function.
///
/// Computes the payment against loan principal plus interest.
/// Replicates the behavior of `numpy_financial.pmt` exactly.
///
/// **Decimal Support:**
/// `Decimal` is not supported because `ndarray` uses double-precision floats
/// for quantitative simulations, which allow vectorization.
///
/// **Preconditions:**
/// - All input arrays must not be disposed.
/// - It is an error if any input array is disposed.
/// - It is an error if [out] has incompatible shape or dtype.
///
/// {@example /example/financial_example.dart lang=dart}
NDArray<Float64> pmt(
  NDArray<Float64> rate,
  NDArray<Float64> nper,
  NDArray<Float64> pv, {
  NDArray<Float64>? fv,
  PaymentDue when = PaymentDue.end,
  NDArray<Float64>? out,
}) {
  if (rate.isDisposed ||
      nper.isDisposed ||
      pv.isDisposed ||
      (fv != null && fv.isDisposed) ||
      (out != null && out.isDisposed)) {
    throw StateError('Cannot perform operation on a disposed array.');
  }
  return _computeTVM(
    rate: rate,
    nper: nper,
    pv: pv,
    fv: fv,
    mode: _TVMMode.pmt,
    when: when,
    out: out,
  );
}

/// Net Present Value function.
///
/// Replicates the behavior of `numpy_financial.npv` exactly.
///
/// **Decimal Support:**
/// `Decimal` is not supported because `ndarray` uses double-precision floats
/// for quantitative simulations, which allow vectorization.
///
/// **Preconditions:**
/// - [rate] and [values] must not be disposed.
/// - [values] must have rank $\ge 1$.
/// - It is an error if any input array is disposed.
/// - It is an error if [values] rank is less than 1.
/// - It is an error if [out] has incompatible shape or dtype.
///
/// {@example /example/financial_example.dart lang=dart}
NDArray<Float64> npv(
  NDArray<Float64> rate,
  NDArray<Float64> values, {
  NDArray<Float64>? out,
}) {
  if (rate.isDisposed || values.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot perform operation on a disposed array.');
  }
  if (values.rank < 1) {
    throw ArgumentError.value(
      values.rank,
      'values',
      'Must be at least 1D (got rank ${values.rank})',
    );
  }

  final expectedOutShape = <int>[
    ...rate.shape,
    ...values.shape.sublist(0, values.rank - 1),
  ];

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, expectedOutShape) ||
        out.dtype != DType.float64) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape or dtype (expected shape $expectedOutShape and dtype ${DType.float64}, got shape ${out.shape} and dtype ${out.dtype})',
      );
    }
  }

  // Fast single-pass path when rate is a scalar (size 1).
  if (rate.size == 1) {
    final bool outSharesMem =
        out != null &&
        (!out.isContiguous ||
            sharesMemory(rate, out) ||
            sharesMemory(values, out));

    if (outSharesMem) {
      return NDArray.scope(() {
        final temp = NDArray<Float64>.create(expectedOutShape, DType.float64);
        _computeNpvScalarRate(rate, values, temp);
        temp.copy(out: out);
        return out;
      });
    }

    if (out != null) {
      return NDArray.scope(() {
        _computeNpvScalarRate(rate, values, out);
        return out;
      });
    }

    return NDArray.scope(() {
      final result = NDArray<Float64>.create(expectedOutShape, DType.float64);
      _computeNpvScalarRate(rate, values, result);
      return result.detachToParentScope();
    });
  }

  // Fallback path when rate is multi-element.
  return NDArray.scope(() {
    final N = values.shape.last;
    final t = NDArray<Float64>.arange(0.0, N.toDouble(), dtype: DType.float64);

    final rateRank = rate.rank;
    final valuesRank = values.rank;

    final List<int> rateExpandedShape = [
      ...rate.shape,
      ...List.filled(valuesRank - 1, 1),
      1,
    ];
    final List<int> valuesExpandedShape = [
      ...List.filled(rateRank, 1),
      ...values.shape,
    ];

    final NDArray<Float64> rateExpanded = rate.reshape(rateExpandedShape);
    final NDArray<Float64> valuesExpanded = values.reshape(valuesExpandedShape);

    final one = NDArray<Float64>.scalar(1.0, dtype: DType.float64);
    final NDArray<Float64> onePlusRate = add(one, rateExpanded);
    final NDArray<Float64> discount = power(onePlusRate, t);

    final NDArray<Float64> divided = divide(valuesExpanded, discount);

    final sumAxis = divided.rank - 1;
    final NDArray<Float64> result = sum(divided, axis: sumAxis, out: out);
    if (out != null) {
      if (!identical(result, out)) {
        result.copy(out: out);
      }
      return out;
    }
    return result.detachToParentScope();
  });
}

void _computeNpvScalarRate(
  NDArray<Float64> rate,
  NDArray<Float64> values,
  NDArray<Float64> dest,
) {
  final N = values.shape.last;
  final numBatches = dest.size;
  final pDest = dest.pointer.cast<ffi.Double>();

  if (N == 0) {
    for (var b = 0; b < numBatches; b++) {
      pDest[b] = 0.0;
    }
    return;
  }

  final r = rate.pointer.cast<ffi.Double>()[0];
  final valuesContig = values.isContiguous ? values : values.copy();
  final pVal = valuesContig.pointer.cast<ffi.Double>();

  if (1.0 + r > 0) {
    final invFactor = 1.0 / (1.0 + r);
    for (var b = 0; b < numBatches; b++) {
      final base = b * N;
      var discount = 1.0;
      var total = 0.0;
      for (var t = 0; t < N; t++) {
        total += pVal[base + t] * discount;
        discount *= invFactor;
      }
      pDest[b] = total;
    }
  } else {
    for (var b = 0; b < numBatches; b++) {
      final base = b * N;
      var total = 0.0;
      for (var t = 0; t < N; t++) {
        total += pVal[base + t] * math.pow(1.0 + r, -t);
      }
      pDest[b] = total;
    }
  }
}

NDArray<Float64> _computeTVM({
  required NDArray<Float64> rate,
  required NDArray<Float64> nper,
  NDArray<Float64>? pmt,
  NDArray<Float64>? pv,
  NDArray<Float64>? fv,
  required _TVMMode mode,
  required PaymentDue when,
  NDArray<Float64>? out,
}) {
  var commonShape = broadcastShapes(rate.shape, nper.shape);
  if (pmt != null) commonShape = broadcastShapes(commonShape, pmt.shape);
  if (pv != null) commonShape = broadcastShapes(commonShape, pv.shape);
  if (fv != null) commonShape = broadcastShapes(commonShape, fv.shape);

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, commonShape) || out.dtype != DType.float64) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape or dtype (expected shape $commonShape and dtype ${DType.float64}, got shape ${out.shape} and dtype ${out.dtype})',
      );
    }
  }

  final bool outSharesMem =
      out != null &&
      (!out.isContiguous ||
          sharesMemory(rate, out) ||
          sharesMemory(nper, out) ||
          (pmt != null && sharesMemory(pmt, out)) ||
          (pv != null && sharesMemory(pv, out)) ||
          (fv != null && sharesMemory(fv, out)));

  if (outSharesMem) {
    return NDArray.scope(() {
      final temp = NDArray<Float64>.create(commonShape, DType.float64);
      _computeTVMDirect(
        rate: rate,
        nper: nper,
        pmt: pmt,
        pv: pv,
        fv: fv,
        mode: mode,
        when: when,
        dest: temp,
      );
      temp.copy(out: out);
      return out;
    });
  }

  if (out != null) {
    return NDArray.scope(() {
      _computeTVMDirect(
        rate: rate,
        nper: nper,
        pmt: pmt,
        pv: pv,
        fv: fv,
        mode: mode,
        when: when,
        dest: out,
      );
      return out;
    });
  }

  return NDArray.scope(() {
    final result = NDArray<Float64>.create(commonShape, DType.float64);
    _computeTVMDirect(
      rate: rate,
      nper: nper,
      pmt: pmt,
      pv: pv,
      fv: fv,
      mode: mode,
      when: when,
      dest: result,
    );
    return result.detachToParentScope();
  });
}

void _computeTVMDirect({
  required NDArray<Float64> rate,
  required NDArray<Float64> nper,
  NDArray<Float64>? pmt,
  NDArray<Float64>? pv,
  NDArray<Float64>? fv,
  required _TVMMode mode,
  required PaymentDue when,
  required NDArray<Float64> dest,
}) {
  final size = dest.size;
  if (size == 0) return;

  final commonShape = dest.shape;
  final double w = when == PaymentDue.begin ? 1.0 : 0.0;
  final pDest = dest.pointer.cast<ffi.Double>();

  final bool fastRate =
      rate.size == 1 ||
      (rate.isContiguous && listEquals(rate.shape, commonShape));
  final bool fastNper =
      nper.size == 1 ||
      (nper.isContiguous && listEquals(nper.shape, commonShape));
  final bool fastPmt =
      pmt == null ||
      pmt.size == 1 ||
      (pmt.isContiguous && listEquals(pmt.shape, commonShape));
  final bool fastPv =
      pv == null ||
      pv.size == 1 ||
      (pv.isContiguous && listEquals(pv.shape, commonShape));
  final bool fastFv =
      fv == null ||
      fv.size == 1 ||
      (fv.isContiguous && listEquals(fv.shape, commonShape));

  if (fastRate && fastNper && fastPmt && fastPv && fastFv) {
    final sRate = rate.size == 1 ? 0 : 1;
    final sNper = nper.size == 1 ? 0 : 1;
    final sPmt = (pmt == null || pmt.size == 1) ? 0 : 1;
    final sPv = (pv == null || pv.size == 1) ? 0 : 1;
    final sFv = (fv == null || fv.size == 1) ? 0 : 1;

    final pRate = rate.pointer.cast<ffi.Double>();
    final pNper = nper.pointer.cast<ffi.Double>();
    final pPmt = pmt?.pointer.cast<ffi.Double>();
    final pPv = pv?.pointer.cast<ffi.Double>();
    final pFv = fv?.pointer.cast<ffi.Double>();

    switch (mode) {
      case _TVMMode.fv:
        final ptrPmt = pPmt!;
        final ptrPv = pPv!;
        if (sRate == 1 && sNper == 1 && sPmt == 1 && sPv == 1) {
          for (var i = 0; i < size; i++) {
            final r = pRate[i];
            final n = pNper[i];
            final p = ptrPmt[i];
            final v = ptrPv[i];
            if (r == 0.0) {
              pDest[i] = -(v + p * n);
            } else {
              final temp = math.pow(1.0 + r, n).toDouble();
              final factor = (p * (1.0 + r * w)) / r;
              pDest[i] = -(v * temp + factor * (temp - 1.0));
            }
          }
        } else {
          for (var i = 0; i < size; i++) {
            final r = pRate[i * sRate];
            final n = pNper[i * sNper];
            final p = ptrPmt[i * sPmt];
            final v = ptrPv[i * sPv];
            if (r == 0.0) {
              pDest[i] = -(v + p * n);
            } else {
              final temp = math.pow(1.0 + r, n).toDouble();
              final factor = (p * (1.0 + r * w)) / r;
              pDest[i] = -(v * temp + factor * (temp - 1.0));
            }
          }
        }
        return;

      case _TVMMode.pv:
        final ptrPmt = pPmt!;
        final ptrFv = pFv!;
        if (sRate == 1 && sNper == 1 && sPmt == 1 && sFv == 1) {
          for (var i = 0; i < size; i++) {
            final r = pRate[i];
            final n = pNper[i];
            final p = ptrPmt[i];
            final f = ptrFv[i];
            if (r == 0.0) {
              pDest[i] = -(f + p * n);
            } else {
              final temp = math.pow(1.0 + r, n).toDouble();
              final factor = (p * (1.0 + r * w)) / r;
              pDest[i] = -(f + factor * (temp - 1.0)) / temp;
            }
          }
        } else {
          for (var i = 0; i < size; i++) {
            final r = pRate[i * sRate];
            final n = pNper[i * sNper];
            final p = ptrPmt[i * sPmt];
            final f = ptrFv[i * sFv];
            if (r == 0.0) {
              pDest[i] = -(f + p * n);
            } else {
              final temp = math.pow(1.0 + r, n).toDouble();
              final factor = (p * (1.0 + r * w)) / r;
              pDest[i] = -(f + factor * (temp - 1.0)) / temp;
            }
          }
        }
        return;

      case _TVMMode.pmt:
        final ptrPv = pPv!;
        if (sRate == 1 && sNper == 1 && sPv == 1 && sFv == 1 && pFv != null) {
          for (var i = 0; i < size; i++) {
            final r = pRate[i];
            final n = pNper[i];
            final v = ptrPv[i];
            final f = pFv[i];
            if (r == 0.0) {
              pDest[i] = -(f + v) / n;
            } else {
              final temp = math.pow(1.0 + r, n).toDouble();
              final fact = (1.0 + r * w) * (temp - 1.0) / r;
              pDest[i] = -(f + v * temp) / fact;
            }
          }
        } else {
          for (var i = 0; i < size; i++) {
            final r = pRate[i * sRate];
            final n = pNper[i * sNper];
            final v = ptrPv[i * sPv];
            final f = pFv != null ? pFv[i * sFv] : 0.0;
            if (r == 0.0) {
              pDest[i] = -(f + v) / n;
            } else {
              final temp = math.pow(1.0 + r, n).toDouble();
              final fact = (1.0 + r * w) * (temp - 1.0) / r;
              pDest[i] = -(f + v * temp) / fact;
            }
          }
        }
        return;
    }
  }

  // General strided / broadcasted path:
  _computeTVMStrided(
    rate: rate,
    nper: nper,
    pmt: pmt,
    pv: pv,
    fv: fv,
    mode: mode,
    w: w,
    dest: dest,
  );
}

void _computeTVMStrided({
  required NDArray<Float64> rate,
  required NDArray<Float64> nper,
  NDArray<Float64>? pmt,
  NDArray<Float64>? pv,
  NDArray<Float64>? fv,
  required _TVMMode mode,
  required double w,
  required NDArray<Float64> dest,
}) {
  final commonShape = dest.shape;
  final ndim = commonShape.length;
  final size = dest.size;
  final pDest = dest.pointer.cast<ffi.Double>();

  final bRate = broadcastTo(rate, commonShape);
  final bNper = broadcastTo(nper, commonShape);
  final bPmt = pmt != null ? broadcastTo(pmt, commonShape) : null;
  final bPv = pv != null ? broadcastTo(pv, commonShape) : null;
  final bFv = fv != null ? broadcastTo(fv, commonShape) : null;

  final pRate = bRate.pointer.cast<ffi.Double>();
  final pNper = bNper.pointer.cast<ffi.Double>();
  final pPmt = bPmt?.pointer.cast<ffi.Double>();
  final pPv = bPv?.pointer.cast<ffi.Double>();
  final pFv = bFv?.pointer.cast<ffi.Double>();

  final stridesRate = bRate.strides;
  final stridesNper = bNper.strides;
  final stridesPmt = bPmt?.strides;
  final stridesPv = bPv?.strides;
  final stridesFv = bFv?.strides;

  var offRate = 0;
  var offNper = 0;
  var offPmt = 0;
  var offPv = 0;
  var offFv = 0;

  final coords = List<int>.filled(ndim, 0);

  for (var i = 0; i < size; i++) {
    final r = pRate[offRate];
    final n = pNper[offNper];

    switch (mode) {
      case _TVMMode.fv:
        final p = pPmt![offPmt];
        final v = pPv![offPv];
        if (r == 0.0) {
          pDest[i] = -(v + p * n);
        } else {
          final temp = math.pow(1.0 + r, n).toDouble();
          final factor = (p * (1.0 + r * w)) / r;
          pDest[i] = -(v * temp + factor * (temp - 1.0));
        }
      case _TVMMode.pv:
        final p = pPmt![offPmt];
        final f = pFv![offFv];
        if (r == 0.0) {
          pDest[i] = -(f + p * n);
        } else {
          final temp = math.pow(1.0 + r, n).toDouble();
          final factor = (p * (1.0 + r * w)) / r;
          pDest[i] = -(f + factor * (temp - 1.0)) / temp;
        }
      case _TVMMode.pmt:
        final v = pPv![offPv];
        final f = pFv != null ? pFv[offFv] : 0.0;
        if (r == 0.0) {
          pDest[i] = -(f + v) / n;
        } else {
          final temp = math.pow(1.0 + r, n).toDouble();
          final fact = (1.0 + r * w) * (temp - 1.0) / r;
          pDest[i] = -(f + v * temp) / fact;
        }
    }

    if (ndim > 0) {
      for (var d = ndim - 1; d >= 0; d--) {
        coords[d]++;
        if (coords[d] < commonShape[d]) {
          offRate += stridesRate[d];
          offNper += stridesNper[d];
          if (stridesPmt != null) offPmt += stridesPmt[d];
          if (stridesPv != null) offPv += stridesPv[d];
          if (stridesFv != null) offFv += stridesFv[d];
          break;
        }
        coords[d] = 0;
        final dimMinus1 = commonShape[d] - 1;
        offRate -= dimMinus1 * stridesRate[d];
        offNper -= dimMinus1 * stridesNper[d];
        if (stridesPmt != null) offPmt -= dimMinus1 * stridesPmt[d];
        if (stridesPv != null) offPv -= dimMinus1 * stridesPv[d];
        if (stridesFv != null) offFv -= dimMinus1 * stridesFv[d];
      }
    }
  }
}

/// Internal Rate of Return function.
///
/// Replicates the behavior of `numpy_financial.irr` exactly.
///
/// **Decimal Support:**
/// `Decimal` is not supported because `ndarray` uses double-precision floats
/// for quantitative simulations, which allow vectorization.
///
/// **Preconditions:**
/// - [values] must not be disposed and must be a 1-D array.
/// - It is an error if [values] or [out] is disposed.
/// - It is an error if [values] is not a 1-D array.
///
/// **Throws:**
/// - Throws [NoRealSolutionException] if [raiseExceptions] is `true` and no real solution exists.
///
/// {@example /example/financial_example.dart lang=dart}
NDArray<Float64> irr(
  NDArray<Float64> values, {
  bool raiseExceptions = false,
  NDArray<Float64>? out,
}) {
  if (values.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot perform operation on a disposed array.');
  }
  if (values.rank != 1) {
    throw ArgumentError.value(
      values.rank,
      'values',
      'Must be a 1D array (got rank ${values.rank})',
    );
  }
  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, const <int>[]) || out.dtype != DType.float64) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape or dtype (expected shape [] and dtype ${DType.float64}, got shape ${out.shape} and dtype ${out.dtype})',
      );
    }
  }

  return NDArray.scope(() {
    // Strip leading zeros to find the actual cash flow start.
    // This returns a zero-copy sliced view of the original values array.
    final coeffs = _getStrippedCoeffs(values);

    if (coeffs == null || _hasSameSign(coeffs)) {
      if (raiseExceptions) {
        throw const NoRealSolutionException(
          'No real solution exists for IRR since all cashflows are of the same sign.',
        );
      }
      if (out != null) {
        out.setCell([], double.nan);
        return out;
      }
      final result = NDArray<Float64>.create([], DType.float64);
      result.setCell([], double.nan);
      return result.detachToParentScope();
    }

    final n = coeffs.shape[0] - 1;
    if (n <= 0) {
      if (raiseExceptions) {
        throw const NoRealSolutionException(
          'No real solution is found for IRR.',
        );
      }
      if (out != null) {
        out.setCell([], double.nan);
        return out;
      }
      final result = NDArray<Float64>.create([], DType.float64);
      result.setCell([], double.nan);
      return result.detachToParentScope();
    }

    final companion = NDArray<Float64>.zeros([n, n], DType.float64);
    for (var j = 0; j < n; j++) {
      companion.setCellFlat(
        j,
        -coeffs.getCellFlat(j + 1) / coeffs.getCellFlat(0),
      );
    }
    for (var i = 1; i < n; i++) {
      companion.setCellFlat(i * n + i - 1, 1.0);
    }

    final eigResult = eig(companion);
    final List<double> eirr = [];
    for (var i = 0; i < n; i++) {
      final root = eigResult.eigenvalues.getCellFlat(i);
      if (root.imag.abs() < 1e-12) {
        final r = root.real - 1.0;
        if (r >= -1.0) {
          eirr.add(r);
        }
      }
    }

    final double selectedRate;
    if (eirr.isEmpty) {
      if (raiseExceptions) {
        throw const NoRealSolutionException(
          'No real solution is found for IRR.',
        );
      }
      selectedRate = double.nan;
    } else if (eirr.length == 1) {
      selectedRate = eirr[0];
    } else {
      selectedRate = _irrDefaultSelection(eirr);
    }

    if (out != null) {
      out.setCell([], selectedRate);
      return out;
    }
    final result = NDArray<Float64>.create([], DType.float64);
    result.setCell([], selectedRate);
    return result.detachToParentScope();
  });
}

/// Strips leading zero coefficients from the cash flow values.
///
/// Returns a zero-copy sliced view of [values] starting from the first non-zero element.
/// Returns `null` if [values] contains only zeros.
NDArray<Float64>? _getStrippedCoeffs(NDArray<Float64> values) {
  // Leverage FFI-accelerated nonzero search to find non-zero elements.
  final nonZeroIndicesList = nonzero(values);
  if (nonZeroIndicesList.isEmpty) return null;

  final indices = nonZeroIndicesList[0];
  if (indices.shape[0] == 0) {
    return null; // All cash flows are zero.
  }

  final firstNonZeroIndex = indices.getCell([0]);
  final lastNonZeroIndex = indices.getCell([indices.shape[0] - 1]);
  if (firstNonZeroIndex == 0 && lastNonZeroIndex == values.shape[0] - 1) {
    return values; // No leading or trailing zeros to strip.
  }

  // Return a sliced view spanning from the first to the last non-zero index.
  // This avoids copying any data from the original NDArray.
  return values.slice([
    Slice(start: firstNonZeroIndex, stop: lastNonZeroIndex + 1),
  ]);
}

/// Checks if all elements in [coeffs] have the same sign (all positive or all negative).
///
/// Returns `true` if empty.
///
/// **Implementation Note:**
/// Iterating manually in Dart is used here instead of bulk FFI operations (like
/// `min`/`max` or `greater` + `nonzero`) because:
/// 1. It allows for an immediate early-exit as soon as a sign change is found.
/// 2. It avoids allocating intermediate boolean/coordinate arrays.
/// Benchmarks show that for typical cash flows (N < 100), this manual loop is
/// faster than bulk FFI pipelines.
bool _hasSameSign(NDArray<Float64> coeffs) {
  final length = coeffs.shape[0];
  if (length <= 1) return true;

  final first = coeffs.getCell([0]) as num;
  if (first > 0) {
    return findIndex(coeffs, CompareOp.lessEqual, 0.0) == null;
  } else if (first < 0) {
    return findIndex(coeffs, CompareOp.greaterEqual, 0.0) == null;
  }
  return false;
}

double _irrDefaultSelection(List<double> eirr) {
  bool sameSign;
  if (eirr.isEmpty) {
    throw StateError('Cannot select from empty list of roots');
  }
  final first = eirr[0];
  if (first > 0) {
    sameSign = eirr.every((x) => x > 0);
  } else {
    sameSign = eirr.every((x) => x < 0);
  }

  List<double> filtered = List.from(eirr);
  if (!sameSign) {
    var posSum = 0.0;
    var negSum = 0.0;
    for (final x in eirr) {
      if (x > 0) {
        posSum += x;
      } else if (x < 0) {
        negSum += x;
      }
    }
    if (posSum >= negSum) {
      filtered = eirr.where((x) => x >= 0).toList();
    } else {
      filtered = eirr.where((x) => x < 0).toList();
    }
  }

  if (filtered.isEmpty) {
    filtered = List.from(eirr);
  }
  double minVal = filtered[0];
  double minAbs = minVal.abs();
  for (var i = 1; i < filtered.length; i++) {
    final absVal = filtered[i].abs();
    if (absVal < minAbs) {
      minAbs = absVal;
      minVal = filtered[i];
    }
  }
  return minVal;
}
