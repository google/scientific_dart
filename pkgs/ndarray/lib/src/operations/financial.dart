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

import '../ndarray.dart';
import '../exceptions.dart';
import 'math.dart';
import 'stats.dart';
import 'linalg.dart';
import 'sorting.dart';

/// Specifies when payments are due within a period in financial calculations.
enum PaymentDue {
  /// Payments are due at the end of each period (default).
  end,

  /// Payments are due at the beginning of each period.
  begin,
}

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
  dynamic when = 0,
  NDArray<Float64>? out,
}) {
  if (rate.isDisposed ||
      nper.isDisposed ||
      pmt.isDisposed ||
      pv.isDisposed ||
      (out != null && out.isDisposed) ||
      (when is NDArray && when.isDisposed)) {
    throw StateError('Cannot perform operation on a disposed array.');
  }
  return NDArray.scope(() {
    final whenArr = _parseWhen(when);

    final one = NDArray<Float64>.scalar(Float64(1.0), dtype: DType.float64);
    final zero = NDArray<Float64>.scalar(Float64(0.0), dtype: DType.float64);

    // temp = (1 + rate) ** nper
    final NDArray<Float64> onePlusRate = add(one, rate);
    final NDArray<Float64> temp = power(onePlusRate, nper);

    // fv_zero = - (pv + pmt * nper)
    final NDArray<Float64> pmtNper = multiply(pmt, nper);
    final NDArray<Float64> pvPlusPmtNper = add(pv, pmtNper);
    final NDArray<Float64> fvZero = negative(pvPlusPmtNper);

    // fv_nonzero = - (pv * temp + pmt * (1 + rate * when) / rate * (temp - 1))
    final NDArray<Float64> rateWhen = multiply(rate, whenArr);
    final NDArray<Float64> rateWhenPlusOne = add(one, rateWhen);
    final NDArray<Float64> pmtRateWhenPlusOne = multiply(pmt, rateWhenPlusOne);
    final NDArray<Float64> factor = divide(pmtRateWhenPlusOne, rate);
    final NDArray<Float64> tempMinusOne = subtract(temp, one);
    final NDArray<Float64> term2 = multiply(factor, tempMinusOne);
    final NDArray<Float64> fvNonzeroInner = add(multiply(pv, temp), term2);
    final NDArray<Float64> fvNonzero = negative(fvNonzeroInner);

    final cond = equal(rate, zero);
    final result = where(cond, fvZero, fvNonzero, out) as NDArray<Float64>;
    return result.detachToParentScope();
  });
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
  dynamic when = 0,
  NDArray<Float64>? out,
}) {
  if (rate.isDisposed ||
      nper.isDisposed ||
      pmt.isDisposed ||
      fv.isDisposed ||
      (out != null && out.isDisposed) ||
      (when is NDArray && when.isDisposed)) {
    throw StateError('Cannot perform operation on a disposed array.');
  }
  return NDArray.scope(() {
    final whenArr = _parseWhen(when);

    final one = NDArray<Float64>.scalar(Float64(1.0), dtype: DType.float64);
    final zero = NDArray<Float64>.scalar(Float64(0.0), dtype: DType.float64);

    // temp = (1 + rate) ** nper
    final NDArray<Float64> onePlusRate = add(one, rate);
    final NDArray<Float64> temp = power(onePlusRate, nper);

    // pv_zero = - (fv + pmt * nper)
    final NDArray<Float64> pmtNper = multiply(pmt, nper);
    final NDArray<Float64> fvPlusPmtNper = add(fv, pmtNper);
    final NDArray<Float64> pvZero = negative(fvPlusPmtNper);

    // pv_nonzero = - (fv + pmt * (1 + rate * when) / rate * (temp - 1)) / temp
    final NDArray<Float64> rateWhen = multiply(rate, whenArr);
    final NDArray<Float64> rateWhenPlusOne = add(one, rateWhen);
    final NDArray<Float64> pmtRateWhenPlusOne = multiply(pmt, rateWhenPlusOne);
    final NDArray<Float64> factor = divide(pmtRateWhenPlusOne, rate);
    final NDArray<Float64> tempMinusOne = subtract(temp, one);
    final NDArray<Float64> term2 = multiply(factor, tempMinusOne);
    final NDArray<Float64> pvNonzeroInner = divide(add(fv, term2), temp);
    final NDArray<Float64> pvNonzero = negative(pvNonzeroInner);

    final cond = equal(rate, zero);
    final result = where(cond, pvZero, pvNonzero, out) as NDArray<Float64>;
    return result.detachToParentScope();
  });
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
    throw ArgumentError('values must be at least 1D');
  }

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

    final one = NDArray<Float64>.scalar(Float64(1.0), dtype: DType.float64);
    final NDArray<Float64> onePlusRate = add(one, rateExpanded);
    final NDArray<Float64> discount = power(onePlusRate, t);

    final NDArray<Float64> divided = divide(valuesExpanded, discount);

    final sumAxis = divided.rank - 1;
    final NDArray<Float64> result = sum(divided, axis: sumAxis, out: out);
    return result.detachToParentScope();
  });
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
    throw ArgumentError('values must be a 1D array');
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
      final result = out ?? NDArray<Float64>.create([], DType.float64);
      result.setCell([], Float64(double.nan));
      return result.detachToParentScope();
    }

    final n = coeffs.shape[0] - 1;
    if (n <= 0) {
      if (raiseExceptions) {
        throw const NoRealSolutionException(
          'No real solution is found for IRR.',
        );
      }
      final result = out ?? NDArray<Float64>.create([], DType.float64);
      result.setCell([], Float64(double.nan));
      return result.detachToParentScope();
    }

    final companion = NDArray<Float64>.zeros([n, n], DType.float64);
    for (var j = 0; j < n; j++) {
      companion.setCellFlat(
        j,
        Float64(-coeffs.getCellFlat(j + 1) / coeffs.getCellFlat(0)),
      );
    }
    for (var i = 1; i < n; i++) {
      companion.setCellFlat(i * n + i - 1, Float64(1.0));
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

    final result = out ?? NDArray<Float64>.create([], DType.float64);
    result.setCell([], Float64(selectedRate));
    return result.detachToParentScope();
  });
}

NDArray<Float64> _parseWhen(dynamic when) {
  if (when is PaymentDue) {
    final val = when == PaymentDue.begin ? 1.0 : 0.0;
    return NDArray<Float64>.scalar(Float64(val), dtype: DType.float64);
  }
  if (when is NDArray) {
    if (when.dtype != DType.float64) {
      throw ArgumentError('when NDArray must be of type DType.float64');
    }
    return when as NDArray<Float64>;
  }
  double val;
  if (when is String) {
    final lower = when.toLowerCase();
    if (lower == 'begin' ||
        lower == 'beginning' ||
        lower == '1' ||
        lower == 'start') {
      val = 1.0;
    } else if (lower == 'end' || lower == '0' || lower == 'finish') {
      val = 0.0;
    } else {
      throw ArgumentError('Invalid when value: $when');
    }
  } else if (when is num) {
    val = when.toDouble();
  } else {
    throw ArgumentError('Invalid when type: ${when.runtimeType}');
  }
  return NDArray<Float64>.scalar(Float64(val), dtype: DType.float64);
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

  // The first element in the indices array gives the index of the first non-zero cash flow.
  final firstNonZeroIndex = indices.getCell([0]);
  if (firstNonZeroIndex == 0) {
    return values; // No leading zeros to strip.
  }

  // Return a sliced view starting at the first non-zero index.
  // This avoids copying any data from the original NDArray.
  return values.slice([Slice(start: firstNonZeroIndex)]);
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

  final first = coeffs.getCell([0]);
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
