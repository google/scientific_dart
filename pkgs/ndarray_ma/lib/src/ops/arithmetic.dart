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

part of '../masked_array.dart';

MaskedArray<T> _maAdd<T extends DTypeTag>(MaskedArray<T> self, Object? other) =>
    _maBinary<T, T>(self, other, '+', (a, b) => a + b, isDivision: false);

MaskedArray<T> _maSubtract<T extends DTypeTag>(
  MaskedArray<T> self,
  Object? other,
) => _maBinary<T, T>(self, other, '-', (a, b) => a - b, isDivision: false);

MaskedArray<T> _maMultiply<T extends DTypeTag>(
  MaskedArray<T> self,
  Object? other,
) => _maBinary<T, T>(self, other, '*', (a, b) => a * b, isDivision: false);

MaskedArray<T> _maFloorDivide<T extends DTypeTag>(
  MaskedArray<T> self,
  Object? other,
) => _maBinary<T, T>(self, other, '~/', (a, b) => a ~/ b, isDivision: true);

MaskedArray<T> _maRemainder<T extends DTypeTag>(
  MaskedArray<T> self,
  Object? other,
) => _maBinary<T, T>(self, other, '%', (a, b) => a % b, isDivision: true);

/// True division. The runtime dtype of `a / b` on an [NDArray] follows
/// NumPy's `true_divide` rule, which is exactly the [DivideOf] projection
/// that [M] is bound to by the callers.
MaskedArray<M> _maDivide<T extends DTypeTag, M extends DTypeTag>(
  MaskedArray<T> self,
  Object? other,
) => _maBinary<T, M>(
  self,
  other,
  '/',
  (a, b) => (a / b) as NDArray<M>,
  isDivision: true,
);

MaskedArray<Boolean> _maCompare<T extends DTypeTag>(
  MaskedArray<T> self,
  Object? other,
  String operator,
  NDArray<Boolean> Function(NDArray<T> a, Object operand) compare,
) => _maBinary<T, Boolean>(self, other, operator, compare, isDivision: false);

/// Rejects an array operand whose dtype differs from the receiver's, mirroring
/// the contract of the [NDArray] operators.
void _checkSameDTypeOperand(
  DType receiverDType,
  DType operandDType,
  Object operand,
  String operator,
) {
  if (operandDType != receiverDType) {
    throw ArgumentError.value(
      operand,
      'other',
      'Must have the same dtype as the receiver ($receiverDType) for operator '
          '$operator, but has dtype $operandDType. Cast explicitly with '
          '.astype(...) before combining arrays of different dtypes.',
    );
  }
}

bool _isZeroScalar(Object scalar) => switch (scalar) {
  num() => scalar == 0,
  bool() => !scalar,
  Complex() => scalar.real == 0 && scalar.imag == 0,
  _ => false,
};

Object _oneLikeScalar(Object scalar) => switch (scalar) {
  int() => 1,
  double() => 1.0,
  bool() => true,
  _ => Complex(1, 0),
};

/// Shared implementation of the binary element-wise operations.
///
/// [compute] receives the receiver's data and either an `NDArray<T>` operand
/// (already validated to have the receiver's dtype) or the raw scalar
/// operand, and delegates to the corresponding [NDArray] operation so that
/// scalar conversion, broadcasting, and error contracts are identical to
/// those of [NDArray].
///
/// When [isDivision] is `true`, positions where the divisor is masked or zero
/// are replaced by one before computing (so integer floor division and
/// remainder never trap), and are masked in the result.
MaskedArray<R> _maBinary<T extends DTypeTag, R extends DTypeTag>(
  MaskedArray<T> self,
  Object? other,
  String operator,
  NDArray<R> Function(NDArray<T> a, Object operand) compute, {
  required bool isDivision,
}) {
  final NDArray<T>? otherData;
  final NDArray<Boolean>? otherMask;
  switch (other) {
    case MaskedArray():
      _checkSameDTypeOperand(self.dtype, other.dtype, other, operator);
      otherData = other.data as NDArray<T>;
      otherMask = other.mask;
    case NDArray():
      _checkSameDTypeOperand(self.dtype, other.dtype, other, operator);
      otherData = other as NDArray<T>;
      otherMask = null;
    case num() || bool() || Complex():
      otherData = null;
      otherMask = null;
    default:
      throw ArgumentError.value(
        other,
        'other',
        'Must be a MaskedArray, an NDArray, a num, a bool, or a Complex.',
      );
  }

  return NDArray.scope(() {
    final NDArray<R> resultData;
    final NDArray<Boolean> resultMask;
    if (otherData == null) {
      final scalar = other!;
      if (isDivision && _isZeroScalar(scalar)) {
        // Division by a zero scalar is a domain error everywhere: compute
        // against one so integer kernels never trap, and mask the result.
        resultData = compute(self.data, _oneLikeScalar(scalar));
        resultMask = NDArray.full(resultData.shape, true, dtype: DType.boolean);
      } else {
        resultData = compute(self.data, scalar);
        resultMask = self.mask.copy();
      }
    } else {
      var combinedMask = otherMask != null
          ? ndops.logicalOr(self.mask, otherMask)
          : self.mask;
      Object operand = otherData;
      if (isDivision) {
        final isZero = otherData.eq(_zeroValue(otherData.dtype));
        combinedMask = ndops.logicalOr(combinedMask, isZero);
        final one = _wrapScalar<T>(_oneValue(otherData.dtype), otherData.dtype);
        operand = ndops.where(combinedMask, one, otherData) as NDArray<T>;
      }
      resultData = compute(self.data, operand);
      resultMask = ndops.broadcastTo(combinedMask, resultData.shape).copy();
    }
    return dispatchCreateMaskedArray(
          resultData.detachToParentScope(),
          resultMask.detachToParentScope(),
          fillValue: self.fillValue,
        )
        as MaskedArray<R>;
  });
}
