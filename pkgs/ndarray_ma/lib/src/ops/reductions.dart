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

DType<DTypeTag> _accumulatorDType(DType<DTypeTag> dtype) => switch (dtype) {
  DType.boolean || DType.int8 || DType.int16 || DType.int32 => DType.int64,
  DType.uint8 || DType.uint16 || DType.uint32 => DType.uint64,
  _ => dtype,
};

DType<DTypeTag> _doublePrecisionDType(DType<DTypeTag> dtype) =>
    dtype.isComplex ? DType.complex128 : DType.float64;

MaskedArray<R> _maSum<T extends DTypeTag, R extends DTypeTag>(
  MaskedArray<T> self, {
  int? axis,
}) => _reduction<T, R>(
  self,
  (arr, {axis}) =>
      ndops.sumAs(arr, _accumulatorDType(self.dtype) as DType<R>, axis: axis),
  _zeroValue(self.dtype),
  axis,
);

MaskedArray<R> _maProd<T extends DTypeTag, R extends DTypeTag>(
  MaskedArray<T> self, {
  int? axis,
}) => _reduction<T, R>(
  self,
  (arr, {axis}) =>
      ndops.prodAs(arr, _accumulatorDType(self.dtype) as DType<R>, axis: axis),
  _oneValue(self.dtype),
  axis,
);

MaskedArray<T> _maMin<T extends DTypeTag>(MaskedArray<T> self, {int? axis}) {
  if (self.dtype.isComplex || self.dtype == DType.boolean) {
    throw UnsupportedError('Unsupported dtype for min: ${self.dtype}');
  }
  return _reduction<T, T>(
    self,
    (a, {axis}) => ndops.min(a, axis: axis),
    _maxValue(self.dtype),
    axis,
  );
}

MaskedArray<T> _maMax<T extends DTypeTag>(MaskedArray<T> self, {int? axis}) {
  if (self.dtype.isComplex || self.dtype == DType.boolean) {
    throw UnsupportedError('Unsupported dtype for max: ${self.dtype}');
  }
  return _reduction<T, T>(
    self,
    (a, {axis}) => ndops.max(a, axis: axis),
    _minValue(self.dtype),
    axis,
  );
}

NDArray<Int64> _maCount(MaskedArray self, {int? axis}) {
  return NDArray.scope(() {
    final zeros = NDArray<Int64>.zeros(self.shape, DType.int64);
    final ones = NDArray<Int64>.ones(self.shape, DType.int64);
    final validMap = ndops.where(self.mask, zeros, ones) as NDArray<Int64>;
    final result = ndops.sumAs(validMap, DType.int64, axis: axis);
    return result.detachToParentScope();
  });
}

MaskedArray<D> _maMean<T extends DTypeTag, D extends DTypeTag>(
  MaskedArray<T> self, {
  int? axis,
}) {
  return NDArray.scope(() {
    final targetDType = _doublePrecisionDType(self.dtype) as DType<D>;
    final s = _maSum<T, DTypeTag>(self, axis: axis).astype<D>(targetDType);
    final c = self.count(axis: axis).astype<D>(targetDType);
    return _maDivide<D, D>(s, c).detachToParentScope();
  });
}

MaskedArray<Float64> _maVariance(MaskedArray self, {int? axis}) {
  return NDArray.scope(() {
    final targetDType = _doublePrecisionDType(self.dtype);
    final selfPromoted = self.astype(targetDType);
    final m = _maMean<DTypeTag, DTypeTag>(selfPromoted, axis: axis);
    final mExpanded = axis != null ? m.expandDims(axis) : m;
    final diff = selfPromoted.subtract(mExpanded);
    final MaskedArray<Float64> diffSq;
    if (self.dtype.isComplex) {
      final mag = diff.mapUnary<Float64>(
        (d) => ndops.abs(d as NDArray<Complex128>),
      );
      diffSq = mag.multiply(mag);
    } else {
      diffSq = diff.multiply(diff) as MaskedArray<Float64>;
    }
    return _maMean<Float64, Float64>(diffSq, axis: axis).detachToParentScope();
  });
}

MaskedArray<Float64> _maStd(MaskedArray self, {int? axis}) {
  return NDArray.scope(() {
    final v = self.variance(axis: axis);
    final result = v.mapUnary<Float64>((data) => ndops.sqrt(data));
    return result.detachToParentScope();
  });
}

MaskedArray<R> _reduction<T extends DTypeTag, R extends DTypeTag>(
  MaskedArray<T> self,
  NDArray<R> Function(NDArray<T>, {int? axis}) ndOp,
  dynamic fillValueForReduction,
  int? axis,
) {
  return NDArray.scope(() {
    final fillArray = _wrapScalar<T>(fillValueForReduction, self.dtype);
    final filledData =
        ndops.where(self.mask, fillArray, self.data) as NDArray<T>;
    final resultData = ndOp(filledData, axis: axis);
    final resultMask = ndops.all(self.mask, axis: axis);
    return dispatchCreateMaskedArray(
          resultData.detachToParentScope(),
          resultMask.detachToParentScope(),
          fillValue: self.fillValue,
        )
        as MaskedArray<R>;
  });
}

Object _zeroValue(DType dtype) => switch (dtype) {
  DType.float64 || DType.float32 || DType.float16 || DType.bfloat16 => 0.0,
  DType.complex128 || DType.complex64 => Complex(0, 0),
  DType.int64 ||
  DType.int32 ||
  DType.int16 ||
  DType.int8 ||
  DType.uint64 ||
  DType.uint32 ||
  DType.uint16 ||
  DType.uint8 => 0,
  DType.boolean => false,
};

Object _oneValue(DType dtype) => switch (dtype) {
  DType.float64 || DType.float32 || DType.float16 || DType.bfloat16 => 1.0,
  DType.complex128 || DType.complex64 => Complex(1, 0),
  DType.int64 ||
  DType.int32 ||
  DType.int16 ||
  DType.int8 ||
  DType.uint64 ||
  DType.uint32 ||
  DType.uint16 ||
  DType.uint8 => 1,
  DType.boolean => true,
};
