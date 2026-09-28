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

MaskedArray<T> _maSum<T extends DTypeTag>(MaskedArray<T> self, {int? axis}) =>
    _reduction<T>(
      self,
      (arr, {axis}) => ndops.sumAs<T, T>(
        arr,
        (self.dtype == DType.boolean ? DType.int64 : self.dtype) as DType<T>,
        axis: axis,
      ),
      _zeroValue(self.dtype),
      axis,
    );

MaskedArray<T> _maProd<T extends DTypeTag>(MaskedArray<T> self, {int? axis}) =>
    _reduction<T>(
      self,
      (arr, {axis}) => ndops.prodAs<T, T>(
        arr,
        (self.dtype == DType.boolean ? DType.int64 : self.dtype) as DType<T>,
        axis: axis,
      ),
      _oneValue(self.dtype),
      axis,
    );

MaskedArray<T> _maMin<T extends DTypeTag>(MaskedArray<T> self, {int? axis}) {
  if (self.dtype.isComplex || self.dtype == DType.boolean) {
    throw UnsupportedError('Unsupported dtype for min: ${self.dtype}');
  }
  return _reduction<T>(
    self,
    (a, {axis}) => ndops.min<T>(a, axis: axis),
    _maxValue(self.dtype),
    axis,
  );
}

MaskedArray<T> _maMax<T extends DTypeTag>(MaskedArray<T> self, {int? axis}) {
  if (self.dtype.isComplex || self.dtype == DType.boolean) {
    throw UnsupportedError('Unsupported dtype for max: ${self.dtype}');
  }
  return _reduction<T>(
    self,
    (a, {axis}) => ndops.max<T>(a, axis: axis),
    _minValue(self.dtype),
    axis,
  );
}

NDArray<Int32> _maCount(MaskedArray self, {int? axis}) {
  return NDArray.scope(() {
    final zeros = NDArray<Int32>.zeros(self.shape, DType.int32);
    final ones = NDArray<Int32>.ones(self.shape, DType.int32);
    final validMap = ndops.where(self.mask, zeros, ones) as NDArray<Int32>;
    final result = ndops.sumAs(validMap, DType.int32, axis: axis);
    return result.detachToParentScope();
  });
}

MaskedArray<DTypeTag> _maMean(MaskedArray self, {int? axis}) {
  return NDArray.scope(() {
    final s = self.sum(axis: axis);
    final c = self.count(axis: axis);
    return s.divide(c).detachToParentScope();
  });
}

MaskedArray<DTypeTag> _maVariance(MaskedArray self, {int? axis}) {
  return NDArray.scope(() {
    final m = self.mean(axis: axis);
    final MaskedArray<DTypeTag> mExpanded;
    if (axis != null) {
      mExpanded = m.expandDims(axis);
    } else {
      mExpanded = m;
    }
    final diff = self.subtract(mExpanded);
    final diffSq = diff.multiply(diff);
    final result = diffSq.mean(axis: axis);
    return result.detachToParentScope();
  });
}

MaskedArray<DTypeTag> _maStd(MaskedArray self, {int? axis}) {
  return NDArray.scope(() {
    final v = self.variance(axis: axis);
    final result = v.mapUnary((data) => ndops.sqrt(data as NDArray<AnySpec>));
    return result.detachToParentScope();
  });
}

MaskedArray<T> _reduction<T extends DTypeTag>(
  MaskedArray<T> self,
  NDArray<T> Function(NDArray<T>, {int? axis}) ndOp,
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
        as MaskedArray<T>;
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
