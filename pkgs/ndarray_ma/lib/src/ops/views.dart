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

MaskedArray<T> _maReshape<T extends DTypeTag>(
  MaskedArray<T> self,
  List<int> newShape,
) {
  return MaskedArray<T>(
    self.data.reshape(newShape),
    self.mask.reshape(newShape),
    fillValue: self.fillValue,
  );
}

MaskedArray<T> _maTranspose<T extends DTypeTag>(
  MaskedArray<T> self, [
  List<int>? axes,
]) {
  return MaskedArray<T>(
    self.data.transpose(axes),
    self.mask.transpose(axes),
    fillValue: self.fillValue,
  );
}

MaskedArray<T> _maExpandDims<T extends DTypeTag>(
  MaskedArray<T> self,
  int axis,
) {
  return MaskedArray<T>(
    self.data.expandDims(axis),
    self.mask.expandDims(axis),
    fillValue: self.fillValue,
  );
}

NDArray<T> _maCompressed<T extends DTypeTag>(MaskedArray<T> self) {
  return NDArray.scope(() {
    final flatData = self.data.reshape([self.size]);
    final flatMask = self.mask.reshape([self.size]);
    final invertedMask = ndops.logicalNot(flatMask);
    final result = flatData.applyMask(invertedMask);
    return result.detachToParentScope();
  });
}

NDArray<T> _maFilled<T extends DTypeTag>(
  MaskedArray<T> self, {
  Object? fillValue,
}) {
  return NDArray.scope(() {
    final val = fillValue ?? self.fillValue;
    final fillArray = _wrapScalar<T>(val, self.dtype);
    final result = ndops.where(self.mask, fillArray, self.data) as NDArray<T>;
    return result.detachToParentScope();
  });
}

MaskedArray<R> _maMapUnary<T extends DTypeTag, R extends DTypeTag>(
  MaskedArray<T> self,
  NDArray<R> Function(NDArray<T>) ufunc,
) {
  final mappedData = ufunc(self.data);
  final newFillValue = _coerceOrGetDefault(self.fillValue, mappedData.dtype);
  return dispatchCreateMaskedArray(
        mappedData,
        self.mask,
        fillValue: newFillValue,
      )
      as MaskedArray<R>;
}
