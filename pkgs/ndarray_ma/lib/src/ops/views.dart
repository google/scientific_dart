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

MaskedArray<T> _maSqueeze<T extends DTypeTag>(
  MaskedArray<T> self, [
  int? axis,
]) {
  return MaskedArray<T>(
    self.data.squeeze(axis: axis),
    self.mask.squeeze(axis: axis),
    fillValue: self.fillValue,
  );
}

MaskedArray<T> _maFlatten<T extends DTypeTag>(
  MaskedArray<T> self, {
  MaskedArray<DTypeTag>? out,
}) {
  return NDArray.scope(() {
    final flatData = self.data.flatten();
    final flatMask = self.mask.flatten();
    return _writeOrCreateMaskedArray<T>(
      flatData,
      flatMask,
      fillValue: self.fillValue,
      out: out,
    );
  });
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

NDArray<T> _maFilled<T extends DTypeTag, Out extends T>(
  MaskedArray<T> self, {
  Object? fillValue,
  NDArray<Out>? out,
}) {
  return NDArray.scope(() {
    final val = fillValue ?? self.fillValue;
    final fillArray = _wrapScalar<T>(val, self.dtype);
    final result =
        ndops.where<T>(self.mask, fillArray, self.data, out) as NDArray<T>;
    return out ?? result.detachToParentScope();
  });
}

MaskedArray<R> _maMapUnary<T extends DTypeTag, R extends DTypeTag>(
  MaskedArray<T> self,
  NDArray<R> Function(NDArray<T>) ufunc, {
  MaskedArray<DTypeTag>? out,
}) {
  return NDArray.scope(() {
    final mappedData = ufunc(self.data);
    final maskCopy = self.mask.copy();
    final newFillValue = _coerceOrGetDefault(self.fillValue, mappedData.dtype);
    return _writeOrCreateMaskedArray<R>(
      mappedData,
      maskCopy,
      fillValue: newFillValue,
      out: out,
    );
  });
}

MaskedArray<T> _maClip<T extends DTypeTag>(
  MaskedArray<T> self, {
  num? aMin,
  num? aMax,
  MaskedArray<DTypeTag>? out,
}) {
  return _maMapUnary<T, T>(
    self,
    (d) => ndops.clip(d, min: aMin, max: aMax),
    out: out,
  );
}

MaskedArray<T> _maWhere<T extends DTypeTag>(
  Object condition,
  MaskedArray<T> x,
  MaskedArray<T> y, {
  MaskedArray<DTypeTag>? out,
}) {
  _checkSameDTypeOperand(x.dtype, y.dtype, y, 'where');
  return NDArray.scope(() {
    final NDArray<Boolean> condData;
    final NDArray<Boolean>? condMask;
    switch (condition) {
      case MaskedArray<Boolean>():
        condData = condition.data;
        condMask = condition.mask;
      case NDArray<Boolean>():
        condData = condition;
        condMask = null;
      default:
        throw ArgumentError.value(
          condition,
          'condition',
          'Must be a MaskedArray<Boolean> or NDArray<Boolean>.',
        );
    }
    final resultData = ndops.where<T>(condData, x.data, y.data) as NDArray<T>;
    var selectedMask =
        ndops.where<Boolean>(condData, x.mask, y.mask) as NDArray<Boolean>;
    if (condMask != null) {
      selectedMask = ndops.logicalOr(selectedMask, condMask);
    }
    final resultMask = ndops.broadcastTo(selectedMask, resultData.shape).copy();
    return _writeOrCreateMaskedArray<T>(
      resultData,
      resultMask,
      fillValue: x.fillValue,
      out: out,
    );
  });
}

MaskedArray<T> _maConcatenate<T extends DTypeTag>(
  List<MaskedArray<T>> arrays, {
  int axis = 0,
  MaskedArray<DTypeTag>? out,
}) {
  if (arrays.isEmpty) {
    throw ArgumentError.value(arrays, 'arrays', 'Must not be empty.');
  }
  final first = arrays.first;
  for (var i = 1; i < arrays.length; i++) {
    _checkSameDTypeOperand(
      first.dtype,
      arrays[i].dtype,
      arrays[i],
      'concatenate',
    );
  }
  return NDArray.scope(() {
    final dataList = [for (final a in arrays) a.data.asAnySpec];
    final maskList = [for (final a in arrays) a.mask];
    final resultData = ndops.concatenate(dataList, axis: axis) as NDArray<T>;
    final resultMask = ndops.concatenate(maskList, axis: axis);
    return _writeOrCreateMaskedArray<T>(
      resultData,
      resultMask,
      fillValue: first.fillValue,
      out: out,
    );
  });
}
