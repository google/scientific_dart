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

MaskedArray<DTypeTag> _maAdd(MaskedArray self, dynamic other) =>
    _binaryOp(self, other, 'add');
MaskedArray<DTypeTag> _maSubtract(MaskedArray self, dynamic other) =>
    _binaryOp(self, other, 'sub');
MaskedArray<DTypeTag> _maMultiply(MaskedArray self, dynamic other) =>
    _binaryOp(self, other, 'mul');
MaskedArray<DTypeTag> _maDivide(MaskedArray self, dynamic other) =>
    _binaryOp(self, other, 'div', isDivide: true);

MaskedArray<DTypeTag> _binaryOp(
  MaskedArray self,
  dynamic other,
  String opName, {
  bool isDivide = false,
}) {
  return NDArray.scope(() {
    final NDArray<DTypeTag> otherData;
    final NDArray<Boolean>? otherMask;

    if (other is MaskedArray) {
      otherData = other.data;
      otherMask = other.mask;
    } else if (other is NDArray) {
      otherData = other;
      otherMask = null;
    } else {
      otherData = _wrapScalar(other, self.dtype);
      otherMask = null;
    }

    final targetDType = _resolveDType(self.dtype, otherData.dtype);

    // We need to determine the result shape to broadcast masks correctly.
    final broadcastResult = ndops.broadcast(self.data, otherData);
    final resultShape = broadcastResult.shape;

    final NDArray<DTypeTag> divisorData;
    if (isDivide) {
      final combinedMask = otherMask != null
          ? ndops.logicalOr(self.mask, otherMask)
          : self.mask;
      final broadcastedCombinedMask = ndops.broadcastTo(
        combinedMask,
        resultShape,
      );
      final ones = _wrapScalar(1, otherData.dtype);
      divisorData =
          ndops.where(broadcastedCombinedMask, ones, otherData)
              as NDArray<DTypeTag>;
    } else {
      divisorData = otherData;
    }

    final resultData = _dispatchBinary(
      self.data,
      divisorData,
      opName,
      targetDType,
    );

    final broadcastedMaskA = ndops.broadcastTo(self.mask, resultData.shape);
    NDArray<Boolean> resultMask;

    if (otherMask != null) {
      final broadcastedMaskB = ndops.broadcastTo(otherMask, resultData.shape);
      resultMask = ndops.logicalOr(broadcastedMaskA, broadcastedMaskB);
    } else {
      resultMask = broadcastedMaskA.copy();
    }

    if (isDivide) {
      final zeroArray = NDArray.zeros([], otherData.dtype);
      final isZero = ndops.equal(otherData, zeroArray);
      final broadcastedIsZero = ndops.broadcastTo(isZero, resultData.shape);
      final finalMask = ndops.logicalOr(resultMask, broadcastedIsZero);
      return dispatchCreateMaskedArray(
        resultData.detachToParentScope(),
        finalMask.detachToParentScope(),
        fillValue: self.fillValue,
      );
    }

    return dispatchCreateMaskedArray(
      resultData.detachToParentScope(),
      resultMask.detachToParentScope(),
      fillValue: self.fillValue,
    );
  });
}
