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
import 'dart:typed_data';
import '../dtype.dart';
import '../buffer.dart';
import 'compute_engine.dart';
import 'wgsl/wgsl_templates.dart';
import 'wgsl/wgsl_types.dart';

/// Standard binary operation kernel types.
enum BinaryOp {
  /// Addition operation (`a + b`).
  add,

  /// Subtraction operation (`a - b`).
  subtract,

  /// Multiplication operation (`a * b`).
  multiply,

  /// Division operation (`a / b`).
  divide,

  /// Power exponentiation operation (`pow(a, b)`).
  power,

  /// Remainder/modulo operation (`a % b`).
  remainder,

  /// Element-wise maximum operation (`max(a, b)`).
  maximum,

  /// Element-wise minimum operation (`min(a, b)`).
  minimum,

  /// Equality comparison (`a == b`).
  equal,

  /// Inequality comparison (`a != b`).
  notEqual,

  /// Greater-than comparison (`a > b`).
  greater,

  /// Less-than comparison (`a < b`).
  less,

  /// Greater-than-or-equal comparison (`a >= b`).
  greaterEqual,

  /// Less-than-or-equal comparison (`a <= b`).
  lessEqual,
}

/// Standard unary operation kernel types.
enum UnaryOp {
  /// Negation operation (`-x`).
  negate,

  /// Absolute value operation (`abs(x)`).
  abs,

  /// Square root operation (`sqrt(x)`).
  sqrt,

  /// Exponential operation (`exp(x)`).
  exp,

  /// Natural logarithm operation (`log(x)`).
  log,

  /// Sine trigonometric operation (`sin(x)`).
  sin,

  /// Cosine trigonometric operation (`cos(x)`).
  cos,

  /// Tangent trigonometric operation (`tan(x)`).
  tan,

  /// Inverse sine operation (`asin(x)`).
  asin,

  /// Inverse cosine operation (`acos(x)`).
  acos,

  /// Inverse tangent operation (`atan(x)`).
  atan,

  /// Hyperbolic sine operation (`sinh(x)`).
  sinh,

  /// Hyperbolic cosine operation (`cosh(x)`).
  cosh,

  /// Hyperbolic tangent operation (`tanh(x)`).
  tanh,

  /// Floor rounding operation (`floor(x)`).
  floor,

  /// Ceiling rounding operation (`ceil(x)`).
  ceil,

  /// Nearest integer rounding operation (`round(x)`).
  round,
}

/// Execution kernels for GPU compute operations.
extension type const GpuKernels._(Object? _) {
  static bool _isComparisonOp(BinaryOp op) => switch (op) {
    BinaryOp.equal ||
    BinaryOp.notEqual ||
    BinaryOp.greater ||
    BinaryOp.less ||
    BinaryOp.greaterEqual ||
    BinaryOp.lessEqual => true,
    _ => false,
  };

  static bool _isWgslUnarySupported(UnaryOp op, DType dtype) {
    if (dtype == DType.float32) return true;
    if (dtype == DType.int32) {
      return op == UnaryOp.negate || op == UnaryOp.abs;
    }
    if (dtype == DType.uint32) {
      return op == UnaryOp.abs;
    }
    return false;
  }

  /// Packs 40 `u32` words matching WGSL `StridedMetadata`.
  static List<int> packStridedMetadata({
    required int rank,
    required int totalElements,
    required int offsetA,
    required int offsetB,
    required int offsetOut,
    required List<int> shape,
    required List<int> stridesA,
    required List<int> stridesB,
    required List<int> stridesOut,
  }) {
    final words = List<int>.filled(40, 0);
    words[0] = totalElements;
    words[1] = rank;
    words[2] = 0;
    words[3] = 0;
    for (var d = 0; d < 8; d++) {
      words[4 + d] = d < shape.length ? shape[d] : 1;
      words[12 + d] = d < stridesA.length ? (stridesA[d] & 0xFFFFFFFF) : 0;
      words[20 + d] = d < stridesB.length ? (stridesB[d] & 0xFFFFFFFF) : 0;
      words[28 + d] = d < stridesOut.length ? (stridesOut[d] & 0xFFFFFFFF) : 0;
    }
    words[36] = offsetA;
    words[37] = offsetB;
    words[38] = offsetOut;
    words[39] = 0;
    return words;
  }

  static const _packStridedMetadata = packStridedMetadata;

  /// Packs 48 `u32` words matching WGSL `WhereUniforms`.
  static List<int> _packWhereUniforms({
    required int totalElements,
    required int rank,
    required int offsetCond,
    required int offsetX,
    required int offsetY,
    required int offsetOut,
    required List<int> shape,
    required List<int> stridesCond,
    required List<int> stridesX,
    required List<int> stridesY,
    required List<int> stridesOut,
  }) {
    final words = List<int>.filled(48, 0);
    words[0] = totalElements;
    words[1] = rank;
    words[2] = offsetCond;
    words[3] = offsetX;
    words[4] = offsetY;
    words[5] = offsetOut;
    for (var d = 0; d < 8; d++) {
      words[8 + d] = d < shape.length ? shape[d] : 1;
      words[16 + d] = d < stridesCond.length
          ? (stridesCond[d] & 0xFFFFFFFF)
          : 0;
      words[24 + d] = d < stridesX.length ? (stridesX[d] & 0xFFFFFFFF) : 0;
      words[32 + d] = d < stridesY.length ? (stridesY[d] & 0xFFFFFFFF) : 0;
      words[40 + d] = d < stridesOut.length ? (stridesOut[d] & 0xFFFFFFFF) : 0;
    }
    return words;
  }

  /// Executes an elementwise binary kernel with support for multidimensional broadcasting and non-contiguous striding.
  static void executeBinaryOp({
    required BinaryOp op,
    required GpuBuffer srcA,
    required List<int> shapeA,
    required List<int> stridesA,
    required int offsetA,
    required DType dtypeA,
    required GpuBuffer srcB,
    required List<int> shapeB,
    required List<int> stridesB,
    required int offsetB,
    required DType dtypeB,
    required GpuBuffer dst,
    required List<int> outShape,
    required List<int> outStrides,
    required int offsetDst,
    required DType dtypeDst,
  }) {
    final bStridesA = broadcastStrides(shapeA, stridesA, outShape);
    final bStridesB = broadcastStrides(shapeB, stridesB, outShape);
    final totalElements = computeSize(outShape);
    if (totalElements == 0) return;

    if (!srcA.device.backend.isSimulated &&
        dtypeA == dtypeB &&
        dtypeB == dtypeDst &&
        WgslDType.isNativelySupportedStorageDType(dtypeA) &&
        !_isComparisonOp(op) &&
        (op != BinaryOp.power || dtypeA == DType.float32) &&
        dst.address != srcA.address &&
        dst.address != srcB.address) {
      final wgslDType = WgslDType.fromDType(dtypeA);
      final isContiguous =
          isContiguousLayout(shapeA, stridesA) &&
          isContiguousLayout(shapeB, stridesB) &&
          isContiguousLayout(outShape, outStrides) &&
          offsetA == 0 &&
          offsetB == 0 &&
          offsetDst == 0 &&
          areShapesEqual(shapeA, outShape) &&
          areShapesEqual(shapeB, outShape);

      if (isContiguous) {
        final shaderModule = WgslTemplates.elementwiseBinary(
          op: op.name,
          dtype: wgslDType,
          strided: false,
        );
        final dispatch = shaderModule.calculateDispatch1D(totalElements);
        srcA.device.backend.dispatchComputePipeline(
          shaderModule: shaderModule,
          buffers: [srcA, srcB, dst],
          uniforms: [totalElements, 0, 0, 0],
          workgroupsX: dispatch.workgroupsX,
          workgroupsY: dispatch.workgroupsY,
          workgroupsZ: dispatch.workgroupsZ,
        );
        return;
      } else if (outShape.length <= 8) {
        final shaderModule = WgslTemplates.elementwiseBinary(
          op: op.name,
          dtype: wgslDType,
          strided: true,
        );
        final dispatch = shaderModule.calculateDispatch1D(totalElements);
        final uniforms = _packStridedMetadata(
          rank: outShape.length,
          totalElements: totalElements,
          offsetA: offsetA,
          offsetB: offsetB,
          offsetOut: offsetDst,
          shape: outShape,
          stridesA: bStridesA,
          stridesB: bStridesB,
          stridesOut: outStrides,
        );
        srcA.device.backend.dispatchComputePipeline(
          shaderModule: shaderModule,
          buffers: [srcA, srcB, dst],
          uniforms: uniforms,
          workgroupsX: dispatch.workgroupsX,
          workgroupsY: dispatch.workgroupsY,
          workgroupsZ: dispatch.workgroupsZ,
        );
        return;
      }
    }

    final rank = outShape.length;
    final coords = List<int>.filled(rank, 0);

    final isComplex =
        dtypeA == DType.complex64 ||
        dtypeA == DType.complex128 ||
        dtypeB == DType.complex64 ||
        dtypeB == DType.complex128 ||
        dtypeDst == DType.complex64 ||
        dtypeDst == DType.complex128;

    for (var i = 0; i < totalElements; i++) {
      // Calculate source element offsets from multidimensional coordinates
      var elemIndexA = 0;
      var elemIndexB = 0;
      var elemIndexDst = 0;

      for (var d = 0; d < rank; d++) {
        elemIndexA += coords[d] * bStridesA[d];
        elemIndexB += coords[d] * bStridesB[d];
        elemIndexDst += coords[d] * outStrides[d];
      }

      if (isComplex) {
        final valA = readBufferAny(
          srcA,
          dtypeA,
          elemIndexA,
          offsetElements: offsetA,
        );
        final valB = readBufferAny(
          srcB,
          dtypeB,
          elemIndexB,
          offsetElements: offsetB,
        );

        final result = _applyComplexBinary(op, valA, valB);

        writeBufferAny(
          dst,
          dtypeDst,
          elemIndexDst,
          result,
          offsetElements: offsetDst,
        );
      } else {
        final valA = readBufferValue(
          srcA,
          dtypeA,
          elemIndexA,
          offsetElements: offsetA,
        );
        final valB = readBufferValue(
          srcB,
          dtypeB,
          elemIndexB,
          offsetElements: offsetB,
        );

        final result = _applyBinary(op, valA, valB);

        writeBufferValue(
          dst,
          dtypeDst,
          elemIndexDst,
          result,
          offsetElements: offsetDst,
        );
      }

      // Increment multidimensional coordinate
      for (var d = rank - 1; d >= 0; d--) {
        coords[d]++;
        if (coords[d] < outShape[d]) {
          break;
        }
        coords[d] = 0;
      }
    }
  }

  /// Executes an elementwise unary kernel on a tensor.
  static void executeUnaryOp({
    required UnaryOp op,
    required GpuBuffer src,
    required List<int> shape,
    required List<int> strides,
    required int offsetSrc,
    required DType dtypeSrc,
    required GpuBuffer dst,
    required List<int> outStrides,
    required int offsetDst,
    required DType dtypeDst,
  }) {
    final totalElements = computeSize(shape);
    if (totalElements == 0) return;

    if (!src.device.backend.isSimulated &&
        dtypeSrc == dtypeDst &&
        _isWgslUnarySupported(op, dtypeSrc) &&
        dst.address != src.address) {
      final wgslDType = WgslDType.fromDType(dtypeSrc);
      if (isContiguousLayout(shape, strides) &&
          isContiguousLayout(shape, outStrides) &&
          offsetSrc == 0 &&
          offsetDst == 0) {
        final shaderModule = WgslTemplates.elementwiseUnary(
          op: op.name,
          dtype: wgslDType,
          strided: false,
        );
        final dispatch = shaderModule.calculateDispatch1D(totalElements);
        src.device.backend.dispatchComputePipeline(
          shaderModule: shaderModule,
          buffers: [src, dst],
          uniforms: [totalElements, 0, 0, 0],
          workgroupsX: dispatch.workgroupsX,
          workgroupsY: dispatch.workgroupsY,
          workgroupsZ: dispatch.workgroupsZ,
        );
        return;
      } else if (shape.length <= 8) {
        final shaderModule = WgslTemplates.elementwiseUnary(
          op: op.name,
          dtype: wgslDType,
          strided: true,
        );
        final dispatch = shaderModule.calculateDispatch1D(totalElements);
        final uniforms = _packStridedMetadata(
          rank: shape.length,
          totalElements: totalElements,
          offsetA: offsetSrc,
          offsetB: 0,
          offsetOut: offsetDst,
          shape: shape,
          stridesA: strides,
          stridesB: const [],
          stridesOut: outStrides,
        );
        src.device.backend.dispatchComputePipeline(
          shaderModule: shaderModule,
          buffers: [src, dst],
          uniforms: uniforms,
          workgroupsX: dispatch.workgroupsX,
          workgroupsY: dispatch.workgroupsY,
          workgroupsZ: dispatch.workgroupsZ,
        );
        return;
      }
    }

    final rank = shape.length;
    final coords = List<int>.filled(rank, 0);

    final isComplex =
        dtypeSrc == DType.complex64 ||
        dtypeSrc == DType.complex128 ||
        dtypeDst == DType.complex64 ||
        dtypeDst == DType.complex128;

    for (var i = 0; i < totalElements; i++) {
      var elemIndexSrc = 0;
      var elemIndexDst = 0;

      for (var d = 0; d < rank; d++) {
        elemIndexSrc += coords[d] * strides[d];
        elemIndexDst += coords[d] * outStrides[d];
      }

      if (isComplex) {
        final val = readBufferAny(
          src,
          dtypeSrc,
          elemIndexSrc,
          offsetElements: offsetSrc,
        );
        final result = _applyComplexUnary(op, val);

        writeBufferAny(
          dst,
          dtypeDst,
          elemIndexDst,
          result,
          offsetElements: offsetDst,
        );
      } else {
        final val = readBufferValue(
          src,
          dtypeSrc,
          elemIndexSrc,
          offsetElements: offsetSrc,
        );
        final result = _applyUnary(op, val);

        writeBufferValue(
          dst,
          dtypeDst,
          elemIndexDst,
          result,
          offsetElements: offsetDst,
        );
      }

      for (var d = rank - 1; d >= 0; d--) {
        coords[d]++;
        if (coords[d] < shape[d]) {
          break;
        }
        coords[d] = 0;
      }
    }
  }

  /// Executes a reduction kernel across specified axes or the entire tensor.
  static void executeReduction({
    required String op, // 'sum', 'mean', 'prod', 'min', 'max'
    required GpuBuffer src,
    required List<int> shape,
    required List<int> strides,
    required int offsetSrc,
    required DType dtypeSrc,
    required GpuBuffer dst,
    required List<int> outShape,
    required List<int> outStrides,
    required int offsetDst,
    required DType dtypeDst,
    int? axis,
  }) {
    final isComplex =
        dtypeSrc == DType.complex64 || dtypeSrc == DType.complex128;

    if (axis == null) {
      // Full reduction to scalar
      final totalElements = computeSize(shape);
      if (totalElements == 0) {
        if (isComplex) {
          writeBufferAny(
            dst,
            dtypeDst,
            0,
            Complex(0.0, 0.0),
            offsetElements: offsetDst,
          );
        } else {
          writeBufferValue(dst, dtypeDst, 0, 0.0, offsetElements: offsetDst);
        }
        return;
      }

      if (!src.device.backend.isSimulated &&
          dtypeSrc == dtypeDst &&
          WgslDType.isNativelySupportedStorageDType(dtypeSrc) &&
          (op != 'mean' || dtypeSrc == DType.float32) &&
          dst.address != src.address) {
        final wgslDType = WgslDType.fromDType(dtypeSrc);
        if (isContiguousLayout(shape, strides) &&
            offsetSrc == 0 &&
            offsetDst == 0) {
          final shaderModule = WgslTemplates.treeReduction(
            op: op,
            dtype: wgslDType,
            strided: false,
          );
          src.device.backend.dispatchComputePipeline(
            shaderModule: shaderModule,
            buffers: [src, dst],
            uniforms: [totalElements, 0, 0, 0],
            workgroupsX: 1,
            workgroupsY: 1,
            workgroupsZ: 1,
          );
          return;
        } else if (shape.length <= 8) {
          final shaderModule = WgslTemplates.treeReduction(
            op: op,
            dtype: wgslDType,
            strided: true,
          );
          final uniforms = _packStridedMetadata(
            rank: shape.length,
            totalElements: totalElements,
            offsetA: offsetSrc,
            offsetB: 0,
            offsetOut: offsetDst,
            shape: shape,
            stridesA: strides,
            stridesB: const [],
            stridesOut: const [],
          );
          src.device.backend.dispatchComputePipeline(
            shaderModule: shaderModule,
            buffers: [src, dst],
            uniforms: uniforms,
            workgroupsX: 1,
            workgroupsY: 1,
            workgroupsZ: 1,
          );
          return;
        }
      }

      final rank = shape.length;
      final coords = List<int>.filled(rank, 0);

      if (isComplex) {
        var accum = _initialComplexReductionValue(op);

        for (var i = 0; i < totalElements; i++) {
          var elemIndex = 0;
          for (var d = 0; d < rank; d++) {
            elemIndex += coords[d] * strides[d];
          }

          final rawVal = readBufferAny(
            src,
            dtypeSrc,
            elemIndex,
            offsetElements: offsetSrc,
          );
          accum = _combineComplexReduction(op, accum, _toComplex(rawVal));

          for (var d = rank - 1; d >= 0; d--) {
            coords[d]++;
            if (coords[d] < shape[d]) {
              break;
            }
            coords[d] = 0;
          }
        }

        if (op == 'mean') {
          accum = Complex(
            accum.real / totalElements,
            accum.imag / totalElements,
          );
        }

        writeBufferAny(dst, dtypeDst, 0, accum, offsetElements: offsetDst);
      } else {
        var accum = _initialReductionValue(op);

        for (var i = 0; i < totalElements; i++) {
          var elemIndex = 0;
          for (var d = 0; d < rank; d++) {
            elemIndex += coords[d] * strides[d];
          }

          final val = readBufferValue(
            src,
            dtypeSrc,
            elemIndex,
            offsetElements: offsetSrc,
          );
          accum = _combineReduction(op, accum, val, i);

          for (var d = rank - 1; d >= 0; d--) {
            coords[d]++;
            if (coords[d] < shape[d]) {
              break;
            }
            coords[d] = 0;
          }
        }

        if (op == 'mean') {
          accum = accum / totalElements;
        }

        writeBufferValue(dst, dtypeDst, 0, accum, offsetElements: offsetDst);
      }
    } else {
      // Reduction along a single axis
      final normAxis = axis < 0 ? axis + shape.length : axis;
      final axisSize = shape[normAxis];
      final totalOut = computeSize(outShape);
      if (totalOut == 0) return;

      if (!src.device.backend.isSimulated &&
          dtypeSrc == dtypeDst &&
          WgslDType.isNativelySupportedStorageDType(dtypeSrc) &&
          (op != 'mean' || dtypeSrc == DType.float32) &&
          outShape.length <= 8 &&
          dst.address != src.address) {
        final nonAxisStrides = <int>[];
        if (outShape.length == shape.length) {
          // keepDims == true: outShape has 1 at normAxis
          for (var d = 0; d < shape.length; d++) {
            nonAxisStrides.add(d == normAxis ? 0 : strides[d]);
          }
        } else {
          for (var d = 0; d < shape.length; d++) {
            if (d != normAxis) nonAxisStrides.add(strides[d]);
          }
        }
        final shaderModule = WgslTemplates.axisReduction(
          op: op,
          dtype: WgslDType.fromDType(dtypeSrc),
        );
        final dispatch = shaderModule.calculateDispatch1D(totalOut);
        final uniforms = _packStridedMetadata(
          rank: outShape.length,
          totalElements: totalOut,
          offsetA: offsetSrc,
          offsetB: 0,
          offsetOut: offsetDst,
          shape: outShape,
          stridesA: nonAxisStrides,
          stridesB: [axisSize, strides[normAxis]],
          stridesOut: outStrides,
        );
        src.device.backend.dispatchComputePipeline(
          shaderModule: shaderModule,
          buffers: [src, dst],
          uniforms: uniforms,
          workgroupsX: dispatch.workgroupsX,
          workgroupsY: dispatch.workgroupsY,
          workgroupsZ: dispatch.workgroupsZ,
        );
        return;
      }

      final outRank = outShape.length;
      final outCoords = List<int>.filled(outRank, 0);

      for (var outIndex = 0; outIndex < totalOut; outIndex++) {
        var dstElemIndex = 0;
        for (var d = 0; d < outRank; d++) {
          dstElemIndex += outCoords[d] * outStrides[d];
        }

        if (isComplex) {
          var accum = _initialComplexReductionValue(op);

          for (var a = 0; a < axisSize; a++) {
            var srcElemIndex = 0;
            var inDim = 0;
            for (var d = 0; d < shape.length; d++) {
              if (d == normAxis) {
                srcElemIndex += a * strides[d];
              } else {
                srcElemIndex += outCoords[inDim] * strides[d];
                inDim++;
              }
            }

            final rawVal = readBufferAny(
              src,
              dtypeSrc,
              srcElemIndex,
              offsetElements: offsetSrc,
            );
            accum = _combineComplexReduction(op, accum, _toComplex(rawVal));
          }

          if (op == 'mean') {
            accum = Complex(accum.real / axisSize, accum.imag / axisSize);
          }

          writeBufferAny(
            dst,
            dtypeDst,
            dstElemIndex,
            accum,
            offsetElements: offsetDst,
          );
        } else {
          var accum = _initialReductionValue(op);

          for (var a = 0; a < axisSize; a++) {
            // Reconstruct input coords from output coords + axis index
            var srcElemIndex = 0;
            var inDim = 0;
            for (var d = 0; d < shape.length; d++) {
              if (d == normAxis) {
                srcElemIndex += a * strides[d];
              } else {
                srcElemIndex += outCoords[inDim] * strides[d];
                inDim++;
              }
            }

            final val = readBufferValue(
              src,
              dtypeSrc,
              srcElemIndex,
              offsetElements: offsetSrc,
            );
            accum = _combineReduction(op, accum, val, a);
          }

          if (op == 'mean') {
            accum = accum / axisSize;
          }

          writeBufferValue(
            dst,
            dtypeDst,
            dstElemIndex,
            accum,
            offsetElements: offsetDst,
          );
        }

        for (var d = outRank - 1; d >= 0; d--) {
          outCoords[d]++;
          if (outCoords[d] < outShape[d]) {
            break;
          }
          outCoords[d] = 0;
        }
      }
    }
  }

  /// Executes tiled 2D or batched N-D matrix multiplication.
  static void executeMatmul({
    required GpuBuffer srcA,
    required List<int> shapeA,
    required List<int> stridesA,
    required int offsetA,
    required DType dtypeA,
    required GpuBuffer srcB,
    required List<int> shapeB,
    required List<int> stridesB,
    required int offsetB,
    required DType dtypeB,
    required GpuBuffer dst,
    required List<int> outShape,
    required List<int> outStrides,
    required int offsetDst,
    required DType dtypeDst,
  }) {
    final rankA = shapeA.length;
    final rankB = shapeB.length;
    final isComplex =
        dtypeA == DType.complex64 ||
        dtypeA == DType.complex128 ||
        dtypeB == DType.complex64 ||
        dtypeB == DType.complex128 ||
        dtypeDst == DType.complex64 ||
        dtypeDst == DType.complex128;

    if (rankA == 2 && rankB == 2) {
      final M = shapeA[0];
      final K = shapeA[1];
      final N = shapeB[1];
      if (M == 0 || N == 0) return;

      if (!srcA.device.backend.isSimulated &&
          K > 0 &&
          dtypeA == DType.float32 &&
          dtypeB == DType.float32 &&
          dtypeDst == DType.float32 &&
          dst.address != srcA.address &&
          dst.address != srcB.address &&
          stridesA[0] >= 0 &&
          stridesA[1] >= 0 &&
          stridesB[0] >= 0 &&
          stridesB[1] >= 0 &&
          outStrides[0] >= 0 &&
          outStrides[1] >= 0) {
        final shaderModule = WgslTemplates.tiledMatmul(
          tileSize: 16,
          dtype: WgslDType.float32,
        );

        final alphaBits = ByteData(4)..setFloat32(0, 1.0, Endian.little);
        final betaBits = ByteData(4)..setFloat32(0, 0.0, Endian.little);

        final uniforms = <int>[
          M,
          N,
          K,
          stridesA[0],
          stridesA[1],
          stridesB[0],
          stridesB[1],
          outStrides[0],
          outStrides[1],
          offsetA,
          offsetB,
          offsetDst,
          alphaBits.getUint32(0, Endian.little),
          betaBits.getUint32(0, Endian.little),
          0,
          0,
        ];

        srcA.device.backend.dispatchComputePipeline(
          shaderModule: shaderModule,
          buffers: [srcA, srcB, dst],
          uniforms: uniforms,
          workgroupsX: (N + 15) ~/ 16,
          workgroupsY: (M + 15) ~/ 16,
          workgroupsZ: 1,
        );
        return;
      }

      for (var m = 0; m < M; m++) {
        for (var n = 0; n < N; n++) {
          final indexDst = m * outStrides[0] + n * outStrides[1];
          if (isComplex) {
            var sumReal = 0.0;
            var sumImag = 0.0;
            for (var k = 0; k < K; k++) {
              final indexA = m * stridesA[0] + k * stridesA[1];
              final indexB = k * stridesB[0] + n * stridesB[1];
              final a = _toComplex(
                readBufferAny(srcA, dtypeA, indexA, offsetElements: offsetA),
              );
              final b = _toComplex(
                readBufferAny(srcB, dtypeB, indexB, offsetElements: offsetB),
              );
              sumReal += a.real * b.real - a.imag * b.imag;
              sumImag += a.real * b.imag + a.imag * b.real;
            }
            writeBufferAny(
              dst,
              dtypeDst,
              indexDst,
              Complex(sumReal, sumImag),
              offsetElements: offsetDst,
            );
          } else {
            var sum = 0.0;
            for (var k = 0; k < K; k++) {
              final indexA = m * stridesA[0] + k * stridesA[1];
              final indexB = k * stridesB[0] + n * stridesB[1];
              final a = readBufferValue(
                srcA,
                dtypeA,
                indexA,
                offsetElements: offsetA,
              );
              final b = readBufferValue(
                srcB,
                dtypeB,
                indexB,
                offsetElements: offsetB,
              );
              sum += a * b;
            }
            writeBufferValue(
              dst,
              dtypeDst,
              indexDst,
              sum,
              offsetElements: offsetDst,
            );
          }
        }
      }
    } else if (rankA == 1 && rankB == 1) {
      // 1D dot product
      final K = shapeA[0];
      if (isComplex) {
        var sumReal = 0.0;
        var sumImag = 0.0;
        for (var k = 0; k < K; k++) {
          final a = _toComplex(
            readBufferAny(
              srcA,
              dtypeA,
              k * stridesA[0],
              offsetElements: offsetA,
            ),
          );
          final b = _toComplex(
            readBufferAny(
              srcB,
              dtypeB,
              k * stridesB[0],
              offsetElements: offsetB,
            ),
          );
          sumReal += a.real * b.real - a.imag * b.imag;
          sumImag += a.real * b.imag + a.imag * b.real;
        }
        writeBufferAny(
          dst,
          dtypeDst,
          0,
          Complex(sumReal, sumImag),
          offsetElements: offsetDst,
        );
      } else {
        var sum = 0.0;
        for (var k = 0; k < K; k++) {
          final a = readBufferValue(
            srcA,
            dtypeA,
            k * stridesA[0],
            offsetElements: offsetA,
          );
          final b = readBufferValue(
            srcB,
            dtypeB,
            k * stridesB[0],
            offsetElements: offsetB,
          );
          sum += a * b;
        }
        writeBufferValue(dst, dtypeDst, 0, sum, offsetElements: offsetDst);
      }
    } else {
      // Batched N-D matrix multiplication
      final batchShapeA = shapeA.sublist(0, rankA - 2);
      final batchShapeB = shapeB.sublist(0, rankB - 2);
      final batchOutShape = broadcastShapes(batchShapeA, batchShapeB);

      final M = shapeA[rankA - 2];
      final K = shapeA[rankA - 1];
      final N = shapeB[rankB - 1];

      final batchSize = computeSize(batchOutShape);
      final batchRank = batchOutShape.length;
      final batchCoords = List<int>.filled(batchRank, 0);

      final batchStridesA = broadcastStrides(
        batchShapeA,
        stridesA.sublist(0, rankA - 2),
        batchOutShape,
      );
      final batchStridesB = broadcastStrides(
        batchShapeB,
        stridesB.sublist(0, rankB - 2),
        batchOutShape,
      );
      final batchStridesDst = outStrides.sublist(0, outStrides.length - 2);

      for (var b = 0; b < batchSize; b++) {
        var baseIndexA = 0;
        var baseIndexB = 0;
        var baseIndexDst = 0;

        for (var d = 0; d < batchRank; d++) {
          baseIndexA += batchCoords[d] * batchStridesA[d];
          baseIndexB += batchCoords[d] * batchStridesB[d];
          baseIndexDst += batchCoords[d] * batchStridesDst[d];
        }

        for (var m = 0; m < M; m++) {
          for (var n = 0; n < N; n++) {
            final indexDst =
                baseIndexDst +
                m * outStrides[outStrides.length - 2] +
                n * outStrides[outStrides.length - 1];
            if (isComplex) {
              var sumReal = 0.0;
              var sumImag = 0.0;
              for (var k = 0; k < K; k++) {
                final indexA =
                    baseIndexA +
                    m * stridesA[rankA - 2] +
                    k * stridesA[rankA - 1];
                final indexB =
                    baseIndexB +
                    k * stridesB[rankB - 2] +
                    n * stridesB[rankB - 1];
                final a = _toComplex(
                  readBufferAny(srcA, dtypeA, indexA, offsetElements: offsetA),
                );
                final b = _toComplex(
                  readBufferAny(srcB, dtypeB, indexB, offsetElements: offsetB),
                );
                sumReal += a.real * b.real - a.imag * b.imag;
                sumImag += a.real * b.imag + a.imag * b.real;
              }
              writeBufferAny(
                dst,
                dtypeDst,
                indexDst,
                Complex(sumReal, sumImag),
                offsetElements: offsetDst,
              );
            } else {
              var sum = 0.0;
              for (var k = 0; k < K; k++) {
                final indexA =
                    baseIndexA +
                    m * stridesA[rankA - 2] +
                    k * stridesA[rankA - 1];
                final indexB =
                    baseIndexB +
                    k * stridesB[rankB - 2] +
                    n * stridesB[rankB - 1];
                final a = readBufferValue(
                  srcA,
                  dtypeA,
                  indexA,
                  offsetElements: offsetA,
                );
                final b = readBufferValue(
                  srcB,
                  dtypeB,
                  indexB,
                  offsetElements: offsetB,
                );
                sum += a * b;
              }
              writeBufferValue(
                dst,
                dtypeDst,
                indexDst,
                sum,
                offsetElements: offsetDst,
              );
            }
          }
        }

        for (var d = batchRank - 1; d >= 0; d--) {
          batchCoords[d]++;
          if (batchCoords[d] < batchOutShape[d]) {
            break;
          }
          batchCoords[d] = 0;
        }
      }
    }
  }

  /// Copies strided data from [src] to contiguous [dst].
  static void copyStrided({
    required GpuBuffer src,
    required List<int> shape,
    required List<int> strides,
    required int offsetSrc,
    required DType dtypeSrc,
    required GpuBuffer dst,
    required List<int> outStrides,
    required int offsetDst,
    required DType dtypeDst,
  }) {
    final totalElements = computeSize(shape);
    if (totalElements == 0) return;

    if (!src.device.backend.isSimulated &&
        dtypeSrc == dtypeDst &&
        WgslDType.isNativelySupportedStorageDType(dtypeSrc) &&
        shape.length <= 8 &&
        dst.address != src.address) {
      final shaderModule = WgslTemplates.tileKernel(
        dtype: WgslDType.fromDType(dtypeSrc),
      );
      final dispatch = shaderModule.calculateDispatch1D(totalElements);
      final uniforms = _packStridedMetadata(
        rank: shape.length,
        totalElements: totalElements,
        offsetA: offsetSrc,
        offsetB: 0,
        offsetOut: offsetDst,
        shape: shape,
        stridesA: strides,
        stridesB: shape,
        stridesOut: outStrides,
      );
      src.device.backend.dispatchComputePipeline(
        shaderModule: shaderModule,
        buffers: [src, dst],
        uniforms: uniforms,
        workgroupsX: dispatch.workgroupsX,
        workgroupsY: dispatch.workgroupsY,
        workgroupsZ: dispatch.workgroupsZ,
      );
      return;
    }

    final rank = shape.length;
    final coords = List<int>.filled(rank, 0);

    for (var i = 0; i < totalElements; i++) {
      var elemIndexSrc = 0;
      var elemIndexDst = 0;

      for (var d = 0; d < rank; d++) {
        elemIndexSrc += coords[d] * strides[d];
        elemIndexDst += coords[d] * outStrides[d];
      }

      final val = readBufferAny(
        src,
        dtypeSrc,
        elemIndexSrc,
        offsetElements: offsetSrc,
      );
      writeBufferAny(
        dst,
        dtypeDst,
        elemIndexDst,
        val,
        offsetElements: offsetDst,
      );

      for (var d = rank - 1; d >= 0; d--) {
        coords[d]++;
        if (coords[d] < shape[d]) {
          break;
        }
        coords[d] = 0;
      }
    }
  }

  static Complex _toComplex(Object? v) {
    if (v is Complex) return v;
    if (v is num) return Complex(v.toDouble(), 0.0);
    if (v is bool) return Complex(v ? 1.0 : 0.0, 0.0);
    return Complex(0.0, 0.0);
  }

  static Object _applyComplexBinary(BinaryOp op, Object? rawA, Object? rawB) {
    final a = _toComplex(rawA);
    final b = _toComplex(rawB);
    switch (op) {
      case BinaryOp.add:
        return Complex(a.real + b.real, a.imag + b.imag);
      case BinaryOp.subtract:
        return Complex(a.real - b.real, a.imag - b.imag);
      case BinaryOp.multiply:
        return Complex(
          a.real * b.real - a.imag * b.imag,
          a.real * b.imag + a.imag * b.real,
        );
      case BinaryOp.divide:
        final denom = b.real * b.real + b.imag * b.imag;
        if (denom == 0) return Complex(double.nan, double.nan);
        return Complex(
          (a.real * b.real + a.imag * b.imag) / denom,
          (a.imag * b.real - a.real * b.imag) / denom,
        );
      case BinaryOp.equal:
        return (a.real == b.real && a.imag == b.imag) ? 1 : 0;
      case BinaryOp.notEqual:
        return (a.real != b.real || a.imag != b.imag) ? 1 : 0;
      default:
        return Complex(a.real, a.imag);
    }
  }

  static Object _applyComplexUnary(UnaryOp op, Object? raw) {
    final c = _toComplex(raw);
    switch (op) {
      case UnaryOp.negate:
        return Complex(-c.real, -c.imag);
      case UnaryOp.abs:
        return math.sqrt(c.real * c.real + c.imag * c.imag);
      case UnaryOp.exp:
        final expR = math.exp(c.real);
        return Complex(expR * math.cos(c.imag), expR * math.sin(c.imag));
      default:
        return c;
    }
  }

  static Complex _initialComplexReductionValue(String op) {
    switch (op) {
      case 'sum':
      case 'mean':
        return Complex(0.0, 0.0);
      case 'prod':
        return Complex(1.0, 0.0);
      case 'min':
        return Complex(double.infinity, 0.0);
      case 'max':
        return Complex(double.negativeInfinity, 0.0);
      default:
        return Complex(0.0, 0.0);
    }
  }

  static Complex _combineComplexReduction(
    String op,
    Complex current,
    Complex value,
  ) {
    switch (op) {
      case 'sum':
      case 'mean':
        return Complex(current.real + value.real, current.imag + value.imag);
      case 'prod':
        return Complex(
          current.real * value.real - current.imag * value.imag,
          current.real * value.imag + current.imag * value.real,
        );
      case 'min':
        final absVal = math.sqrt(
          value.real * value.real + value.imag * value.imag,
        );
        final absCur = math.sqrt(
          current.real * current.real + current.imag * current.imag,
        );
        return absVal < absCur ? value : current;
      case 'max':
        final absVal = math.sqrt(
          value.real * value.real + value.imag * value.imag,
        );
        final absCur = math.sqrt(
          current.real * current.real + current.imag * current.imag,
        );
        return absVal > absCur ? value : current;
      default:
        return current;
    }
  }

  static double _applyBinary(BinaryOp op, double a, double b) {
    switch (op) {
      case BinaryOp.add:
        return a + b;
      case BinaryOp.subtract:
        return a - b;
      case BinaryOp.multiply:
        return a * b;
      case BinaryOp.divide:
        return a / b;
      case BinaryOp.power:
        return math.pow(a, b).toDouble();
      case BinaryOp.remainder:
        return a % b;
      case BinaryOp.maximum:
        return math.max(a, b);
      case BinaryOp.minimum:
        return math.min(a, b);
      case BinaryOp.equal:
        return (a == b) ? 1.0 : 0.0;
      case BinaryOp.notEqual:
        return (a != b) ? 1.0 : 0.0;
      case BinaryOp.greater:
        return (a > b) ? 1.0 : 0.0;
      case BinaryOp.less:
        return (a < b) ? 1.0 : 0.0;
      case BinaryOp.greaterEqual:
        return (a >= b) ? 1.0 : 0.0;
      case BinaryOp.lessEqual:
        return (a <= b) ? 1.0 : 0.0;
    }
  }

  static double _applyUnary(UnaryOp op, double v) {
    switch (op) {
      case UnaryOp.negate:
        return -v;
      case UnaryOp.abs:
        return v.abs();
      case UnaryOp.sqrt:
        return math.sqrt(v);
      case UnaryOp.exp:
        return math.exp(v);
      case UnaryOp.log:
        return math.log(v);
      case UnaryOp.sin:
        return math.sin(v);
      case UnaryOp.cos:
        return math.cos(v);
      case UnaryOp.tan:
        return math.tan(v);
      case UnaryOp.asin:
        return math.asin(v);
      case UnaryOp.acos:
        return math.acos(v);
      case UnaryOp.atan:
        return math.atan(v);
      case UnaryOp.sinh:
        return (math.exp(v) - math.exp(-v)) / 2.0;
      case UnaryOp.cosh:
        return (math.exp(v) + math.exp(-v)) / 2.0;
      case UnaryOp.tanh:
        final ep = math.exp(v);
        final em = math.exp(-v);
        return (ep - em) / (ep + em);
      case UnaryOp.floor:
        return v.floorToDouble();
      case UnaryOp.ceil:
        return v.ceilToDouble();
      case UnaryOp.round:
        return v.roundToDouble();
    }
  }

  static double _initialReductionValue(String op) {
    switch (op) {
      case 'sum':
      case 'mean':
        return 0.0;
      case 'prod':
        return 1.0;
      case 'min':
        return double.infinity;
      case 'max':
        return double.negativeInfinity;
      default:
        return 0.0;
    }
  }

  static double _combineReduction(
    String op,
    double current,
    double value,
    int index,
  ) {
    switch (op) {
      case 'sum':
      case 'mean':
        return current + value;
      case 'prod':
        return current * value;
      case 'min':
        return math.min(current, value);
      case 'max':
        return math.max(current, value);
      default:
        return current;
    }
  }

  /// Dispatches conditional ternary selection (where).
  static void executeWhere({
    required GpuBuffer cond,
    required List<int> shapeCond,
    required List<int> stridesCond,
    required int offsetCond,
    required GpuBuffer srcX,
    required List<int> shapeX,
    required List<int> stridesX,
    required int offsetX,
    required DType dtypeX,
    required GpuBuffer srcY,
    required List<int> shapeY,
    required List<int> stridesY,
    required int offsetY,
    required DType dtypeY,
    required GpuBuffer dst,
    required List<int> outShape,
    required List<int> outStrides,
    required int offsetDst,
    required DType dtypeDst,
  }) {
    final totalElements = computeSize(outShape);
    if (totalElements == 0) return;

    final rank = outShape.length;
    final bStridesCond = broadcastStrides(shapeCond, stridesCond, outShape);
    final bStridesX = broadcastStrides(shapeX, stridesX, outShape);
    final bStridesY = broadcastStrides(shapeY, stridesY, outShape);

    if (!srcX.device.backend.isSimulated &&
        dtypeX == dtypeY &&
        dtypeY == dtypeDst &&
        WgslDType.isNativelySupportedStorageDType(dtypeDst) &&
        rank <= 8 &&
        dst.address != srcX.address &&
        dst.address != srcY.address &&
        dst.address != cond.address) {
      final shaderModule = WgslTemplates.whereKernel(
        dtype: WgslDType.fromDType(dtypeDst),
      );
      final dispatch = shaderModule.calculateDispatch1D(totalElements);
      final uniforms = _packWhereUniforms(
        totalElements: totalElements,
        rank: rank,
        offsetCond: offsetCond,
        offsetX: offsetX,
        offsetY: offsetY,
        offsetOut: offsetDst,
        shape: outShape,
        stridesCond: bStridesCond,
        stridesX: bStridesX,
        stridesY: bStridesY,
        stridesOut: outStrides,
      );
      srcX.device.backend.dispatchComputePipeline(
        shaderModule: shaderModule,
        buffers: [cond, srcX, srcY, dst],
        uniforms: uniforms,
        workgroupsX: dispatch.workgroupsX,
        workgroupsY: dispatch.workgroupsY,
        workgroupsZ: dispatch.workgroupsZ,
      );
      return;
    }

    final coords = List<int>.filled(rank, 0);

    for (var i = 0; i < totalElements; i++) {
      var elemIndexCond = 0;
      var elemIndexX = 0;
      var elemIndexY = 0;
      var elemIndexDst = 0;

      for (var d = 0; d < rank; d++) {
        elemIndexCond += coords[d] * bStridesCond[d];
        elemIndexX += coords[d] * bStridesX[d];
        elemIndexY += coords[d] * bStridesY[d];
        elemIndexDst += coords[d] * outStrides[d];
      }

      final cVal = readBufferValue(
        cond,
        DType.boolean,
        elemIndexCond,
        offsetElements: offsetCond,
      );
      final isTrue = cVal != 0.0;

      final val = isTrue
          ? readBufferAny(srcX, dtypeX, elemIndexX, offsetElements: offsetX)
          : readBufferAny(srcY, dtypeY, elemIndexY, offsetElements: offsetY);

      writeBufferAny(
        dst,
        dtypeDst,
        elemIndexDst,
        val,
        offsetElements: offsetDst,
      );

      for (var d = rank - 1; d >= 0; d--) {
        coords[d]++;
        if (coords[d] < outShape[d]) break;
        coords[d] = 0;
      }
    }
  }

  /// Extracts elements along an axis according to coordinates in [indices].
  static void executeTakeAlongAxis({
    required GpuBuffer src,
    required List<int> shapeSrc,
    required List<int> stridesSrc,
    required int offsetSrc,
    required DType dtypeSrc,
    required GpuBuffer indices,
    required List<int> shapeIndices,
    required List<int> stridesIndices,
    required int offsetIndices,
    required DType dtypeIndices,
    required GpuBuffer dst,
    required List<int> outShape,
    required List<int> outStrides,
    required int offsetDst,
    required DType dtypeDst,
    required int axis,
  }) {
    final totalElements = computeSize(outShape);
    final rank = outShape.length;
    final normAxis = axis < 0 ? axis + rank : axis;
    final axisLen = shapeSrc[normAxis];
    final coords = List<int>.filled(rank, 0);

    for (var i = 0; i < totalElements; i++) {
      var elemIndexIndices = 0;
      var elemIndexDst = 0;

      for (var d = 0; d < rank; d++) {
        elemIndexIndices += coords[d] * stridesIndices[d];
        elemIndexDst += coords[d] * outStrides[d];
      }

      final indexValNum = readBufferValue(
        indices,
        dtypeIndices,
        elemIndexIndices,
        offsetElements: offsetIndices,
      );
      var k = indexValNum.toInt();
      if (k < 0) k += axisLen;
      if (k < 0 || k >= axisLen) {
        throw IndexError.withLength(
          k,
          axisLen,
          name: 'take_along_axis index out of bounds',
        );
      }

      var elemIndexSrc = 0;
      for (var d = 0; d < rank; d++) {
        final c = (d == normAxis) ? k : coords[d];
        elemIndexSrc += c * stridesSrc[d];
      }

      final val = readBufferAny(
        src,
        dtypeSrc,
        elemIndexSrc,
        offsetElements: offsetSrc,
      );
      writeBufferAny(
        dst,
        dtypeDst,
        elemIndexDst,
        val,
        offsetElements: offsetDst,
      );

      for (var d = rank - 1; d >= 0; d--) {
        coords[d]++;
        if (coords[d] < outShape[d]) break;
        coords[d] = 0;
      }
    }
  }

  /// Inserts [values] into [arr] along an axis according to [indices].
  static void executePutAlongAxis({
    required GpuBuffer arr,
    required List<int> shapeArr,
    required List<int> stridesArr,
    required int offsetArr,
    required DType dtypeArr,
    required GpuBuffer indices,
    required List<int> shapeIndices,
    required List<int> stridesIndices,
    required int offsetIndices,
    required DType dtypeIndices,
    required GpuBuffer values,
    required List<int> shapeVal,
    required List<int> stridesVal,
    required int offsetVal,
    required DType dtypeVal,
    required int axis,
  }) {
    final totalElements = computeSize(shapeIndices);
    final rank = shapeArr.length;
    final normAxis = axis < 0 ? axis + rank : axis;
    final axisLen = shapeArr[normAxis];
    final bStridesVal = broadcastStrides(shapeVal, stridesVal, shapeIndices);
    final coords = List<int>.filled(rank, 0);

    for (var i = 0; i < totalElements; i++) {
      var elemIndexIndices = 0;
      var elemIndexVal = 0;

      for (var d = 0; d < rank; d++) {
        elemIndexIndices += coords[d] * stridesIndices[d];
        elemIndexVal += coords[d] * bStridesVal[d];
      }

      final indexValNum = readBufferValue(
        indices,
        dtypeIndices,
        elemIndexIndices,
        offsetElements: offsetIndices,
      );
      var k = indexValNum.toInt();
      if (k < 0) k += axisLen;
      if (k < 0 || k >= axisLen) {
        throw IndexError.withLength(
          k,
          axisLen,
          name: 'put_along_axis index out of bounds',
        );
      }

      var elemIndexArr = 0;
      for (var d = 0; d < rank; d++) {
        final c = (d == normAxis) ? k : coords[d];
        elemIndexArr += c * stridesArr[d];
      }

      final val = readBufferAny(
        values,
        dtypeVal,
        elemIndexVal,
        offsetElements: offsetVal,
      );
      writeBufferAny(
        arr,
        dtypeArr,
        elemIndexArr,
        val,
        offsetElements: offsetArr,
      );

      for (var d = rank - 1; d >= 0; d--) {
        coords[d]++;
        if (coords[d] < shapeIndices[d]) break;
        coords[d] = 0;
      }
    }
  }

  /// Concatenates a list of tensors along [axis].
  static void executeConcatenate({
    required List<GpuBuffer> srcBuffers,
    required List<List<int>> srcShapes,
    required List<List<int>> srcStrides,
    required List<int> srcOffsets,
    required List<DType> srcDtypes,
    required GpuBuffer dst,
    required List<int> outShape,
    required List<int> outStrides,
    required int offsetDst,
    required DType dtypeDst,
    required int axis,
  }) {
    final rank = outShape.length;
    final normAxis = axis < 0 ? axis + rank : axis;
    var axisOffset = 0;

    for (var arrayIndex = 0; arrayIndex < srcBuffers.length; arrayIndex++) {
      final src = srcBuffers[arrayIndex];
      final shape = srcShapes[arrayIndex];
      final strides = srcStrides[arrayIndex];
      final offset = srcOffsets[arrayIndex];
      final dtype = srcDtypes[arrayIndex];
      final total = computeSize(shape);
      final coords = List<int>.filled(rank, 0);

      for (var i = 0; i < total; i++) {
        var elemIndexSrc = 0;
        var elemIndexDst = 0;

        for (var d = 0; d < rank; d++) {
          elemIndexSrc += coords[d] * strides[d];
          final outCoord = (d == normAxis) ? coords[d] + axisOffset : coords[d];
          elemIndexDst += outCoord * outStrides[d];
        }

        final val = readBufferAny(
          src,
          dtype,
          elemIndexSrc,
          offsetElements: offset,
        );
        writeBufferAny(
          dst,
          dtypeDst,
          elemIndexDst,
          val,
          offsetElements: offsetDst,
        );

        for (var d = rank - 1; d >= 0; d--) {
          coords[d]++;
          if (coords[d] < shape[d]) break;
          coords[d] = 0;
        }
      }

      axisOffset += shape[normAxis];
    }
  }

  /// Pads a tensor with specified padding widths.
  static void executePad({
    required GpuBuffer src,
    required List<int> shapeSrc,
    required List<int> stridesSrc,
    required int offsetSrc,
    required DType dtypeSrc,
    required GpuBuffer dst,
    required List<int> outShape,
    required List<int> outStrides,
    required int offsetDst,
    required DType dtypeDst,
    required List<List<int>> padWidth,
    required Object? constantValue,
  }) {
    final totalElements = computeSize(outShape);
    final rank = outShape.length;
    final coords = List<int>.filled(rank, 0);

    for (var i = 0; i < totalElements; i++) {
      var elemIndexDst = 0;
      var inBounds = true;
      var elemIndexSrc = 0;

      for (var d = 0; d < rank; d++) {
        elemIndexDst += coords[d] * outStrides[d];
        final srcCoord = coords[d] - padWidth[d][0];
        if (srcCoord < 0 || srcCoord >= shapeSrc[d]) {
          inBounds = false;
        } else {
          elemIndexSrc += srcCoord * stridesSrc[d];
        }
      }

      if (inBounds) {
        final val = readBufferAny(
          src,
          dtypeSrc,
          elemIndexSrc,
          offsetElements: offsetSrc,
        );
        writeBufferAny(
          dst,
          dtypeDst,
          elemIndexDst,
          val,
          offsetElements: offsetDst,
        );
      } else {
        writeBufferAny(
          dst,
          dtypeDst,
          elemIndexDst,
          constantValue,
          offsetElements: offsetDst,
        );
      }

      for (var d = rank - 1; d >= 0; d--) {
        coords[d]++;
        if (coords[d] < outShape[d]) break;
        coords[d] = 0;
      }
    }
  }

  /// Repeatedly tiles a tensor along all dimensions.
  static void executeTile({
    required GpuBuffer src,
    required List<int> shapeSrc,
    required List<int> stridesSrc,
    required int offsetSrc,
    required DType dtypeSrc,
    required GpuBuffer dst,
    required List<int> outShape,
    required List<int> outStrides,
    required int offsetDst,
    required DType dtypeDst,
  }) {
    final totalElements = computeSize(outShape);
    if (totalElements == 0) return;

    final rank = outShape.length;
    final padRank = rank - shapeSrc.length;
    final paddedShapeSrc = List<int>.filled(padRank, 1, growable: true)
      ..addAll(shapeSrc);
    final paddedStridesSrc = List<int>.filled(padRank, 0, growable: true)
      ..addAll(stridesSrc);

    if (!src.device.backend.isSimulated &&
        dtypeSrc == dtypeDst &&
        WgslDType.isNativelySupportedStorageDType(dtypeDst) &&
        rank <= 8 &&
        dst.address != src.address) {
      final shaderModule = WgslTemplates.tileKernel(
        dtype: WgslDType.fromDType(dtypeDst),
      );
      final dispatch = shaderModule.calculateDispatch1D(totalElements);
      final uniforms = _packStridedMetadata(
        rank: rank,
        totalElements: totalElements,
        offsetA: offsetSrc,
        offsetB: 0,
        offsetOut: offsetDst,
        shape: outShape,
        stridesA: paddedStridesSrc,
        stridesB: paddedShapeSrc,
        stridesOut: outStrides,
      );
      src.device.backend.dispatchComputePipeline(
        shaderModule: shaderModule,
        buffers: [src, dst],
        uniforms: uniforms,
        workgroupsX: dispatch.workgroupsX,
        workgroupsY: dispatch.workgroupsY,
        workgroupsZ: dispatch.workgroupsZ,
      );
      return;
    }

    final coords = List<int>.filled(rank, 0);

    for (var i = 0; i < totalElements; i++) {
      var elemIndexDst = 0;
      var elemIndexSrc = 0;

      for (var d = 0; d < rank; d++) {
        elemIndexDst += coords[d] * outStrides[d];
        final srcCoord = coords[d] % paddedShapeSrc[d];
        elemIndexSrc += srcCoord * paddedStridesSrc[d];
      }

      final val = readBufferAny(
        src,
        dtypeSrc,
        elemIndexSrc,
        offsetElements: offsetSrc,
      );
      writeBufferAny(
        dst,
        dtypeDst,
        elemIndexDst,
        val,
        offsetElements: offsetDst,
      );

      for (var d = rank - 1; d >= 0; d--) {
        coords[d]++;
        if (coords[d] < outShape[d]) break;
        coords[d] = 0;
      }
    }
  }

  /// Extracts upper or lower triangular portion of a 2D or batched 2D tensor.
  static void executeTriangular({
    required GpuBuffer src,
    required List<int> shapeSrc,
    required List<int> stridesSrc,
    required int offsetSrc,
    required DType dtypeSrc,
    required GpuBuffer dst,
    required List<int> outStrides,
    required int offsetDst,
    required DType dtypeDst,
    required int k,
    required bool upper,
  }) {
    final totalElements = computeSize(shapeSrc);
    final rank = shapeSrc.length;
    final coords = List<int>.filled(rank, 0);

    for (var i = 0; i < totalElements; i++) {
      var elemIndexSrc = 0;
      var elemIndexDst = 0;

      for (var d = 0; d < rank; d++) {
        elemIndexSrc += coords[d] * stridesSrc[d];
        elemIndexDst += coords[d] * outStrides[d];
      }

      final row = coords[rank - 2];
      final col = coords[rank - 1];
      final keep = upper ? (col - row >= k) : (col - row <= k);

      if (keep) {
        final val = readBufferAny(
          src,
          dtypeSrc,
          elemIndexSrc,
          offsetElements: offsetSrc,
        );
        writeBufferAny(
          dst,
          dtypeDst,
          elemIndexDst,
          val,
          offsetElements: offsetDst,
        );
      } else {
        writeBufferAny(
          dst,
          dtypeDst,
          elemIndexDst,
          0.0,
          offsetElements: offsetDst,
        );
      }

      for (var d = rank - 1; d >= 0; d--) {
        coords[d]++;
        if (coords[d] < shapeSrc[d]) break;
        coords[d] = 0;
      }
    }
  }
}
