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
  add,
  subtract,
  multiply,
  divide,
  power,
  remainder,
  maximum,
  minimum,
  equal,
  notEqual,
  greater,
  less,
  greaterEqual,
  lessEqual,
}

/// Standard unary operation kernel types.
enum UnaryOp {
  negate,
  abs,
  sqrt,
  exp,
  log,
  sin,
  cos,
  tan,
  asin,
  acos,
  atan,
  sinh,
  cosh,
  tanh,
  floor,
  ceil,
  round,
}

/// Execution kernels for GPU compute operations.
final class GpuKernels {
  GpuKernels._();

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
    final bStridesA = ShapeUtils.broadcastStrides(shapeA, stridesA, outShape);
    final bStridesB = ShapeUtils.broadcastStrides(shapeB, stridesB, outShape);
    final totalElements = ShapeUtils.computeSize(outShape);

    if (!srcA.device.backend.isSimulated &&
        dtypeA == DType.float32 &&
        dtypeB == DType.float32 &&
        dtypeDst == DType.float32 &&
        ShapeUtils.isContiguous(shapeA, stridesA) &&
        ShapeUtils.isContiguous(shapeB, stridesB) &&
        ShapeUtils.isContiguous(outShape, outStrides) &&
        offsetA == 0 &&
        offsetB == 0 &&
        offsetDst == 0 &&
        shapeA.length == outShape.length &&
        shapeB.length == outShape.length) {
      final shaderModule = WgslTemplates.elementwiseBinary(
        op: op.name,
        dtype: WgslDType.float32,
        strided: false,
      );
      srcA.device.backend.dispatchComputePipeline(
        shaderModule: shaderModule,
        buffers: [srcA, srcB, dst],
        uniforms: [totalElements, 0, 0, 0],
        workgroupsX: math.min(65535, (totalElements + 255) ~/ 256),
      );
      return;
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
      var elemIdxA = 0;
      var elemIdxB = 0;
      var elemIdxDst = 0;

      for (var d = 0; d < rank; d++) {
        elemIdxA += coords[d] * bStridesA[d];
        elemIdxB += coords[d] * bStridesB[d];
        elemIdxDst += coords[d] * outStrides[d];
      }

      if (isComplex) {
        final valA = ComputeEngine.readAny(
          srcA,
          dtypeA,
          elemIdxA,
          offsetElements: offsetA,
        );
        final valB = ComputeEngine.readAny(
          srcB,
          dtypeB,
          elemIdxB,
          offsetElements: offsetB,
        );

        final result = _applyComplexBinary(op, valA, valB);

        ComputeEngine.writeAny(
          dst,
          dtypeDst,
          elemIdxDst,
          result,
          offsetElements: offsetDst,
        );
      } else {
        final valA = ComputeEngine.readValue(
          srcA,
          dtypeA,
          elemIdxA,
          offsetElements: offsetA,
        );
        final valB = ComputeEngine.readValue(
          srcB,
          dtypeB,
          elemIdxB,
          offsetElements: offsetB,
        );

        final result = _applyBinary(op, valA, valB);

        ComputeEngine.writeValue(
          dst,
          dtypeDst,
          elemIdxDst,
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
    final totalElements = ShapeUtils.computeSize(shape);

    if (!src.device.backend.isSimulated &&
        dtypeSrc == DType.float32 &&
        dtypeDst == DType.float32 &&
        ShapeUtils.isContiguous(shape, strides) &&
        ShapeUtils.isContiguous(shape, outStrides) &&
        offsetSrc == 0 &&
        offsetDst == 0) {
      final shaderModule = WgslTemplates.elementwiseUnary(
        op: op.name,
        dtype: WgslDType.float32,
        strided: false,
      );
      src.device.backend.dispatchComputePipeline(
        shaderModule: shaderModule,
        buffers: [src, dst],
        uniforms: [totalElements, 0, 0, 0],
        workgroupsX: math.min(65535, (totalElements + 255) ~/ 256),
      );
      return;
    }

    final rank = shape.length;
    final coords = List<int>.filled(rank, 0);

    final isComplex =
        dtypeSrc == DType.complex64 ||
        dtypeSrc == DType.complex128 ||
        dtypeDst == DType.complex64 ||
        dtypeDst == DType.complex128;

    for (var i = 0; i < totalElements; i++) {
      var elemIdxSrc = 0;
      var elemIdxDst = 0;

      for (var d = 0; d < rank; d++) {
        elemIdxSrc += coords[d] * strides[d];
        elemIdxDst += coords[d] * outStrides[d];
      }

      if (isComplex) {
        final val = ComputeEngine.readAny(
          src,
          dtypeSrc,
          elemIdxSrc,
          offsetElements: offsetSrc,
        );
        final result = _applyComplexUnary(op, val);

        ComputeEngine.writeAny(
          dst,
          dtypeDst,
          elemIdxDst,
          result,
          offsetElements: offsetDst,
        );
      } else {
        final val = ComputeEngine.readValue(
          src,
          dtypeSrc,
          elemIdxSrc,
          offsetElements: offsetSrc,
        );
        final result = _applyUnary(op, val);

        ComputeEngine.writeValue(
          dst,
          dtypeDst,
          elemIdxDst,
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
      final totalElements = ShapeUtils.computeSize(shape);
      if (totalElements == 0) {
        if (isComplex) {
          ComputeEngine.writeAny(
            dst,
            dtypeDst,
            0,
            Complex(0.0, 0.0),
            offsetElements: offsetDst,
          );
        } else {
          ComputeEngine.writeValue(
            dst,
            dtypeDst,
            0,
            0.0,
            offsetElements: offsetDst,
          );
        }
        return;
      }

      final rank = shape.length;
      final coords = List<int>.filled(rank, 0);

      if (isComplex) {
        var accum = _initialComplexReductionValue(op);

        for (var i = 0; i < totalElements; i++) {
          var elemIdx = 0;
          for (var d = 0; d < rank; d++) {
            elemIdx += coords[d] * strides[d];
          }

          final rawVal = ComputeEngine.readAny(
            src,
            dtypeSrc,
            elemIdx,
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

        ComputeEngine.writeAny(
          dst,
          dtypeDst,
          0,
          accum,
          offsetElements: offsetDst,
        );
      } else {
        var accum = _initialReductionValue(op);

        for (var i = 0; i < totalElements; i++) {
          var elemIdx = 0;
          for (var d = 0; d < rank; d++) {
            elemIdx += coords[d] * strides[d];
          }

          final val = ComputeEngine.readValue(
            src,
            dtypeSrc,
            elemIdx,
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

        ComputeEngine.writeValue(
          dst,
          dtypeDst,
          0,
          accum,
          offsetElements: offsetDst,
        );
      }
    } else {
      // Reduction along a single axis
      final normAxis = axis < 0 ? axis + shape.length : axis;
      final axisSize = shape[normAxis];
      final totalOut = ShapeUtils.computeSize(outShape);

      final outRank = outShape.length;
      final outCoords = List<int>.filled(outRank, 0);

      for (var outIdx = 0; outIdx < totalOut; outIdx++) {
        var dstElemIdx = 0;
        for (var d = 0; d < outRank; d++) {
          dstElemIdx += outCoords[d] * outStrides[d];
        }

        if (isComplex) {
          var accum = _initialComplexReductionValue(op);

          for (var a = 0; a < axisSize; a++) {
            var srcElemIdx = 0;
            var inDim = 0;
            for (var d = 0; d < shape.length; d++) {
              if (d == normAxis) {
                srcElemIdx += a * strides[d];
              } else {
                srcElemIdx += outCoords[inDim] * strides[d];
                inDim++;
              }
            }

            final rawVal = ComputeEngine.readAny(
              src,
              dtypeSrc,
              srcElemIdx,
              offsetElements: offsetSrc,
            );
            accum = _combineComplexReduction(op, accum, _toComplex(rawVal));
          }

          if (op == 'mean') {
            accum = Complex(accum.real / axisSize, accum.imag / axisSize);
          }

          ComputeEngine.writeAny(
            dst,
            dtypeDst,
            dstElemIdx,
            accum,
            offsetElements: offsetDst,
          );
        } else {
          var accum = _initialReductionValue(op);

          for (var a = 0; a < axisSize; a++) {
            // Reconstruct input coords from output coords + axis index
            var srcElemIdx = 0;
            var inDim = 0;
            for (var d = 0; d < shape.length; d++) {
              if (d == normAxis) {
                srcElemIdx += a * strides[d];
              } else {
                srcElemIdx += outCoords[inDim] * strides[d];
                inDim++;
              }
            }

            final val = ComputeEngine.readValue(
              src,
              dtypeSrc,
              srcElemIdx,
              offsetElements: offsetSrc,
            );
            accum = _combineReduction(op, accum, val, a);
          }

          if (op == 'mean') {
            accum = accum / axisSize;
          }

          ComputeEngine.writeValue(
            dst,
            dtypeDst,
            dstElemIdx,
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

      if (!srcA.device.backend.isSimulated &&
          dtypeA == DType.float32 &&
          dtypeB == DType.float32 &&
          dtypeDst == DType.float32) {
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
          final idxDst = m * outStrides[0] + n * outStrides[1];
          if (isComplex) {
            var sumReal = 0.0;
            var sumImag = 0.0;
            for (var k = 0; k < K; k++) {
              final idxA = m * stridesA[0] + k * stridesA[1];
              final idxB = k * stridesB[0] + n * stridesB[1];
              final a = _toComplex(
                ComputeEngine.readAny(
                  srcA,
                  dtypeA,
                  idxA,
                  offsetElements: offsetA,
                ),
              );
              final b = _toComplex(
                ComputeEngine.readAny(
                  srcB,
                  dtypeB,
                  idxB,
                  offsetElements: offsetB,
                ),
              );
              sumReal += a.real * b.real - a.imag * b.imag;
              sumImag += a.real * b.imag + a.imag * b.real;
            }
            ComputeEngine.writeAny(
              dst,
              dtypeDst,
              idxDst,
              Complex(sumReal, sumImag),
              offsetElements: offsetDst,
            );
          } else {
            var sum = 0.0;
            for (var k = 0; k < K; k++) {
              final idxA = m * stridesA[0] + k * stridesA[1];
              final idxB = k * stridesB[0] + n * stridesB[1];
              final a = ComputeEngine.readValue(
                srcA,
                dtypeA,
                idxA,
                offsetElements: offsetA,
              );
              final b = ComputeEngine.readValue(
                srcB,
                dtypeB,
                idxB,
                offsetElements: offsetB,
              );
              sum += a * b;
            }
            ComputeEngine.writeValue(
              dst,
              dtypeDst,
              idxDst,
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
            ComputeEngine.readAny(
              srcA,
              dtypeA,
              k * stridesA[0],
              offsetElements: offsetA,
            ),
          );
          final b = _toComplex(
            ComputeEngine.readAny(
              srcB,
              dtypeB,
              k * stridesB[0],
              offsetElements: offsetB,
            ),
          );
          sumReal += a.real * b.real - a.imag * b.imag;
          sumImag += a.real * b.imag + a.imag * b.real;
        }
        ComputeEngine.writeAny(
          dst,
          dtypeDst,
          0,
          Complex(sumReal, sumImag),
          offsetElements: offsetDst,
        );
      } else {
        var sum = 0.0;
        for (var k = 0; k < K; k++) {
          final a = ComputeEngine.readValue(
            srcA,
            dtypeA,
            k * stridesA[0],
            offsetElements: offsetA,
          );
          final b = ComputeEngine.readValue(
            srcB,
            dtypeB,
            k * stridesB[0],
            offsetElements: offsetB,
          );
          sum += a * b;
        }
        ComputeEngine.writeValue(
          dst,
          dtypeDst,
          0,
          sum,
          offsetElements: offsetDst,
        );
      }
    } else {
      // Batched N-D matrix multiplication
      final batchShapeA = shapeA.sublist(0, rankA - 2);
      final batchShapeB = shapeB.sublist(0, rankB - 2);
      final batchOutShape = ShapeUtils.broadcastShapes(
        batchShapeA,
        batchShapeB,
      );

      final M = shapeA[rankA - 2];
      final K = shapeA[rankA - 1];
      final N = shapeB[rankB - 1];

      final batchSize = ShapeUtils.computeSize(batchOutShape);
      final batchRank = batchOutShape.length;
      final batchCoords = List<int>.filled(batchRank, 0);

      final batchStridesA = ShapeUtils.broadcastStrides(
        batchShapeA,
        stridesA.sublist(0, rankA - 2),
        batchOutShape,
      );
      final batchStridesB = ShapeUtils.broadcastStrides(
        batchShapeB,
        stridesB.sublist(0, rankB - 2),
        batchOutShape,
      );
      final batchStridesDst = outStrides.sublist(0, outStrides.length - 2);

      for (var b = 0; b < batchSize; b++) {
        var baseIdxA = 0;
        var baseIdxB = 0;
        var baseIdxDst = 0;

        for (var d = 0; d < batchRank; d++) {
          baseIdxA += batchCoords[d] * batchStridesA[d];
          baseIdxB += batchCoords[d] * batchStridesB[d];
          baseIdxDst += batchCoords[d] * batchStridesDst[d];
        }

        for (var m = 0; m < M; m++) {
          for (var n = 0; n < N; n++) {
            final idxDst =
                baseIdxDst +
                m * outStrides[outStrides.length - 2] +
                n * outStrides[outStrides.length - 1];
            if (isComplex) {
              var sumReal = 0.0;
              var sumImag = 0.0;
              for (var k = 0; k < K; k++) {
                final idxA =
                    baseIdxA +
                    m * stridesA[rankA - 2] +
                    k * stridesA[rankA - 1];
                final idxB =
                    baseIdxB +
                    k * stridesB[rankB - 2] +
                    n * stridesB[rankB - 1];
                final a = _toComplex(
                  ComputeEngine.readAny(
                    srcA,
                    dtypeA,
                    idxA,
                    offsetElements: offsetA,
                  ),
                );
                final b = _toComplex(
                  ComputeEngine.readAny(
                    srcB,
                    dtypeB,
                    idxB,
                    offsetElements: offsetB,
                  ),
                );
                sumReal += a.real * b.real - a.imag * b.imag;
                sumImag += a.real * b.imag + a.imag * b.real;
              }
              ComputeEngine.writeAny(
                dst,
                dtypeDst,
                idxDst,
                Complex(sumReal, sumImag),
                offsetElements: offsetDst,
              );
            } else {
              var sum = 0.0;
              for (var k = 0; k < K; k++) {
                final idxA =
                    baseIdxA +
                    m * stridesA[rankA - 2] +
                    k * stridesA[rankA - 1];
                final idxB =
                    baseIdxB +
                    k * stridesB[rankB - 2] +
                    n * stridesB[rankB - 1];
                final a = ComputeEngine.readValue(
                  srcA,
                  dtypeA,
                  idxA,
                  offsetElements: offsetA,
                );
                final b = ComputeEngine.readValue(
                  srcB,
                  dtypeB,
                  idxB,
                  offsetElements: offsetB,
                );
                sum += a * b;
              }
              ComputeEngine.writeValue(
                dst,
                dtypeDst,
                idxDst,
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
    final totalElements = ShapeUtils.computeSize(shape);
    final rank = shape.length;
    final coords = List<int>.filled(rank, 0);

    for (var i = 0; i < totalElements; i++) {
      var elemIdxSrc = 0;
      var elemIdxDst = 0;

      for (var d = 0; d < rank; d++) {
        elemIdxSrc += coords[d] * strides[d];
        elemIdxDst += coords[d] * outStrides[d];
      }

      final val = ComputeEngine.readAny(
        src,
        dtypeSrc,
        elemIdxSrc,
        offsetElements: offsetSrc,
      );
      ComputeEngine.writeAny(
        dst,
        dtypeDst,
        elemIdxDst,
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

  static Complex _toComplex(dynamic v) {
    if (v is Complex) return v;
    if (v is num) return Complex(v.toDouble(), 0.0);
    if (v is bool) return Complex(v ? 1.0 : 0.0, 0.0);
    return Complex(0.0, 0.0);
  }

  static dynamic _applyComplexBinary(BinaryOp op, dynamic rawA, dynamic rawB) {
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

  static dynamic _applyComplexUnary(UnaryOp op, dynamic raw) {
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
    final totalElements = ShapeUtils.computeSize(outShape);
    final rank = outShape.length;
    final bStridesCond = ShapeUtils.broadcastStrides(
      shapeCond,
      stridesCond,
      outShape,
    );
    final bStridesX = ShapeUtils.broadcastStrides(shapeX, stridesX, outShape);
    final bStridesY = ShapeUtils.broadcastStrides(shapeY, stridesY, outShape);
    final coords = List<int>.filled(rank, 0);

    for (var i = 0; i < totalElements; i++) {
      var elemIdxCond = 0;
      var elemIdxX = 0;
      var elemIdxY = 0;
      var elemIdxDst = 0;

      for (var d = 0; d < rank; d++) {
        elemIdxCond += coords[d] * bStridesCond[d];
        elemIdxX += coords[d] * bStridesX[d];
        elemIdxY += coords[d] * bStridesY[d];
        elemIdxDst += coords[d] * outStrides[d];
      }

      final cVal = ComputeEngine.readValue(
        cond,
        DType.boolean,
        elemIdxCond,
        offsetElements: offsetCond,
      );
      final isTrue = cVal != 0.0;

      final val = isTrue
          ? ComputeEngine.readAny(
              srcX,
              dtypeX,
              elemIdxX,
              offsetElements: offsetX,
            )
          : ComputeEngine.readAny(
              srcY,
              dtypeY,
              elemIdxY,
              offsetElements: offsetY,
            );

      ComputeEngine.writeAny(
        dst,
        dtypeDst,
        elemIdxDst,
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
    required List<int> shapeIdx,
    required List<int> stridesIdx,
    required int offsetIdx,
    required DType dtypeIdx,
    required GpuBuffer dst,
    required List<int> outShape,
    required List<int> outStrides,
    required int offsetDst,
    required DType dtypeDst,
    required int axis,
  }) {
    final totalElements = ShapeUtils.computeSize(outShape);
    final rank = outShape.length;
    final normAxis = axis < 0 ? axis + rank : axis;
    final axisLen = shapeSrc[normAxis];
    final coords = List<int>.filled(rank, 0);

    for (var i = 0; i < totalElements; i++) {
      var elemIdxIdx = 0;
      var elemIdxDst = 0;

      for (var d = 0; d < rank; d++) {
        elemIdxIdx += coords[d] * stridesIdx[d];
        elemIdxDst += coords[d] * outStrides[d];
      }

      final idxValNum = ComputeEngine.readValue(
        indices,
        dtypeIdx,
        elemIdxIdx,
        offsetElements: offsetIdx,
      );
      var k = idxValNum.toInt();
      if (k < 0) k += axisLen;
      if (k < 0 || k >= axisLen) {
        throw IndexError.withLength(
          k,
          axisLen,
          name: 'take_along_axis index out of bounds',
        );
      }

      var elemIdxSrc = 0;
      for (var d = 0; d < rank; d++) {
        final c = (d == normAxis) ? k : coords[d];
        elemIdxSrc += c * stridesSrc[d];
      }

      final val = ComputeEngine.readAny(
        src,
        dtypeSrc,
        elemIdxSrc,
        offsetElements: offsetSrc,
      );
      ComputeEngine.writeAny(
        dst,
        dtypeDst,
        elemIdxDst,
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
    required List<int> shapeIdx,
    required List<int> stridesIdx,
    required int offsetIdx,
    required DType dtypeIdx,
    required GpuBuffer values,
    required List<int> shapeVal,
    required List<int> stridesVal,
    required int offsetVal,
    required DType dtypeVal,
    required int axis,
  }) {
    final totalElements = ShapeUtils.computeSize(shapeIdx);
    final rank = shapeArr.length;
    final normAxis = axis < 0 ? axis + rank : axis;
    final axisLen = shapeArr[normAxis];
    final bStridesVal = ShapeUtils.broadcastStrides(
      shapeVal,
      stridesVal,
      shapeIdx,
    );
    final coords = List<int>.filled(rank, 0);

    for (var i = 0; i < totalElements; i++) {
      var elemIdxIdx = 0;
      var elemIdxVal = 0;

      for (var d = 0; d < rank; d++) {
        elemIdxIdx += coords[d] * stridesIdx[d];
        elemIdxVal += coords[d] * bStridesVal[d];
      }

      final idxValNum = ComputeEngine.readValue(
        indices,
        dtypeIdx,
        elemIdxIdx,
        offsetElements: offsetIdx,
      );
      var k = idxValNum.toInt();
      if (k < 0) k += axisLen;
      if (k < 0 || k >= axisLen) {
        throw IndexError.withLength(
          k,
          axisLen,
          name: 'put_along_axis index out of bounds',
        );
      }

      var elemIdxArr = 0;
      for (var d = 0; d < rank; d++) {
        final c = (d == normAxis) ? k : coords[d];
        elemIdxArr += c * stridesArr[d];
      }

      final val = ComputeEngine.readAny(
        values,
        dtypeVal,
        elemIdxVal,
        offsetElements: offsetVal,
      );
      ComputeEngine.writeAny(
        arr,
        dtypeArr,
        elemIdxArr,
        val,
        offsetElements: offsetArr,
      );

      for (var d = rank - 1; d >= 0; d--) {
        coords[d]++;
        if (coords[d] < shapeIdx[d]) break;
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

    for (var aIdx = 0; aIdx < srcBuffers.length; aIdx++) {
      final src = srcBuffers[aIdx];
      final shape = srcShapes[aIdx];
      final strides = srcStrides[aIdx];
      final offset = srcOffsets[aIdx];
      final dtype = srcDtypes[aIdx];
      final total = ShapeUtils.computeSize(shape);
      final coords = List<int>.filled(rank, 0);

      for (var i = 0; i < total; i++) {
        var elemIdxSrc = 0;
        var elemIdxDst = 0;

        for (var d = 0; d < rank; d++) {
          elemIdxSrc += coords[d] * strides[d];
          final outCoord = (d == normAxis) ? coords[d] + axisOffset : coords[d];
          elemIdxDst += outCoord * outStrides[d];
        }

        final val = ComputeEngine.readAny(
          src,
          dtype,
          elemIdxSrc,
          offsetElements: offset,
        );
        ComputeEngine.writeAny(
          dst,
          dtypeDst,
          elemIdxDst,
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
    required dynamic constantValue,
  }) {
    final totalElements = ShapeUtils.computeSize(outShape);
    final rank = outShape.length;
    final coords = List<int>.filled(rank, 0);

    for (var i = 0; i < totalElements; i++) {
      var elemIdxDst = 0;
      var inBounds = true;
      var elemIdxSrc = 0;

      for (var d = 0; d < rank; d++) {
        elemIdxDst += coords[d] * outStrides[d];
        final srcCoord = coords[d] - padWidth[d][0];
        if (srcCoord < 0 || srcCoord >= shapeSrc[d]) {
          inBounds = false;
        } else {
          elemIdxSrc += srcCoord * stridesSrc[d];
        }
      }

      if (inBounds) {
        final val = ComputeEngine.readAny(
          src,
          dtypeSrc,
          elemIdxSrc,
          offsetElements: offsetSrc,
        );
        ComputeEngine.writeAny(
          dst,
          dtypeDst,
          elemIdxDst,
          val,
          offsetElements: offsetDst,
        );
      } else {
        ComputeEngine.writeAny(
          dst,
          dtypeDst,
          elemIdxDst,
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
    final totalElements = ShapeUtils.computeSize(outShape);
    final rank = outShape.length;
    final padRank = rank - shapeSrc.length;
    final paddedShapeSrc = List<int>.filled(padRank, 1, growable: true)
      ..addAll(shapeSrc);
    final paddedStridesSrc = List<int>.filled(padRank, 0, growable: true)
      ..addAll(stridesSrc);
    final coords = List<int>.filled(rank, 0);

    for (var i = 0; i < totalElements; i++) {
      var elemIdxDst = 0;
      var elemIdxSrc = 0;

      for (var d = 0; d < rank; d++) {
        elemIdxDst += coords[d] * outStrides[d];
        final srcCoord = coords[d] % paddedShapeSrc[d];
        elemIdxSrc += srcCoord * paddedStridesSrc[d];
      }

      final val = ComputeEngine.readAny(
        src,
        dtypeSrc,
        elemIdxSrc,
        offsetElements: offsetSrc,
      );
      ComputeEngine.writeAny(
        dst,
        dtypeDst,
        elemIdxDst,
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
    final totalElements = ShapeUtils.computeSize(shapeSrc);
    final rank = shapeSrc.length;
    final coords = List<int>.filled(rank, 0);

    for (var i = 0; i < totalElements; i++) {
      var elemIdxSrc = 0;
      var elemIdxDst = 0;

      for (var d = 0; d < rank; d++) {
        elemIdxSrc += coords[d] * stridesSrc[d];
        elemIdxDst += coords[d] * outStrides[d];
      }

      final row = coords[rank - 2];
      final col = coords[rank - 1];
      final keep = upper ? (col - row >= k) : (col - row <= k);

      if (keep) {
        final val = ComputeEngine.readAny(
          src,
          dtypeSrc,
          elemIdxSrc,
          offsetElements: offsetSrc,
        );
        ComputeEngine.writeAny(
          dst,
          dtypeDst,
          elemIdxDst,
          val,
          offsetElements: offsetDst,
        );
      } else {
        ComputeEngine.writeAny(
          dst,
          dtypeDst,
          elemIdxDst,
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
