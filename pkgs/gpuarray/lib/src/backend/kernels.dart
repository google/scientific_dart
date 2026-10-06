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
import '../buffer.dart';
import '../dtype.dart';
import 'compute_engine.dart';
import 'wgsl/wgsl_dtype_codec.dart';
import 'wgsl/wgsl_indexing_kernels.dart';
import 'wgsl/wgsl_manipulation_kernels.dart';
import 'wgsl/wgsl_templates.dart';
import 'wgsl/wgsl_types.dart';
import 'wgsl/wgsl_universal_kernels.dart';

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

  /// Floor division operation (`a ~/ b`).
  floorDivide,

  /// C-style fmod remainder operation.
  fmod,

  /// Elementwise two-argument arctangent (`atan2(a, b)`).
  atan2,

  /// Elementwise hypotenuse (`sqrt(a^2 + b^2)`).
  hypot,

  /// Elementwise copy sign of `b` to magnitude of `a`.
  copysign,

  /// Elementwise `a * 2^b`.
  ldexp,

  /// Elementwise greatest common divisor.
  gcd,

  /// Elementwise least common multiple.
  lcm,

  /// Elementwise bitwise AND (`a & b`).
  bitwiseAnd,

  /// Elementwise bitwise OR (`a | b`).
  bitwiseOr,

  /// Elementwise bitwise XOR (`a ^ b`).
  bitwiseXor,

  /// Elementwise bitwise left shift (`a << b`).
  leftShift,

  /// Elementwise bitwise right shift (`a >> b`).
  rightShift,
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

  /// Sign indication (`sign(x)`).
  sign,

  /// Bitwise NOT / inversion (`~x`).
  bitwiseNot,

  /// Complex conjugate (`conj(x)`).
  conj,

  /// Truncation toward zero (`trunc(x)`).
  trunc,

  /// Round to nearest integer (`rint(x)`).
  rint,

  /// Cube root (`cbrt(x)`).
  cbrt,

  /// Reciprocal (`1 / x`).
  reciprocal,

  /// Elementwise square (`x * x`).
  square,

  /// Reciprocal square root (`1 / sqrt(x)`).
  rsqrt,

  /// `exp(x) - 1`.
  expm1,

  /// `2^x`.
  exp2,

  /// Base-2 logarithm (`log2(x)`).
  log2,

  /// Base-10 logarithm (`log10(x)`).
  log10,

  /// `log(1 + x)`.
  log1p,

  /// Inverse hyperbolic sine (`asinh(x)`).
  asinh,

  /// Inverse hyperbolic cosine (`acosh(x)`).
  acosh,

  /// Inverse hyperbolic tangent (`atanh(x)`).
  atanh,

  /// Degrees to radians.
  deg2rad,

  /// Radians to degrees.
  rad2deg,
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

  static bool _isTemplateBinaryOp(BinaryOp op) => switch (op) {
    BinaryOp.add ||
    BinaryOp.subtract ||
    BinaryOp.multiply ||
    BinaryOp.divide ||
    BinaryOp.power ||
    BinaryOp.remainder ||
    BinaryOp.maximum ||
    BinaryOp.minimum => true,
    _ => false,
  };

  static bool _isWgslUnarySupported(UnaryOp op, DType dtype) {
    final isTemplateOp = switch (op) {
      UnaryOp.negate ||
      UnaryOp.abs ||
      UnaryOp.sqrt ||
      UnaryOp.exp ||
      UnaryOp.log ||
      UnaryOp.sin ||
      UnaryOp.cos ||
      UnaryOp.tan ||
      UnaryOp.asin ||
      UnaryOp.acos ||
      UnaryOp.atan ||
      UnaryOp.sinh ||
      UnaryOp.cosh ||
      UnaryOp.tanh ||
      UnaryOp.floor ||
      UnaryOp.ceil ||
      UnaryOp.round => true,
      _ => false,
    };
    if (!isTemplateOp) return false;
    if (dtype == DType.float32) return true;
    if (dtype == DType.int32) {
      return op == UnaryOp.negate || op == UnaryOp.abs;
    }
    if (dtype == DType.uint32) {
      return op == UnaryOp.abs;
    }
    return false;
  }

  /// Runs [action] with a temporary buffer if [dst] aliases any buffer in [inputs].
  static void _withAliasSafeDst({
    required List<GpuBuffer> inputs,
    required GpuBuffer dst,
    required List<int> outShape,
    required List<int> outStrides,
    required int offsetDst,
    required DType dtypeDst,
    required void Function(
      GpuBuffer safeDst,
      List<int> safeStrides,
      int safeOffset,
    )
    action,
  }) {
    final aliases = inputs.any((b) => identical(b, dst));
    if (!aliases) {
      action(dst, outStrides, offsetDst);
      return;
    }
    final totalElements = computeSize(outShape);
    final tempBytes = math.max(totalElements * dtypeDst.byteWidth, 4);
    final tempDst = dst.device.createBuffer(sizeInBytes: tempBytes);
    final tempStrides = computeCStrides(outShape);
    try {
      action(tempDst, tempStrides, 0);
      copyStrided(
        src: tempDst,
        shape: outShape,
        strides: tempStrides,
        offsetSrc: 0,
        dtypeSrc: dtypeDst,
        dst: dst,
        outStrides: outStrides,
        offsetDst: offsetDst,
        dtypeDst: dtypeDst,
      );
    } finally {
      tempDst.dispose();
    }
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

    _withAliasSafeDst(
      inputs: [srcA, srcB],
      dst: dst,
      outShape: outShape,
      outStrides: outStrides,
      offsetDst: offsetDst,
      dtypeDst: dtypeDst,
      action: (safeDst, safeOutStrides, safeOffsetDst) {
        if (dtypeA == dtypeB &&
            dtypeB == dtypeDst &&
            WgslDType.isNativelySupportedStorageDType(dtypeA) &&
            _isTemplateBinaryOp(op) &&
            (op != BinaryOp.power || dtypeA == DType.float32)) {
          final wgslDType = WgslDType.fromDType(dtypeA);
          final isContiguous =
              isContiguousLayout(shapeA, stridesA) &&
              isContiguousLayout(shapeB, stridesB) &&
              isContiguousLayout(outShape, safeOutStrides) &&
              offsetA == 0 &&
              offsetB == 0 &&
              safeOffsetDst == 0 &&
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
              buffers: [srcA, srcB, safeDst],
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
              offsetOut: safeOffsetDst,
              shape: outShape,
              stridesA: bStridesA,
              stridesB: bStridesB,
              stridesOut: safeOutStrides,
            );
            srcA.device.backend.dispatchComputePipeline(
              shaderModule: shaderModule,
              buffers: [srcA, srcB, safeDst],
              uniforms: uniforms,
              workgroupsX: dispatch.workgroupsX,
              workgroupsY: dispatch.workgroupsY,
              workgroupsZ: dispatch.workgroupsZ,
            );
            return;
          }
        }

        final isComp = _isComparisonOp(op);
        final opDType = isComp ? dtypeA : dtypeDst;
        GpuBuffer? castA;
        GpuBuffer? castB;
        try {
          var bufferA = srcA;
          var sA = bStridesA;
          var offA = offsetA;
          if (dtypeA != opDType) {
            castA = srcA.device.createBuffer(
              sizeInBytes: math.max(totalElements * opDType.byteWidth, 4),
            );
            final cStrides = computeCStrides(outShape);
            copyStrided(
              src: srcA,
              shape: outShape,
              strides: bStridesA,
              offsetSrc: offsetA,
              dtypeSrc: dtypeA,
              dst: castA,
              outStrides: cStrides,
              offsetDst: 0,
              dtypeDst: opDType,
            );
            bufferA = castA;
            sA = cStrides;
            offA = 0;
          }
          var bufferB = srcB;
          var sB = bStridesB;
          var offB = offsetB;
          if (dtypeB != opDType) {
            castB = srcB.device.createBuffer(
              sizeInBytes: math.max(totalElements * opDType.byteWidth, 4),
            );
            final cStrides = computeCStrides(outShape);
            copyStrided(
              src: srcB,
              shape: outShape,
              strides: bStridesB,
              offsetSrc: offsetB,
              dtypeSrc: dtypeB,
              dst: castB,
              outStrides: cStrides,
              offsetDst: 0,
              dtypeDst: opDType,
            );
            bufferB = castB;
            sB = cStrides;
            offB = 0;
          }
          final shaderModule = WgslUniversalKernels.binaryShader(
            op: op.name,
            dtype: opDType,
            isComparison: isComp,
          );
          final dispatch = shaderModule.calculateDispatch1D(totalElements);
          final uniforms = _packStridedMetadata(
            rank: outShape.length,
            totalElements: totalElements,
            offsetA: offA,
            offsetB: offB,
            offsetOut: safeOffsetDst,
            shape: outShape,
            stridesA: sA,
            stridesB: sB,
            stridesOut: safeOutStrides,
          );
          if (isContiguousLayout(outShape, sA) &&
              isContiguousLayout(outShape, sB) &&
              isContiguousLayout(outShape, safeOutStrides)) {
            uniforms[2] = 1;
          }
          srcA.device.backend.dispatchComputePipeline(
            shaderModule: shaderModule,
            buffers: [bufferA, bufferB, safeDst],
            uniforms: uniforms,
            workgroupsX: dispatch.workgroupsX,
            workgroupsY: dispatch.workgroupsY,
            workgroupsZ: dispatch.workgroupsZ,
          );
        } finally {
          castA?.dispose();
          castB?.dispose();
        }
      },
    );
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

    _withAliasSafeDst(
      inputs: [src],
      dst: dst,
      outShape: shape,
      outStrides: outStrides,
      offsetDst: offsetDst,
      dtypeDst: dtypeDst,
      action: (safeDst, safeOutStrides, safeOffsetDst) {
        if (dtypeSrc == dtypeDst && _isWgslUnarySupported(op, dtypeSrc)) {
          final wgslDType = WgslDType.fromDType(dtypeSrc);
          if (isContiguousLayout(shape, strides) &&
              isContiguousLayout(shape, safeOutStrides) &&
              offsetSrc == 0 &&
              safeOffsetDst == 0) {
            final shaderModule = WgslTemplates.elementwiseUnary(
              op: op.name,
              dtype: wgslDType,
              strided: false,
            );
            final dispatch = shaderModule.calculateDispatch1D(totalElements);
            src.device.backend.dispatchComputePipeline(
              shaderModule: shaderModule,
              buffers: [src, safeDst],
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
              offsetOut: safeOffsetDst,
              shape: shape,
              stridesA: strides,
              stridesB: const [],
              stridesOut: safeOutStrides,
            );
            src.device.backend.dispatchComputePipeline(
              shaderModule: shaderModule,
              buffers: [src, safeDst],
              uniforms: uniforms,
              workgroupsX: dispatch.workgroupsX,
              workgroupsY: dispatch.workgroupsY,
              workgroupsZ: dispatch.workgroupsZ,
            );
            return;
          }
        }

        final shaderModule = WgslUniversalKernels.unaryShader(
          op: op.name,
          dtype: dtypeSrc,
        );
        final dispatch = shaderModule.calculateDispatch1D(totalElements);
        final effectiveOutStrides = dtypeSrc == dtypeDst
            ? safeOutStrides
            : computeCStrides(shape);
        final uniforms = _packStridedMetadata(
          rank: shape.length,
          totalElements: totalElements,
          offsetA: offsetSrc,
          offsetB: 0,
          offsetOut: dtypeSrc == dtypeDst ? safeOffsetDst : 0,
          shape: shape,
          stridesA: strides,
          stridesB: const [],
          stridesOut: effectiveOutStrides,
        );
        if (isContiguousLayout(shape, strides) &&
            isContiguousLayout(shape, effectiveOutStrides)) {
          uniforms[2] = 1;
        }
        if (dtypeSrc == dtypeDst) {
          src.device.backend.dispatchComputePipeline(
            shaderModule: shaderModule,
            buffers: [src, safeDst],
            uniforms: uniforms,
            workgroupsX: dispatch.workgroupsX,
            workgroupsY: dispatch.workgroupsY,
            workgroupsZ: dispatch.workgroupsZ,
          );
        } else {
          final tempOut = src.device.createBuffer(
            sizeInBytes: math.max(totalElements * dtypeSrc.byteWidth, 4),
          );
          try {
            src.device.backend.dispatchComputePipeline(
              shaderModule: shaderModule,
              buffers: [src, tempOut],
              uniforms: uniforms,
              workgroupsX: dispatch.workgroupsX,
              workgroupsY: dispatch.workgroupsY,
              workgroupsZ: dispatch.workgroupsZ,
            );
            copyStrided(
              src: tempOut,
              shape: shape,
              strides: computeCStrides(shape),
              offsetSrc: 0,
              dtypeSrc: dtypeSrc,
              dst: safeDst,
              outStrides: safeOutStrides,
              offsetDst: safeOffsetDst,
              dtypeDst: dtypeDst,
            );
          } finally {
            tempOut.dispose();
          }
        }
      },
    );
  }

  /// Executes an elementwise unary boolean predicate (`isnan`, `isinf`, `isfinite`, `signbit`).
  static void executeUnaryPredicate({
    required String op,
    required GpuBuffer src,
    required List<int> shape,
    required List<int> strides,
    required int offsetSrc,
    required DType dtypeSrc,
    required GpuBuffer dst,
    required List<int> outStrides,
    required int offsetDst,
  }) {
    final totalElements = computeSize(shape);
    if (totalElements == 0) return;

    _withAliasSafeDst(
      inputs: [src],
      dst: dst,
      outShape: shape,
      outStrides: outStrides,
      offsetDst: offsetDst,
      dtypeDst: DType.boolean,
      action: (safeDst, safeOutStrides, safeOffsetDst) {
        final shaderModule = WgslUniversalKernels.unaryShader(
          op: op,
          dtype: dtypeSrc,
          isPredicate: true,
        );
        final dispatch = shaderModule.calculateDispatch1D(totalElements);
        final uniforms = _packStridedMetadata(
          rank: shape.length,
          totalElements: totalElements,
          offsetA: offsetSrc,
          offsetB: 0,
          offsetOut: safeOffsetDst,
          shape: shape,
          stridesA: strides,
          stridesB: const [],
          stridesOut: safeOutStrides,
        );
        if (isContiguousLayout(shape, strides) &&
            isContiguousLayout(shape, safeOutStrides)) {
          uniforms[2] = 1;
        }
        src.device.backend.dispatchComputePipeline(
          shaderModule: shaderModule,
          buffers: [src, safeDst],
          uniforms: uniforms,
          workgroupsX: dispatch.workgroupsX,
          workgroupsY: dispatch.workgroupsY,
          workgroupsZ: dispatch.workgroupsZ,
        );
      },
    );
  }

  /// Executes a complex component or phase extraction shader (`real`, `imag`, `angle`).
  static void executeComplexComponent({
    required String op,
    required GpuBuffer src,
    required List<int> shape,
    required List<int> strides,
    required int offsetSrc,
    required DType dtypeSrc,
    required GpuBuffer dst,
    required List<int> outStrides,
    required int offsetDst,
    required DType dtypeDst,
    double scale = 1.0,
  }) {
    final totalElements = computeSize(shape);
    if (totalElements == 0) return;

    _withAliasSafeDst(
      inputs: [src],
      dst: dst,
      outShape: shape,
      outStrides: outStrides,
      offsetDst: offsetDst,
      dtypeDst: dtypeDst,
      action: (safeDst, safeOutStrides, safeOffsetDst) {
        final shaderModule = WgslUniversalKernels.complexComponentShader(
          op: op,
          sourceDType: dtypeSrc,
          targetDType: dtypeDst,
        );
        final dispatch = shaderModule.calculateDispatch1D(totalElements);
        final uniforms = _packStridedMetadata(
          rank: shape.length,
          totalElements: totalElements,
          offsetA: offsetSrc,
          offsetB: 0,
          offsetOut: safeOffsetDst,
          shape: shape,
          stridesA: strides,
          stridesB: const [],
          stridesOut: safeOutStrides,
        );
        if (isContiguousLayout(shape, strides) &&
            isContiguousLayout(shape, safeOutStrides)) {
          uniforms[2] = 1;
        }
        final scaleBits = ByteData(4)..setFloat32(0, scale, Endian.little);
        uniforms[39] = scaleBits.getUint32(0, Endian.little);
        src.device.backend.dispatchComputePipeline(
          shaderModule: shaderModule,
          buffers: [src, safeDst],
          uniforms: uniforms,
          workgroupsX: dispatch.workgroupsX,
          workgroupsY: dispatch.workgroupsY,
          workgroupsZ: dispatch.workgroupsZ,
        );
      },
    );
  }

  /// Replaces NaN, positive infinity, and negative infinity values in [src].
  static void executeNanToNum({
    required GpuBuffer src,
    required List<int> shape,
    required List<int> strides,
    required int offsetSrc,
    required DType dtype,
    required GpuBuffer dst,
    required List<int> outStrides,
    required int offsetDst,
    required double nan,
    required double posinf,
    required double neginf,
  }) {
    final totalElements = computeSize(shape);
    if (totalElements == 0) return;

    _withAliasSafeDst(
      inputs: [src],
      dst: dst,
      outShape: shape,
      outStrides: outStrides,
      offsetDst: offsetDst,
      dtypeDst: dtype,
      action: (safeDst, safeOutStrides, safeOffsetDst) {
        final shaderModule = WgslUniversalKernels.nanToNumShader(dtype);
        final dispatch = shaderModule.calculateDispatch1D(totalElements);
        final bd = ByteData(36);
        bd.setFloat64(0, nan, Endian.little);
        bd.setFloat64(8, posinf, Endian.little);
        bd.setFloat64(16, neginf, Endian.little);
        bd.setFloat32(24, nan, Endian.little);
        bd.setFloat32(28, posinf, Endian.little);
        bd.setFloat32(32, neginf, Endian.little);

        final uniforms = List<int>.filled(36, 0);
        uniforms[0] = totalElements;
        uniforms[1] = shape.length;
        uniforms[2] =
            (isContiguousLayout(shape, strides) &&
                isContiguousLayout(shape, safeOutStrides))
            ? 1
            : 0;
        uniforms[3] = 0;
        for (var d = 0; d < 8; d++) {
          uniforms[4 + d] = d < shape.length ? shape[d] : 1;
          uniforms[12 + d] = d < strides.length ? (strides[d] & 0xFFFFFFFF) : 0;
          uniforms[20 + d] = d < safeOutStrides.length
              ? (safeOutStrides[d] & 0xFFFFFFFF)
              : 0;
        }
        uniforms[28] = offsetSrc;
        uniforms[29] = safeOffsetDst;
        uniforms[30] = bd.getUint32(0, Endian.little);
        uniforms[31] = bd.getUint32(4, Endian.little);
        uniforms[32] = bd.getUint32(8, Endian.little);
        uniforms[33] = bd.getUint32(12, Endian.little);
        uniforms[34] = bd.getUint32(16, Endian.little);
        uniforms[35] = bd.getUint32(20, Endian.little);
        final fullUniforms = <int>[
          ...uniforms,
          bd.getUint32(24, Endian.little),
          bd.getUint32(28, Endian.little),
          bd.getUint32(32, Endian.little),
          0,
        ];
        src.device.backend.dispatchComputePipeline(
          shaderModule: shaderModule,
          buffers: [src, safeDst],
          uniforms: fullUniforms,
          workgroupsX: dispatch.workgroupsX,
          workgroupsY: dispatch.workgroupsY,
          workgroupsZ: dispatch.workgroupsZ,
        );
      },
    );
  }

  /// Executes elementwise `isClose` tolerance comparison outputting [DType.boolean].
  static void executeIsClose({
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
    required DType opDType,
    required double rtol,
    required double atol,
    required bool equalNan,
  }) {
    final bStridesA = broadcastStrides(shapeA, stridesA, outShape);
    final bStridesB = broadcastStrides(shapeB, stridesB, outShape);
    final totalElements = computeSize(outShape);
    if (totalElements == 0) return;

    _withAliasSafeDst(
      inputs: [srcA, srcB],
      dst: dst,
      outShape: outShape,
      outStrides: outStrides,
      offsetDst: offsetDst,
      dtypeDst: DType.boolean,
      action: (safeDst, safeOutStrides, safeOffsetDst) {
        GpuBuffer? castA;
        GpuBuffer? castB;
        try {
          var bufferA = srcA;
          var sA = bStridesA;
          var offA = offsetA;
          if (dtypeA != opDType) {
            castA = srcA.device.createBuffer(
              sizeInBytes: math.max(totalElements * opDType.byteWidth, 4),
            );
            final cStrides = computeCStrides(outShape);
            copyStrided(
              src: srcA,
              shape: outShape,
              strides: bStridesA,
              offsetSrc: offsetA,
              dtypeSrc: dtypeA,
              dst: castA,
              outStrides: cStrides,
              offsetDst: 0,
              dtypeDst: opDType,
            );
            bufferA = castA;
            sA = cStrides;
            offA = 0;
          }
          var bufferB = srcB;
          var sB = bStridesB;
          var offB = offsetB;
          if (dtypeB != opDType) {
            castB = srcB.device.createBuffer(
              sizeInBytes: math.max(totalElements * opDType.byteWidth, 4),
            );
            final cStrides = computeCStrides(outShape);
            copyStrided(
              src: srcB,
              shape: outShape,
              strides: bStridesB,
              offsetSrc: offsetB,
              dtypeSrc: dtypeB,
              dst: castB,
              outStrides: cStrides,
              offsetDst: 0,
              dtypeDst: opDType,
            );
            bufferB = castB;
            sB = cStrides;
            offB = 0;
          }
          final shaderModule = WgslUniversalKernels.isCloseShader(opDType);
          final dispatch = shaderModule.calculateDispatch1D(totalElements);
          final uniforms = _packStridedMetadata(
            rank: outShape.length,
            totalElements: totalElements,
            offsetA: offA,
            offsetB: offB,
            offsetOut: safeOffsetDst,
            shape: outShape,
            stridesA: sA,
            stridesB: sB,
            stridesOut: safeOutStrides,
          );
          var flags = 0;
          if (isContiguousLayout(outShape, sA) &&
              isContiguousLayout(outShape, sB) &&
              isContiguousLayout(outShape, safeOutStrides)) {
            flags |= 1;
          }
          if (equalNan) {
            flags |= 2;
          }
          uniforms[2] = flags;
          final bd = ByteData(8);
          bd.setFloat32(0, rtol, Endian.little);
          bd.setFloat32(4, atol, Endian.little);
          uniforms[3] = bd.getUint32(0, Endian.little);
          uniforms[39] = bd.getUint32(4, Endian.little);
          srcA.device.backend.dispatchComputePipeline(
            shaderModule: shaderModule,
            buffers: [bufferA, bufferB, safeDst],
            uniforms: uniforms,
            workgroupsX: dispatch.workgroupsX,
            workgroupsY: dispatch.workgroupsY,
            workgroupsZ: dispatch.workgroupsZ,
          );
        } finally {
          castA?.dispose();
          castB?.dispose();
        }
      },
    );
  }

  /// Executes a reduction kernel across specified axes or the entire tensor.
  static void executeReduction({
    required String op,
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
    int ddof = 0,
  }) {
    final requiresNonEmpty =
        op == 'min' ||
        op == 'max' ||
        op == 'nanmin' ||
        op == 'nanmax' ||
        op == 'ptp' ||
        op == 'argmin' ||
        op == 'argmax';
    final isArgOp = op == 'argmin' || op == 'argmax';
    final isCountNonzero = op == 'count_nonzero' || op == 'countNonzero';
    final isBoolOp = op == 'all' || op == 'any';
    final isComplexStatOp =
        dtypeSrc.isComplex && (op == 'variance' || op == 'std');

    if (axis == null) {
      final totalElements = computeSize(shape);
      if (totalElements == 0) {
        if (requiresNonEmpty) {
          throw StateError('Cannot compute $op of an empty array.');
        }
        final Object emptyVal = switch (op) {
          'prod' => 1,
          'all' => true,
          'any' => false,
          'mean' || 'nanmean' || 'variance' || 'std' => double.nan,
          _ => 0,
        };
        executeFill(
          dst: dst,
          outShape: outShape,
          outStrides: outStrides,
          offsetDst: offsetDst,
          dtypeDst: dtypeDst,
          value: emptyVal,
        );
        return;
      }

      _withAliasSafeDst(
        inputs: [src],
        dst: dst,
        outShape: outShape,
        outStrides: outStrides,
        offsetDst: offsetDst,
        dtypeDst: dtypeDst,
        action: (safeDst, safeOutStrides, safeOffsetDst) {
          if (dtypeSrc == dtypeDst &&
              WgslDType.isNativelySupportedStorageDType(dtypeSrc) &&
              (op == 'sum' ||
                  op == 'prod' ||
                  op == 'min' ||
                  op == 'max' ||
                  (op == 'mean' && dtypeSrc == DType.float32))) {
            final wgslDType = WgslDType.fromDType(dtypeSrc);
            if (isContiguousLayout(shape, strides) &&
                offsetSrc == 0 &&
                safeOffsetDst == 0) {
              final shaderModule = WgslTemplates.treeReduction(
                op: op,
                dtype: wgslDType,
                strided: false,
              );
              src.device.backend.dispatchComputePipeline(
                shaderModule: shaderModule,
                buffers: [src, safeDst],
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
                offsetOut: safeOffsetDst,
                shape: shape,
                stridesA: strides,
                stridesB: const [],
                stridesOut: const [],
              );
              src.device.backend.dispatchComputePipeline(
                shaderModule: shaderModule,
                buffers: [src, safeDst],
                uniforms: uniforms,
                workgroupsX: 1,
                workgroupsY: 1,
                workgroupsZ: 1,
              );
              return;
            }
          }

          GpuBuffer? castSrc;
          try {
            var effSrc = src;
            var effStrides = strides;
            var effOffset = offsetSrc;
            var effDType = dtypeSrc;
            if (!isArgOp &&
                !isCountNonzero &&
                !isBoolOp &&
                !isComplexStatOp &&
                dtypeSrc != dtypeDst) {
              castSrc = src.device.createBuffer(
                sizeInBytes: math.max(totalElements * dtypeDst.byteWidth, 4),
              );
              final cStrides = computeCStrides(shape);
              copyStrided(
                src: src,
                shape: shape,
                strides: strides,
                offsetSrc: offsetSrc,
                dtypeSrc: dtypeSrc,
                dst: castSrc,
                outStrides: cStrides,
                offsetDst: 0,
                dtypeDst: dtypeDst,
              );
              effSrc = castSrc;
              effStrides = cStrides;
              effOffset = 0;
              effDType = dtypeDst;
            }
            final shaderModule = WgslUniversalKernels.reductionShader(
              op: op,
              dtype: effDType,
              targetDType: dtypeDst,
            );
            final uniforms = List<int>.filled(32, 0);
            uniforms[0] = 1;
            uniforms[1] = shape.length;
            uniforms[2] = 99;
            uniforms[3] = totalElements;
            uniforms[4] = effOffset;
            uniforms[5] = safeOffsetDst;
            uniforms[6] = isContiguousLayout(shape, effStrides) ? 1 : 0;
            uniforms[7] = ddof & 0xFFFFFFFF;
            for (var d = 0; d < 8; d++) {
              uniforms[8 + d] = d < shape.length ? shape[d] : 1;
              uniforms[16 + d] = d < effStrides.length
                  ? (effStrides[d] & 0xFFFFFFFF)
                  : 0;
            }
            src.device.backend.dispatchComputePipeline(
              shaderModule: shaderModule,
              buffers: [effSrc, safeDst],
              uniforms: uniforms,
              workgroupsX: 1,
              workgroupsY: 1,
              workgroupsZ: 1,
            );
          } finally {
            castSrc?.dispose();
          }
        },
      );
    } else {
      final normAxis = axis < 0 ? axis + shape.length : axis;
      final axisSize = shape[normAxis];
      final totalOut = computeSize(outShape);
      if (axisSize == 0) {
        if (requiresNonEmpty) {
          throw StateError('Cannot compute $op along an empty axis (size 0).');
        }
        if (totalOut == 0) return;
        final Object emptyVal = switch (op) {
          'prod' => 1,
          'all' => true,
          'any' => false,
          'mean' || 'nanmean' || 'variance' || 'std' => double.nan,
          _ => 0,
        };
        executeFill(
          dst: dst,
          outShape: outShape,
          outStrides: outStrides,
          offsetDst: offsetDst,
          dtypeDst: dtypeDst,
          value: emptyVal,
        );
        return;
      }
      if (totalOut == 0) return;

      final nonAxisStrides = <int>[];
      if (outShape.length == shape.length) {
        for (var d = 0; d < shape.length; d++) {
          nonAxisStrides.add(d == normAxis ? 0 : strides[d]);
        }
      } else {
        for (var d = 0; d < shape.length; d++) {
          if (d != normAxis) nonAxisStrides.add(strides[d]);
        }
      }

      final totalElements = computeSize(shape);
      _withAliasSafeDst(
        inputs: [src],
        dst: dst,
        outShape: outShape,
        outStrides: outStrides,
        offsetDst: offsetDst,
        dtypeDst: dtypeDst,
        action: (safeDst, safeOutStrides, safeOffsetDst) {
          if (dtypeSrc == dtypeDst &&
              WgslDType.isNativelySupportedStorageDType(dtypeSrc) &&
              (op == 'sum' ||
                  op == 'prod' ||
                  op == 'min' ||
                  op == 'max' ||
                  (op == 'mean' && dtypeSrc == DType.float32)) &&
              outShape.length <= 8) {
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
              offsetOut: safeOffsetDst,
              shape: outShape,
              stridesA: nonAxisStrides,
              stridesB: [axisSize, strides[normAxis]],
              stridesOut: safeOutStrides,
            );
            src.device.backend.dispatchComputePipeline(
              shaderModule: shaderModule,
              buffers: [src, safeDst],
              uniforms: uniforms,
              workgroupsX: dispatch.workgroupsX,
              workgroupsY: dispatch.workgroupsY,
              workgroupsZ: dispatch.workgroupsZ,
            );
            return;
          }

          GpuBuffer? castSrc;
          try {
            var effSrc = src;
            var effStrides = strides;
            var effOffset = offsetSrc;
            var effDType = dtypeSrc;
            if (!isArgOp &&
                !isCountNonzero &&
                !isBoolOp &&
                !isComplexStatOp &&
                dtypeSrc != dtypeDst) {
              castSrc = src.device.createBuffer(
                sizeInBytes: math.max(totalElements * dtypeDst.byteWidth, 4),
              );
              final cStrides = computeCStrides(shape);
              copyStrided(
                src: src,
                shape: shape,
                strides: strides,
                offsetSrc: offsetSrc,
                dtypeSrc: dtypeSrc,
                dst: castSrc,
                outStrides: cStrides,
                offsetDst: 0,
                dtypeDst: dtypeDst,
              );
              effSrc = castSrc;
              effStrides = cStrides;
              effOffset = 0;
              effDType = dtypeDst;
            }
            final reducedOutStrides = <int>[];
            if (safeOutStrides.length == shape.length) {
              for (var d = 0; d < shape.length; d++) {
                if (d != normAxis) reducedOutStrides.add(safeOutStrides[d]);
              }
            } else {
              reducedOutStrides.addAll(safeOutStrides);
            }
            final shaderModule = WgslUniversalKernels.reductionShader(
              op: op,
              dtype: effDType,
              targetDType: dtypeDst,
            );
            final dispatch = shaderModule.calculateDispatch1D(totalOut);
            final uniforms = List<int>.filled(32, 0);
            uniforms[0] = totalOut;
            uniforms[1] = shape.length;
            uniforms[2] = normAxis;
            uniforms[3] = axisSize;
            uniforms[4] = effOffset;
            uniforms[5] = safeOffsetDst;
            uniforms[6] = 0;
            uniforms[7] = ddof & 0xFFFFFFFF;
            for (var d = 0; d < 8; d++) {
              uniforms[8 + d] = d < shape.length ? shape[d] : 1;
              uniforms[16 + d] = d < effStrides.length
                  ? (effStrides[d] & 0xFFFFFFFF)
                  : 0;
              uniforms[24 + d] = d < reducedOutStrides.length
                  ? (reducedOutStrides[d] & 0xFFFFFFFF)
                  : 0;
            }
            src.device.backend.dispatchComputePipeline(
              shaderModule: shaderModule,
              buffers: [effSrc, safeDst],
              uniforms: uniforms,
              workgroupsX: dispatch.workgroupsX,
              workgroupsY: dispatch.workgroupsY,
              workgroupsZ: dispatch.workgroupsZ,
            );
          } finally {
            castSrc?.dispose();
          }
        },
      );
    }
  }

  /// Executes tiled 2D, 1D dot, or batched N-D matrix multiplication on the GPU.
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

    if (rankA == 1 && rankB == 1) {
      executeMatmul(
        srcA: srcA,
        shapeA: [1, shapeA[0]],
        stridesA: [0, stridesA[0]],
        offsetA: offsetA,
        dtypeA: dtypeA,
        srcB: srcB,
        shapeB: [shapeB[0], 1],
        stridesB: [stridesB[0], 0],
        offsetB: offsetB,
        dtypeB: dtypeB,
        dst: dst,
        outShape: const [1, 1],
        outStrides: const [0, 0],
        offsetDst: offsetDst,
        dtypeDst: dtypeDst,
      );
      return;
    }

    final batchShapeA = shapeA.sublist(0, rankA - 2);
    final batchShapeB = shapeB.sublist(0, rankB - 2);
    final batchOutShape = broadcastShapes(batchShapeA, batchShapeB);

    final M = shapeA[rankA - 2];
    final K = shapeA[rankA - 1];
    final N = shapeB[rankB - 1];
    final batchSize = computeSize(batchOutShape);
    if (M == 0 || N == 0 || batchSize == 0) return;

    if (K == 0) {
      executeFill(
        dst: dst,
        outShape: outShape,
        outStrides: outStrides,
        offsetDst: offsetDst,
        dtypeDst: dtypeDst,
        value: 0,
      );
      return;
    }

    _withAliasSafeDst(
      inputs: [srcA, srcB],
      dst: dst,
      outShape: outShape,
      outStrides: outStrides,
      offsetDst: offsetDst,
      dtypeDst: dtypeDst,
      action: (safeDst, safeOutStrides, safeOffsetDst) {
        if (rankA == 2 &&
            rankB == 2 &&
            dtypeA == DType.float32 &&
            dtypeB == DType.float32 &&
            dtypeDst == DType.float32 &&
            stridesA[0] >= 0 &&
            stridesA[1] >= 0 &&
            stridesB[0] >= 0 &&
            stridesB[1] >= 0 &&
            safeOutStrides[0] >= 0 &&
            safeOutStrides[1] >= 0) {
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
            safeOutStrides[0],
            safeOutStrides[1],
            offsetA,
            offsetB,
            safeOffsetDst,
            alphaBits.getUint32(0, Endian.little),
            betaBits.getUint32(0, Endian.little),
            0,
            0,
          ];
          srcA.device.backend.dispatchComputePipeline(
            shaderModule: shaderModule,
            buffers: [srcA, srcB, safeDst],
            uniforms: uniforms,
            workgroupsX: (N + 15) ~/ 16,
            workgroupsY: (M + 15) ~/ 16,
            workgroupsZ: 1,
          );
          return;
        }

        GpuBuffer? castA;
        GpuBuffer? castB;
        try {
          var effA = srcA;
          var effStridesA = stridesA;
          var effOffsetA = offsetA;
          if (dtypeA != dtypeDst) {
            final sizeA = computeSize(shapeA);
            castA = srcA.device.createBuffer(
              sizeInBytes: math.max(sizeA * dtypeDst.byteWidth, 4),
            );
            final cStrides = computeCStrides(shapeA);
            copyStrided(
              src: srcA,
              shape: shapeA,
              strides: stridesA,
              offsetSrc: offsetA,
              dtypeSrc: dtypeA,
              dst: castA,
              outStrides: cStrides,
              offsetDst: 0,
              dtypeDst: dtypeDst,
            );
            effA = castA;
            effStridesA = cStrides;
            effOffsetA = 0;
          }
          var effB = srcB;
          var effStridesB = stridesB;
          var effOffsetB = offsetB;
          if (dtypeB != dtypeDst) {
            final sizeB = computeSize(shapeB);
            castB = srcB.device.createBuffer(
              sizeInBytes: math.max(sizeB * dtypeDst.byteWidth, 4),
            );
            final cStrides = computeCStrides(shapeB);
            copyStrided(
              src: srcB,
              shape: shapeB,
              strides: stridesB,
              offsetSrc: offsetB,
              dtypeSrc: dtypeB,
              dst: castB,
              outStrides: cStrides,
              offsetDst: 0,
              dtypeDst: dtypeDst,
            );
            effB = castB;
            effStridesB = cStrides;
            effOffsetB = 0;
          }

          final batchRank = batchOutShape.length;
          final batchStridesA = broadcastStrides(
            batchShapeA,
            effStridesA.sublist(0, rankA - 2),
            batchOutShape,
          );
          final batchStridesB = broadcastStrides(
            batchShapeB,
            effStridesB.sublist(0, rankB - 2),
            batchOutShape,
          );
          final batchStridesDst = safeOutStrides.sublist(
            0,
            safeOutStrides.length - 2,
          );

          final shaderModule = WgslUniversalKernels.matmulShader(dtypeDst);
          final uniforms = List<int>.filled(48, 0);
          uniforms[0] = M;
          uniforms[1] = N;
          uniforms[2] = K;
          uniforms[3] = batchSize;
          uniforms[4] = batchRank;
          uniforms[5] = effOffsetA;
          uniforms[6] = effOffsetB;
          uniforms[7] = safeOffsetDst;
          uniforms[8] = effStridesA[rankA - 2] & 0xFFFFFFFF;
          uniforms[9] = effStridesA[rankA - 1] & 0xFFFFFFFF;
          uniforms[10] = effStridesB[rankB - 2] & 0xFFFFFFFF;
          uniforms[11] = effStridesB[rankB - 1] & 0xFFFFFFFF;
          uniforms[12] = safeOutStrides[safeOutStrides.length - 2] & 0xFFFFFFFF;
          uniforms[13] = safeOutStrides[safeOutStrides.length - 1] & 0xFFFFFFFF;
          for (var d = 0; d < 8; d++) {
            uniforms[16 + d] = d < batchRank ? batchOutShape[d] : 1;
            uniforms[24 + d] = d < batchRank
                ? (batchStridesA[d] & 0xFFFFFFFF)
                : 0;
            uniforms[32 + d] = d < batchRank
                ? (batchStridesB[d] & 0xFFFFFFFF)
                : 0;
            uniforms[40 + d] = d < batchRank && d < batchStridesDst.length
                ? (batchStridesDst[d] & 0xFFFFFFFF)
                : 0;
          }
          srcA.device.backend.dispatchComputePipeline(
            shaderModule: shaderModule,
            buffers: [effA, effB, safeDst],
            uniforms: uniforms,
            workgroupsX: (N + 15) ~/ 16,
            workgroupsY: (M + 15) ~/ 16,
            workgroupsZ: batchSize,
          );
        } finally {
          castA?.dispose();
          castB?.dispose();
        }
      },
    );
  }

  /// Copies or casts strided data from [src] to [dst] on the GPU.
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

    _withAliasSafeDst(
      inputs: [src],
      dst: dst,
      outShape: shape,
      outStrides: outStrides,
      offsetDst: offsetDst,
      dtypeDst: dtypeDst,
      action: (safeDst, safeOutStrides, safeOffsetDst) {
        if (dtypeSrc == dtypeDst &&
            WgslDType.isNativelySupportedStorageDType(dtypeSrc) &&
            shape.length <= 8) {
          final shaderModule = WgslTemplates.tileKernel(
            dtype: WgslDType.fromDType(dtypeSrc),
          );
          final dispatch = shaderModule.calculateDispatch1D(totalElements);
          final uniforms = _packStridedMetadata(
            rank: shape.length,
            totalElements: totalElements,
            offsetA: offsetSrc,
            offsetB: 0,
            offsetOut: safeOffsetDst,
            shape: shape,
            stridesA: strides,
            stridesB: shape,
            stridesOut: safeOutStrides,
          );
          src.device.backend.dispatchComputePipeline(
            shaderModule: shaderModule,
            buffers: [src, safeDst],
            uniforms: uniforms,
            workgroupsX: dispatch.workgroupsX,
            workgroupsY: dispatch.workgroupsY,
            workgroupsZ: dispatch.workgroupsZ,
          );
          return;
        }

        final shaderModule = WgslUniversalKernels.castOrCopyShader(
          sourceDType: dtypeSrc,
          targetDType: dtypeDst,
        );
        final dispatch = shaderModule.calculateDispatch1D(totalElements);
        final uniforms = _packStridedMetadata(
          rank: shape.length,
          totalElements: totalElements,
          offsetA: offsetSrc,
          offsetB: 0,
          offsetOut: safeOffsetDst,
          shape: shape,
          stridesA: strides,
          stridesB: const [],
          stridesOut: safeOutStrides,
        );
        if (isContiguousLayout(shape, strides) &&
            isContiguousLayout(shape, safeOutStrides)) {
          uniforms[2] = 1;
        }
        src.device.backend.dispatchComputePipeline(
          shaderModule: shaderModule,
          buffers: [src, safeDst],
          uniforms: uniforms,
          workgroupsX: dispatch.workgroupsX,
          workgroupsY: dispatch.workgroupsY,
          workgroupsZ: dispatch.workgroupsZ,
        );
      },
    );
  }

  /// Fills [dst] with [value] on the GPU.
  static void executeFill({
    required GpuBuffer dst,
    required List<int> outShape,
    required List<int> outStrides,
    required int offsetDst,
    required DType dtypeDst,
    required Object? value,
  }) {
    final totalElements = computeSize(outShape);
    if (totalElements == 0) return;

    final shaderModule = WgslUniversalKernels.fillShader(dtypeDst);
    final dispatch = shaderModule.calculateDispatch1D(totalElements);
    final scalarWords = WgslDTypeCodec.packRawScalarWords(dtypeDst, value ?? 0);
    final uniforms = List<int>.filled(24, 0);
    uniforms[0] = totalElements;
    uniforms[1] = outShape.length;
    uniforms[2] = offsetDst;
    uniforms[3] = isContiguousLayout(outShape, outStrides) ? 1 : 0;
    for (var i = 0; i < 4; i++) {
      uniforms[4 + i] = scalarWords[i];
    }
    for (var d = 0; d < 8; d++) {
      uniforms[8 + d] = d < outShape.length ? outShape[d] : 1;
      uniforms[16 + d] = d < outStrides.length
          ? (outStrides[d] & 0xFFFFFFFF)
          : 0;
    }
    dst.device.backend.dispatchComputePipeline(
      shaderModule: shaderModule,
      buffers: [dst],
      uniforms: uniforms,
      workgroupsX: dispatch.workgroupsX,
      workgroupsY: dispatch.workgroupsY,
      workgroupsZ: dispatch.workgroupsZ,
    );
  }

  /// Generates sequence or matrix patterns (`arange`, `linspace`, `logspace`, `eye`, `tri`) on the GPU.
  static void executeGenerator({
    required String mode,
    required GpuBuffer dst,
    required List<int> outShape,
    required int offsetDst,
    required DType dtypeDst,
    double start = 0.0,
    double step = 1.0,
    double base = 10.0,
    int cols = 0,
    int k = 0,
  }) {
    final totalElements = computeSize(outShape);
    if (totalElements == 0) return;

    final shaderModule = WgslUniversalKernels.generatorShader(dtypeDst);
    final dispatch = shaderModule.calculateDispatch1D(totalElements);
    final modeInt = switch (mode) {
      'arange' || 'linspace' => 0,
      'logspace' => 1,
      'eye' => 2,
      'tri' => 3,
      _ => 0,
    };
    final rows = outShape.isNotEmpty ? outShape[0] : 1;
    final startBits = ByteData(4)..setFloat32(0, start, Endian.little);
    final stepBits = ByteData(4)..setFloat32(0, step, Endian.little);
    final baseBits = ByteData(4)..setFloat32(0, base, Endian.little);
    final uniforms = <int>[
      totalElements,
      modeInt,
      rows,
      cols,
      k & 0xFFFFFFFF,
      startBits.getUint32(0, Endian.little),
      stepBits.getUint32(0, Endian.little),
      baseBits.getUint32(0, Endian.little),
    ];
    if (offsetDst == 0) {
      dst.device.backend.dispatchComputePipeline(
        shaderModule: shaderModule,
        buffers: [dst],
        uniforms: uniforms,
        workgroupsX: dispatch.workgroupsX,
        workgroupsY: dispatch.workgroupsY,
        workgroupsZ: dispatch.workgroupsZ,
      );
    } else {
      final tempDst = dst.device.createBuffer(
        sizeInBytes: math.max(totalElements * dtypeDst.byteWidth, 4),
      );
      try {
        dst.device.backend.dispatchComputePipeline(
          shaderModule: shaderModule,
          buffers: [tempDst],
          uniforms: uniforms,
          workgroupsX: dispatch.workgroupsX,
          workgroupsY: dispatch.workgroupsY,
          workgroupsZ: dispatch.workgroupsZ,
        );
        final cStrides = computeCStrides(outShape);
        copyStrided(
          src: tempDst,
          shape: outShape,
          strides: cStrides,
          offsetSrc: 0,
          dtypeSrc: dtypeDst,
          dst: dst,
          outStrides: cStrides,
          offsetDst: offsetDst,
          dtypeDst: dtypeDst,
        );
      } finally {
        tempDst.dispose();
      }
    }
  }

  /// Dispatches conditional ternary selection (`where`) on the GPU.
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

    _withAliasSafeDst(
      inputs: [cond, srcX, srcY],
      dst: dst,
      outShape: outShape,
      outStrides: outStrides,
      offsetDst: offsetDst,
      dtypeDst: dtypeDst,
      action: (safeDst, safeOutStrides, safeOffsetDst) {
        if (dtypeX == dtypeY &&
            dtypeY == dtypeDst &&
            WgslDType.isNativelySupportedStorageDType(dtypeDst) &&
            rank <= 8) {
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
            offsetOut: safeOffsetDst,
            shape: outShape,
            stridesCond: bStridesCond,
            stridesX: bStridesX,
            stridesY: bStridesY,
            stridesOut: safeOutStrides,
          );
          srcX.device.backend.dispatchComputePipeline(
            shaderModule: shaderModule,
            buffers: [cond, srcX, srcY, safeDst],
            uniforms: uniforms,
            workgroupsX: dispatch.workgroupsX,
            workgroupsY: dispatch.workgroupsY,
            workgroupsZ: dispatch.workgroupsZ,
          );
          return;
        }

        GpuBuffer? castX;
        GpuBuffer? castY;
        try {
          var effX = srcX;
          var sX = bStridesX;
          var offX = offsetX;
          if (dtypeX != dtypeDst) {
            castX = srcX.device.createBuffer(
              sizeInBytes: math.max(totalElements * dtypeDst.byteWidth, 4),
            );
            final cStrides = computeCStrides(outShape);
            copyStrided(
              src: srcX,
              shape: outShape,
              strides: bStridesX,
              offsetSrc: offsetX,
              dtypeSrc: dtypeX,
              dst: castX,
              outStrides: cStrides,
              offsetDst: 0,
              dtypeDst: dtypeDst,
            );
            effX = castX;
            sX = cStrides;
            offX = 0;
          }
          var effY = srcY;
          var sY = bStridesY;
          var offY = offsetY;
          if (dtypeY != dtypeDst) {
            castY = srcY.device.createBuffer(
              sizeInBytes: math.max(totalElements * dtypeDst.byteWidth, 4),
            );
            final cStrides = computeCStrides(outShape);
            copyStrided(
              src: srcY,
              shape: outShape,
              strides: bStridesY,
              offsetSrc: offsetY,
              dtypeSrc: dtypeY,
              dst: castY,
              outStrides: cStrides,
              offsetDst: 0,
              dtypeDst: dtypeDst,
            );
            effY = castY;
            sY = cStrides;
            offY = 0;
          }
          final shaderModule = WgslIndexingKernels.whereShader(dtypeDst);
          final dispatch = shaderModule.calculateDispatch1D(totalElements);
          final uniforms = _packWhereUniforms(
            totalElements: totalElements,
            rank: rank,
            offsetCond: offsetCond,
            offsetX: offX,
            offsetY: offY,
            offsetOut: safeOffsetDst,
            shape: outShape,
            stridesCond: bStridesCond,
            stridesX: sX,
            stridesY: sY,
            stridesOut: safeOutStrides,
          );
          srcX.device.backend.dispatchComputePipeline(
            shaderModule: shaderModule,
            buffers: [cond, effX, effY, safeDst],
            uniforms: uniforms,
            workgroupsX: dispatch.workgroupsX,
            workgroupsY: dispatch.workgroupsY,
            workgroupsZ: dispatch.workgroupsZ,
          );
        } finally {
          castX?.dispose();
          castY?.dispose();
        }
      },
    );
  }

  /// Gathers elements from [src] using [indices] (flattened or along [axis]) on the GPU.
  static void executeTake({
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
    int? axis,
  }) {
    final totalElements = computeSize(outShape);
    if (totalElements == 0) return;

    final rankSrc = shapeSrc.length;
    final rankIndices = shapeIndices.length;
    final rankOut = outShape.length;
    final normAxis = axis == null ? 99 : (axis < 0 ? axis + rankSrc : axis);
    final axisLength = axis == null
        ? computeSize(shapeSrc)
        : shapeSrc[normAxis];
    final indicesSize = computeSize(shapeIndices);
    var innerSize = 1;
    if (axis != null) {
      for (var d = normAxis + 1; d < rankSrc; d++) {
        innerSize *= shapeSrc[d];
      }
    }

    final statusBuffer = src.device.createBuffer(sizeInBytes: 8);
    try {
      writeBufferValue(statusBuffer, DType.uint32, 0, 0);
      writeBufferValue(statusBuffer, DType.uint32, 1, 0);
      _withAliasSafeDst(
        inputs: [src, indices],
        dst: dst,
        outShape: outShape,
        outStrides: outStrides,
        offsetDst: offsetDst,
        dtypeDst: dtypeDst,
        action: (safeDst, safeOutStrides, safeOffsetDst) {
          final shaderModule = WgslIndexingKernels.takeShader(
            dtypeDst,
            dtypeIndices,
          );
          final dispatch = shaderModule.calculateDispatch1D(totalElements);
          final uniforms = List<int>.filled(60, 0);
          uniforms[0] = totalElements;
          uniforms[1] = rankSrc;
          uniforms[2] = rankIndices;
          uniforms[3] = rankOut;
          uniforms[4] = normAxis;
          uniforms[5] = axisLength;
          uniforms[6] = indicesSize;
          uniforms[7] = innerSize;
          uniforms[8] = offsetSrc;
          uniforms[9] = offsetIndices;
          uniforms[10] = safeOffsetDst;
          for (var d = 0; d < 8; d++) {
            uniforms[12 + d] = d < rankSrc ? shapeSrc[d] : 1;
            uniforms[20 + d] = d < rankSrc ? (stridesSrc[d] & 0xFFFFFFFF) : 0;
            uniforms[28 + d] = d < rankIndices ? shapeIndices[d] : 1;
            uniforms[36 + d] = d < rankIndices
                ? (stridesIndices[d] & 0xFFFFFFFF)
                : 0;
            uniforms[44 + d] = d < rankOut ? outShape[d] : 1;
            uniforms[52 + d] = d < rankOut
                ? (safeOutStrides[d] & 0xFFFFFFFF)
                : 0;
          }
          src.device.backend.dispatchComputePipeline(
            shaderModule: shaderModule,
            buffers: [src, indices, safeDst, statusBuffer],
            uniforms: uniforms,
            workgroupsX: dispatch.workgroupsX,
            workgroupsY: dispatch.workgroupsY,
            workgroupsZ: dispatch.workgroupsZ,
          );
        },
      );
      final err = readBufferAny(statusBuffer, DType.uint32, 0) as int;
      if (err != 0) {
        throw RangeError(
          'Index out of bounds in take (axisLength=$axisLength).',
        );
      }
    } finally {
      statusBuffer.dispose();
    }
  }

  /// Scatters [values] into flattened positions of [arr] specified by [indices] on the GPU.
  static void executePut({
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
  }) {
    final totalElements = computeSize(shapeIndices);
    if (totalElements == 0) return;

    final arrSize = computeSize(shapeArr);
    final valSize = computeSize(shapeVal);
    if (valSize == 0) {
      throw ArgumentError.value(
        shapeVal,
        'values',
        'Must not be empty when indices is non-empty.',
      );
    }

    final statusBuffer = arr.device.createBuffer(sizeInBytes: 8);
    GpuBuffer? castVal;
    try {
      var effValues = values;
      var effStridesVal = stridesVal;
      var effOffsetVal = offsetVal;
      if (dtypeVal != dtypeArr) {
        castVal = arr.device.createBuffer(
          sizeInBytes: math.max(valSize * dtypeArr.byteWidth, 4),
        );
        final cStrides = computeCStrides(shapeVal);
        copyStrided(
          src: values,
          shape: shapeVal,
          strides: stridesVal,
          offsetSrc: offsetVal,
          dtypeSrc: dtypeVal,
          dst: castVal,
          outStrides: cStrides,
          offsetDst: 0,
          dtypeDst: dtypeArr,
        );
        effValues = castVal;
        effStridesVal = cStrides;
        effOffsetVal = 0;
      }
      writeBufferValue(statusBuffer, DType.uint32, 0, 0);
      writeBufferValue(statusBuffer, DType.uint32, 1, 0);
      final shaderModule = WgslIndexingKernels.putShader(
        dtypeArr,
        dtypeIndices,
      );
      final dispatch = shaderModule.calculateDispatch1D(totalElements);
      final uniforms = List<int>.filled(60, 0);
      uniforms[0] = totalElements;
      uniforms[1] = arrSize;
      uniforms[2] = valSize;
      uniforms[3] = shapeArr.length;
      uniforms[4] = shapeIndices.length;
      uniforms[5] = shapeVal.length;
      uniforms[6] = offsetArr;
      uniforms[7] = offsetIndices;
      uniforms[8] = effOffsetVal;
      for (var d = 0; d < 8; d++) {
        uniforms[12 + d] = d < shapeArr.length ? shapeArr[d] : 1;
        uniforms[20 + d] = d < stridesArr.length
            ? (stridesArr[d] & 0xFFFFFFFF)
            : 0;
        uniforms[28 + d] = d < shapeIndices.length ? shapeIndices[d] : 1;
        uniforms[36 + d] = d < stridesIndices.length
            ? (stridesIndices[d] & 0xFFFFFFFF)
            : 0;
        uniforms[44 + d] = d < shapeVal.length ? shapeVal[d] : 1;
        uniforms[52 + d] = d < effStridesVal.length
            ? (effStridesVal[d] & 0xFFFFFFFF)
            : 0;
      }
      arr.device.backend.dispatchComputePipeline(
        shaderModule: shaderModule,
        buffers: [indices, effValues, arr, statusBuffer],
        uniforms: uniforms,
        workgroupsX: dispatch.workgroupsX,
        workgroupsY: dispatch.workgroupsY,
        workgroupsZ: dispatch.workgroupsZ,
      );
      final err = readBufferAny(statusBuffer, DType.uint32, 0) as int;
      if (err != 0) {
        throw RangeError('Index out of bounds in put (size=$arrSize).');
      }
    } finally {
      castVal?.dispose();
      statusBuffer.dispose();
    }
  }

  /// Extracts elements along an axis according to coordinates in [indices] on the GPU.
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
    if (totalElements == 0) return;

    final rank = outShape.length;
    final normAxis = axis < 0 ? axis + rank : axis;
    final axisLength = shapeSrc[normAxis];
    if (axisLength == 0) {
      throw IndexError.withLength(
        0,
        0,
        name: 'take_along_axis index out of bounds',
      );
    }

    final statusBuffer = src.device.createBuffer(sizeInBytes: 8);
    try {
      writeBufferValue(statusBuffer, DType.uint32, 0, 0);
      writeBufferValue(statusBuffer, DType.uint32, 1, 0);
      _withAliasSafeDst(
        inputs: [src, indices],
        dst: dst,
        outShape: outShape,
        outStrides: outStrides,
        offsetDst: offsetDst,
        dtypeDst: dtypeDst,
        action: (safeDst, safeOutStrides, safeOffsetDst) {
          final shaderModule = WgslIndexingKernels.alongAxisShader(
            dtype: dtypeDst,
            indexDType: dtypeIndices,
            isPut: false,
          );
          final dispatch = shaderModule.calculateDispatch1D(totalElements);
          final uniforms = List<int>.filled(40, 0);
          uniforms[0] = totalElements;
          uniforms[1] = rank;
          uniforms[2] = normAxis;
          uniforms[3] = axisLength;
          uniforms[4] = offsetSrc;
          uniforms[5] = offsetIndices;
          uniforms[6] = safeOffsetDst;
          for (var d = 0; d < 8; d++) {
            uniforms[8 + d] = d < rank ? outShape[d] : 1;
            uniforms[16 + d] = d < rank ? (stridesSrc[d] & 0xFFFFFFFF) : 0;
            uniforms[24 + d] = d < rank ? (stridesIndices[d] & 0xFFFFFFFF) : 0;
            uniforms[32 + d] = d < rank ? (safeOutStrides[d] & 0xFFFFFFFF) : 0;
          }
          src.device.backend.dispatchComputePipeline(
            shaderModule: shaderModule,
            buffers: [src, indices, safeDst, statusBuffer],
            uniforms: uniforms,
            workgroupsX: dispatch.workgroupsX,
            workgroupsY: dispatch.workgroupsY,
            workgroupsZ: dispatch.workgroupsZ,
          );
        },
      );
      final err = readBufferAny(statusBuffer, DType.uint32, 0) as int;
      if (err != 0) {
        throw IndexError.withLength(
          axisLength,
          axisLength,
          name: 'take_along_axis index out of bounds',
        );
      }
    } finally {
      statusBuffer.dispose();
    }
  }

  /// Inserts [values] into [arr] along an axis according to [indices] on the GPU.
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
    if (totalElements == 0) return;

    final rank = shapeArr.length;
    final normAxis = axis < 0 ? axis + rank : axis;
    final axisLength = shapeArr[normAxis];
    if (axisLength == 0) {
      throw IndexError.withLength(
        0,
        0,
        name: 'put_along_axis index out of bounds',
      );
    }

    final statusBuffer = arr.device.createBuffer(sizeInBytes: 8);
    GpuBuffer? castVal;
    try {
      var effValues = values;
      var effStridesVal = stridesVal;
      var effOffsetVal = offsetVal;
      if (dtypeVal != dtypeArr) {
        final valSize = computeSize(shapeVal);
        castVal = arr.device.createBuffer(
          sizeInBytes: math.max(valSize * dtypeArr.byteWidth, 4),
        );
        final cStrides = computeCStrides(shapeVal);
        copyStrided(
          src: values,
          shape: shapeVal,
          strides: stridesVal,
          offsetSrc: offsetVal,
          dtypeSrc: dtypeVal,
          dst: castVal,
          outStrides: cStrides,
          offsetDst: 0,
          dtypeDst: dtypeArr,
        );
        effValues = castVal;
        effStridesVal = cStrides;
        effOffsetVal = 0;
      }
      final bStridesVal = broadcastStrides(
        shapeVal,
        effStridesVal,
        shapeIndices,
      );
      writeBufferValue(statusBuffer, DType.uint32, 0, 0);
      writeBufferValue(statusBuffer, DType.uint32, 1, 0);
      final shaderModule = WgslIndexingKernels.alongAxisShader(
        dtype: dtypeArr,
        indexDType: dtypeIndices,
        isPut: true,
      );
      final dispatch = shaderModule.calculateDispatch1D(totalElements);
      final uniforms = List<int>.filled(40, 0);
      uniforms[0] = totalElements;
      uniforms[1] = rank;
      uniforms[2] = normAxis;
      uniforms[3] = axisLength;
      uniforms[4] = offsetArr;
      uniforms[5] = offsetIndices;
      uniforms[6] = effOffsetVal;
      for (var d = 0; d < 8; d++) {
        uniforms[8 + d] = d < rank ? shapeIndices[d] : 1;
        uniforms[16 + d] = d < rank ? (stridesArr[d] & 0xFFFFFFFF) : 0;
        uniforms[24 + d] = d < rank ? (stridesIndices[d] & 0xFFFFFFFF) : 0;
        uniforms[32 + d] = d < rank ? (bStridesVal[d] & 0xFFFFFFFF) : 0;
      }
      arr.device.backend.dispatchComputePipeline(
        shaderModule: shaderModule,
        buffers: [effValues, indices, arr, statusBuffer],
        uniforms: uniforms,
        workgroupsX: dispatch.workgroupsX,
        workgroupsY: dispatch.workgroupsY,
        workgroupsZ: dispatch.workgroupsZ,
      );
      final err = readBufferAny(statusBuffer, DType.uint32, 0) as int;
      if (err != 0) {
        throw IndexError.withLength(
          axisLength,
          axisLength,
          name: 'put_along_axis index out of bounds',
        );
      }
    } finally {
      castVal?.dispose();
      statusBuffer.dispose();
    }
  }

  /// Concatenates a list of tensors along [axis] on the GPU.
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
      final sliceOffsetDst = offsetDst + axisOffset * outStrides[normAxis];

      copyStrided(
        src: src,
        shape: shape,
        strides: strides,
        offsetSrc: offset,
        dtypeSrc: dtype,
        dst: dst,
        outStrides: outStrides,
        offsetDst: sliceOffsetDst,
        dtypeDst: dtypeDst,
      );

      axisOffset += shape[normAxis];
    }
  }

  /// Pads a tensor with specified padding widths and mode on the GPU.
  ///
  /// [padMode] is `0` for constant, `1` for edge, `2` for reflect,
  /// `3` for symmetric, and `4` for wrap.
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
    int padMode = 0,
  }) {
    final totalElements = computeSize(outShape);
    if (totalElements == 0) return;

    final rank = outShape.length;
    final constWords = WgslDTypeCodec.packRawScalarWords(
      dtypeDst,
      constantValue ?? 0,
    );

    _withAliasSafeDst(
      inputs: [src],
      dst: dst,
      outShape: outShape,
      outStrides: outStrides,
      offsetDst: offsetDst,
      dtypeDst: dtypeDst,
      action: (safeDst, safeOutStrides, safeOffsetDst) {
        final shaderModule = WgslIndexingKernels.padShader(dtypeDst);
        final dispatch = shaderModule.calculateDispatch1D(totalElements);
        final uniforms = List<int>.filled(52, 0);
        uniforms[0] = totalElements;
        uniforms[1] = rank;
        uniforms[2] = offsetSrc;
        uniforms[3] = safeOffsetDst;
        uniforms[4] = padMode;
        for (var i = 0; i < 4; i++) {
          uniforms[8 + i] = constWords[i];
        }
        for (var d = 0; d < 8; d++) {
          uniforms[12 + d] = d < rank ? shapeSrc[d] : 1;
          uniforms[20 + d] = d < rank ? outShape[d] : 1;
          uniforms[28 + d] = d < padWidth.length ? padWidth[d][0] : 0;
          uniforms[36 + d] = d < rank ? (stridesSrc[d] & 0xFFFFFFFF) : 0;
          uniforms[44 + d] = d < rank ? (safeOutStrides[d] & 0xFFFFFFFF) : 0;
        }
        src.device.backend.dispatchComputePipeline(
          shaderModule: shaderModule,
          buffers: [src, safeDst],
          uniforms: uniforms,
          workgroupsX: dispatch.workgroupsX,
          workgroupsY: dispatch.workgroupsY,
          workgroupsZ: dispatch.workgroupsZ,
        );
      },
    );
  }

  /// Repeatedly tiles a tensor along all dimensions on the GPU.
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

    _withAliasSafeDst(
      inputs: [src],
      dst: dst,
      outShape: outShape,
      outStrides: outStrides,
      offsetDst: offsetDst,
      dtypeDst: dtypeDst,
      action: (safeDst, safeOutStrides, safeOffsetDst) {
        if (dtypeSrc == dtypeDst &&
            WgslDType.isNativelySupportedStorageDType(dtypeDst) &&
            rank <= 8) {
          final shaderModule = WgslTemplates.tileKernel(
            dtype: WgslDType.fromDType(dtypeDst),
          );
          final dispatch = shaderModule.calculateDispatch1D(totalElements);
          final uniforms = _packStridedMetadata(
            rank: rank,
            totalElements: totalElements,
            offsetA: offsetSrc,
            offsetB: 0,
            offsetOut: safeOffsetDst,
            shape: outShape,
            stridesA: paddedStridesSrc,
            stridesB: paddedShapeSrc,
            stridesOut: safeOutStrides,
          );
          src.device.backend.dispatchComputePipeline(
            shaderModule: shaderModule,
            buffers: [src, safeDst],
            uniforms: uniforms,
            workgroupsX: dispatch.workgroupsX,
            workgroupsY: dispatch.workgroupsY,
            workgroupsZ: dispatch.workgroupsZ,
          );
          return;
        }

        final shaderModule = WgslIndexingKernels.tileShader(dtypeDst);
        final dispatch = shaderModule.calculateDispatch1D(totalElements);
        final uniforms = List<int>.filled(36, 0);
        uniforms[0] = totalElements;
        uniforms[1] = rank;
        uniforms[2] = offsetSrc;
        uniforms[3] = safeOffsetDst;
        for (var d = 0; d < 8; d++) {
          uniforms[4 + d] = d < rank ? paddedShapeSrc[d] : 1;
          uniforms[12 + d] = d < rank ? outShape[d] : 1;
          uniforms[20 + d] = d < rank ? (paddedStridesSrc[d] & 0xFFFFFFFF) : 0;
          uniforms[28 + d] = d < rank ? (safeOutStrides[d] & 0xFFFFFFFF) : 0;
        }
        src.device.backend.dispatchComputePipeline(
          shaderModule: shaderModule,
          buffers: [src, safeDst],
          uniforms: uniforms,
          workgroupsX: dispatch.workgroupsX,
          workgroupsY: dispatch.workgroupsY,
          workgroupsZ: dispatch.workgroupsZ,
        );
      },
    );
  }

  /// Extracts upper or lower triangular portion of a 2D or batched 2D tensor on the GPU.
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
    if (totalElements == 0) return;

    final rank = shapeSrc.length;
    _withAliasSafeDst(
      inputs: [src],
      dst: dst,
      outShape: shapeSrc,
      outStrides: outStrides,
      offsetDst: offsetDst,
      dtypeDst: dtypeDst,
      action: (safeDst, safeOutStrides, safeOffsetDst) {
        final shaderModule = WgslManipulationKernels.triangularShader(dtypeDst);
        final dispatch = shaderModule.calculateDispatch1D(totalElements);
        final uniforms = List<int>.filled(32, 0);
        uniforms[0] = totalElements;
        uniforms[1] = rank;
        uniforms[2] = k & 0xFFFFFFFF;
        uniforms[3] = upper ? 1 : 0;
        uniforms[4] = offsetSrc;
        uniforms[5] = safeOffsetDst;
        for (var d = 0; d < 8; d++) {
          uniforms[8 + d] = d < rank ? shapeSrc[d] : 1;
          uniforms[16 + d] = d < rank ? (stridesSrc[d] & 0xFFFFFFFF) : 0;
          uniforms[24 + d] = d < rank ? (safeOutStrides[d] & 0xFFFFFFFF) : 0;
        }
        src.device.backend.dispatchComputePipeline(
          shaderModule: shaderModule,
          buffers: [src, safeDst],
          uniforms: uniforms,
          workgroupsX: dispatch.workgroupsX,
          workgroupsY: dispatch.workgroupsY,
          workgroupsZ: dispatch.workgroupsZ,
        );
      },
    );
  }

  /// Repeats elements of [src] `repeats` times (flattened or along [axis]) on the GPU.
  static void executeRepeat({
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
    required int repeats,
    int? axis,
  }) {
    final totalElements = computeSize(outShape);
    if (totalElements == 0) return;

    final rank = shapeSrc.length;
    final normAxis = axis == null ? 99 : (axis < 0 ? axis + rank : axis);
    _withAliasSafeDst(
      inputs: [src],
      dst: dst,
      outShape: outShape,
      outStrides: outStrides,
      offsetDst: offsetDst,
      dtypeDst: dtypeDst,
      action: (safeDst, safeOutStrides, safeOffsetDst) {
        final shaderModule = WgslManipulationKernels.repeatShader(dtypeDst);
        final dispatch = shaderModule.calculateDispatch1D(totalElements);
        final uniforms = List<int>.filled(40, 0);
        uniforms[0] = totalElements;
        uniforms[1] = rank;
        uniforms[2] = normAxis;
        uniforms[3] = repeats;
        uniforms[4] = offsetSrc;
        uniforms[5] = safeOffsetDst;
        for (var d = 0; d < 8; d++) {
          uniforms[8 + d] = d < rank ? shapeSrc[d] : 1;
          uniforms[16 + d] = d < outShape.length ? outShape[d] : 1;
          uniforms[24 + d] = d < rank ? (stridesSrc[d] & 0xFFFFFFFF) : 0;
          uniforms[32 + d] = d < safeOutStrides.length
              ? (safeOutStrides[d] & 0xFFFFFFFF)
              : 0;
        }
        src.device.backend.dispatchComputePipeline(
          shaderModule: shaderModule,
          buffers: [src, safeDst],
          uniforms: uniforms,
          workgroupsX: dispatch.workgroupsX,
          workgroupsY: dispatch.workgroupsY,
          workgroupsZ: dispatch.workgroupsZ,
        );
      },
    );
  }

  /// Rolls elements of [src] by [shift] (flattened or along [axis]) on the GPU.
  static void executeRoll({
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
    required int shift,
    int? axis,
  }) {
    final totalElements = computeSize(outShape);
    if (totalElements == 0) return;

    final rank = outShape.length;
    final normAxis = axis == null ? 99 : (axis < 0 ? axis + rank : axis);
    _withAliasSafeDst(
      inputs: [src],
      dst: dst,
      outShape: outShape,
      outStrides: outStrides,
      offsetDst: offsetDst,
      dtypeDst: dtypeDst,
      action: (safeDst, safeOutStrides, safeOffsetDst) {
        final shaderModule = WgslManipulationKernels.rollShader(dtypeDst);
        final dispatch = shaderModule.calculateDispatch1D(totalElements);
        final uniforms = List<int>.filled(32, 0);
        uniforms[0] = totalElements;
        uniforms[1] = rank;
        uniforms[2] = normAxis;
        uniforms[3] = shift & 0xFFFFFFFF;
        uniforms[4] = offsetSrc;
        uniforms[5] = safeOffsetDst;
        for (var d = 0; d < 8; d++) {
          uniforms[8 + d] = d < rank ? outShape[d] : 1;
          uniforms[16 + d] = d < rank ? (stridesSrc[d] & 0xFFFFFFFF) : 0;
          uniforms[24 + d] = d < rank ? (safeOutStrides[d] & 0xFFFFFFFF) : 0;
        }
        src.device.backend.dispatchComputePipeline(
          shaderModule: shaderModule,
          buffers: [src, safeDst],
          uniforms: uniforms,
          workgroupsX: dispatch.workgroupsX,
          workgroupsY: dispatch.workgroupsY,
          workgroupsZ: dispatch.workgroupsZ,
        );
      },
    );
  }

  /// Constructs a 2D diagonal matrix in [dst] from 1D vector [src] on the GPU.
  static void executeDiag1DTo2D({
    required GpuBuffer src,
    required int srcLength,
    required int strideSrc,
    required int offsetSrc,
    required DType dtypeSrc,
    required GpuBuffer dst,
    required int outSize,
    required List<int> outStrides,
    required int offsetDst,
    required DType dtypeDst,
    required int k,
  }) {
    final totalElements = outSize * outSize;
    if (totalElements == 0) return;

    _withAliasSafeDst(
      inputs: [src],
      dst: dst,
      outShape: [outSize, outSize],
      outStrides: outStrides,
      offsetDst: offsetDst,
      dtypeDst: dtypeDst,
      action: (safeDst, safeOutStrides, safeOffsetDst) {
        final shaderModule = WgslManipulationKernels.diag1DTo2DShader(dtypeDst);
        final dispatch = shaderModule.calculateDispatch1D(totalElements);
        final uniforms = <int>[
          outSize,
          srcLength,
          k & 0xFFFFFFFF,
          offsetSrc,
          strideSrc & 0xFFFFFFFF,
          safeOffsetDst,
          safeOutStrides[0] & 0xFFFFFFFF,
          safeOutStrides[1] & 0xFFFFFFFF,
        ];
        src.device.backend.dispatchComputePipeline(
          shaderModule: shaderModule,
          buffers: [src, safeDst],
          uniforms: uniforms,
          workgroupsX: dispatch.workgroupsX,
          workgroupsY: dispatch.workgroupsY,
          workgroupsZ: dispatch.workgroupsZ,
        );
      },
    );
  }

  /// Extracts a diagonal along `(axis1, axis2)` with [offset] on the GPU.
  static void executeDiagonal({
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
    required int offset,
    required int axis1,
    required int axis2,
  }) {
    final totalElements = computeSize(outShape);
    if (totalElements == 0) return;

    final outRank = outShape.length;
    final rowStart = offset >= 0 ? 0 : -offset;
    final colStart = offset >= 0 ? offset : 0;
    final strideAxis1 = stridesSrc[axis1];
    final strideAxis2 = stridesSrc[axis2];
    final remStridesA = <int>[];
    for (var d = 0; d < shapeSrc.length; d++) {
      if (d != axis1 && d != axis2) {
        remStridesA.add(stridesSrc[d]);
      }
    }

    _withAliasSafeDst(
      inputs: [src],
      dst: dst,
      outShape: outShape,
      outStrides: outStrides,
      offsetDst: offsetDst,
      dtypeDst: dtypeDst,
      action: (safeDst, safeOutStrides, safeOffsetDst) {
        final shaderModule = WgslManipulationKernels.diagonalShader(dtypeDst);
        final dispatch = shaderModule.calculateDispatch1D(totalElements);
        final uniforms = List<int>.filled(32, 0);
        uniforms[0] = totalElements;
        uniforms[1] = outRank;
        uniforms[2] = rowStart;
        uniforms[3] = colStart;
        uniforms[4] = strideAxis1 & 0xFFFFFFFF;
        uniforms[5] = strideAxis2 & 0xFFFFFFFF;
        uniforms[6] = offsetSrc;
        uniforms[7] = safeOffsetDst;
        for (var d = 0; d < 8; d++) {
          uniforms[8 + d] = d < outRank ? outShape[d] : 1;
          uniforms[16 + d] = d < remStridesA.length
              ? (remStridesA[d] & 0xFFFFFFFF)
              : 0;
          uniforms[24 + d] = d < safeOutStrides.length
              ? (safeOutStrides[d] & 0xFFFFFFFF)
              : 0;
        }
        src.device.backend.dispatchComputePipeline(
          shaderModule: shaderModule,
          buffers: [src, safeDst],
          uniforms: uniforms,
          workgroupsX: dispatch.workgroupsX,
          workgroupsY: dispatch.workgroupsY,
          workgroupsZ: dispatch.workgroupsZ,
        );
      },
    );
  }

  /// Computes prefix-sum indices of non-zero elements in [src] on the GPU,
  /// returning `(prefixIndicesBuffer, nonZeroCount)`.
  static (GpuBuffer, int) executeNonZeroScan({
    required GpuBuffer src,
    required List<int> shapeSrc,
    required List<int> stridesSrc,
    required int offsetSrc,
    required DType dtypeSrc,
  }) {
    final totalElements = computeSize(shapeSrc);
    final prefixBuffer = src.device.createBuffer(
      sizeInBytes: math.max(totalElements * 4, 4),
    );
    final countBuffer = src.device.createBuffer(sizeInBytes: 4);
    try {
      if (totalElements == 0) {
        return (prefixBuffer, 0);
      }
      final shaderModule = WgslManipulationKernels.nonZeroScanShader(dtypeSrc);
      final uniforms = List<int>.filled(20, 0);
      uniforms[0] = totalElements;
      uniforms[1] = shapeSrc.length;
      uniforms[2] = offsetSrc;
      for (var d = 0; d < 8; d++) {
        uniforms[4 + d] = d < shapeSrc.length ? shapeSrc[d] : 1;
        uniforms[12 + d] = d < stridesSrc.length
            ? (stridesSrc[d] & 0xFFFFFFFF)
            : 0;
      }
      src.device.backend.dispatchComputePipeline(
        shaderModule: shaderModule,
        buffers: [src, prefixBuffer, countBuffer],
        uniforms: uniforms,
        workgroupsX: 1,
        workgroupsY: 1,
        workgroupsZ: 1,
      );
      final count = readBufferAny(countBuffer, DType.uint32, 0) as int;
      return (prefixBuffer, count);
    } finally {
      countBuffer.dispose();
    }
  }

  /// Scatters non-zero indices or values using [prefixBuffer] on the GPU.
  ///
  /// [mode] is `'flatnonzero'`, `'argwhere'`, `'nonzero_axis'`, or `'extract'`.
  static void executeNonZeroScatter({
    required String mode,
    required GpuBuffer cond,
    required List<int> shapeCond,
    required List<int> stridesCond,
    required int offsetCond,
    required DType dtypeCond,
    required GpuBuffer prefixBuffer,
    required GpuBuffer src,
    required List<int> stridesSrc,
    required int offsetSrc,
    required DType dtypeSrc,
    required GpuBuffer dst,
    required int offsetDst,
    required DType dtypeDst,
    int targetAxis = 0,
  }) {
    final totalElements = computeSize(shapeCond);
    if (totalElements == 0) return;

    final modeInt = switch (mode) {
      'flatnonzero' => 0,
      'argwhere' => 1,
      'nonzero_axis' => 2,
      'extract' => 3,
      _ => 0,
    };
    final shaderModule = WgslManipulationKernels.nonZeroScatterShader(
      dtype: modeInt == 3 ? dtypeSrc : dtypeCond,
      mode: modeInt,
    );
    final dispatch = shaderModule.calculateDispatch1D(totalElements);
    final effStrides = modeInt == 3 ? stridesSrc : stridesCond;
    final uniforms = List<int>.filled(24, 0);
    uniforms[0] = totalElements;
    uniforms[1] = shapeCond.length;
    uniforms[2] = modeInt == 3 ? offsetSrc : offsetCond;
    uniforms[3] = targetAxis;
    uniforms[4] = offsetDst;
    for (var d = 0; d < 8; d++) {
      uniforms[8 + d] = d < shapeCond.length ? shapeCond[d] : 1;
      uniforms[16 + d] = d < effStrides.length
          ? (effStrides[d] & 0xFFFFFFFF)
          : 0;
    }
    cond.device.backend.dispatchComputePipeline(
      shaderModule: shaderModule,
      buffers: [modeInt == 3 ? src : cond, prefixBuffer, dst],
      uniforms: uniforms,
      workgroupsX: dispatch.workgroupsX,
      workgroupsY: dispatch.workgroupsY,
      workgroupsZ: dispatch.workgroupsZ,
    );
  }

  /// Sorts elements or indices along [axis] on the GPU for `sort`, `argsort`,
  /// `partition`, `argpartition`, and `topk`.
  static void executeAxisSort({
    required GpuBuffer src,
    required List<int> shapeSrc,
    required List<int> stridesSrc,
    required int offsetSrc,
    required DType dtypeSrc,
    required int axis,
    required int outAxisSize,
    required bool descending,
    GpuBuffer? dstValues,
    List<int>? outValShape,
    List<int>? outValStrides,
    int offsetDstValues = 0,
    GpuBuffer? dstIndices,
    List<int>? outIndicesShape,
    List<int>? outIndicesStrides,
    int offsetDstIndices = 0,
  }) {
    final totalElements = computeSize(shapeSrc);
    if (totalElements == 0 || outAxisSize == 0) return;

    final rank = shapeSrc.length;
    final normAxis = axis < 0 ? axis + rank : axis;
    final axisSize = shapeSrc[normAxis];
    if (axisSize == 0) return;

    final numSlices = totalElements ~/ axisSize;
    if (numSlices == 0) return;

    final sliceShape = List<int>.of(shapeSrc);
    sliceShape[normAxis] = 1;

    final writeValues = dstValues != null;
    final writeIndices = dstIndices != null;

    final aliasValues = writeValues && identical(src, dstValues);
    final aliasIndices =
        writeIndices &&
        (identical(src, dstIndices) ||
            (writeValues && identical(dstValues, dstIndices)));

    final valTotal = writeValues ? computeSize(outValShape!) : 0;
    final indicesTotal = writeIndices ? computeSize(outIndicesShape!) : 0;

    GpuBuffer? tempValues;
    GpuBuffer? tempIndices;
    GpuBuffer? dummyValues;
    GpuBuffer? dummyIndices;

    try {
      final GpuBuffer effectiveValBuffer;
      final List<int> effectiveValStrides;
      final int effectiveValOffset;
      if (writeValues) {
        if (aliasValues) {
          tempValues = src.device.createBuffer(
            sizeInBytes: math.max(valTotal * dtypeSrc.byteWidth, 4),
          );
          effectiveValBuffer = tempValues;
          effectiveValStrides = computeCStrides(outValShape!);
          effectiveValOffset = 0;
        } else {
          effectiveValBuffer = dstValues;
          effectiveValStrides = outValStrides!;
          effectiveValOffset = offsetDstValues;
        }
      } else {
        dummyValues = src.device.createBuffer(sizeInBytes: 16);
        effectiveValBuffer = dummyValues;
        effectiveValStrides = List<int>.filled(rank, 0);
        effectiveValOffset = 0;
      }

      final GpuBuffer effectiveIndicesBuffer;
      final List<int> effectiveIndicesStrides;
      final int effectiveIndicesOffset;
      if (writeIndices) {
        if (aliasIndices) {
          tempIndices = src.device.createBuffer(
            sizeInBytes: math.max(indicesTotal * DType.int64.byteWidth, 8),
          );
          effectiveIndicesBuffer = tempIndices;
          effectiveIndicesStrides = computeCStrides(outIndicesShape!);
          effectiveIndicesOffset = 0;
        } else {
          effectiveIndicesBuffer = dstIndices;
          effectiveIndicesStrides = outIndicesStrides!;
          effectiveIndicesOffset = offsetDstIndices;
        }
      } else {
        dummyIndices = src.device.createBuffer(sizeInBytes: 16);
        effectiveIndicesBuffer = dummyIndices;
        effectiveIndicesStrides = List<int>.filled(rank, 0);
        effectiveIndicesOffset = 0;
      }

      final shaderModule = WgslIndexingKernels.axisSortShader(dtypeSrc);
      final dispatch = shaderModule.calculateDispatch1D(numSlices * 256);
      final uniforms = List<int>.filled(48, 0);
      uniforms[0] = numSlices;
      uniforms[1] = rank;
      uniforms[2] = normAxis;
      uniforms[3] = axisSize;
      uniforms[4] = outAxisSize;
      uniforms[5] = offsetSrc;
      uniforms[6] = effectiveValOffset;
      uniforms[7] = effectiveIndicesOffset;
      uniforms[8] = stridesSrc[normAxis] & 0xFFFFFFFF;
      uniforms[9] = effectiveValStrides[normAxis] & 0xFFFFFFFF;
      uniforms[10] = effectiveIndicesStrides[normAxis] & 0xFFFFFFFF;
      uniforms[11] = descending ? 1 : 0;
      uniforms[12] = writeValues ? 1 : 0;
      uniforms[13] = writeIndices ? 1 : 0;

      for (var d = 0; d < 8; d++) {
        uniforms[16 + d] = d < rank ? sliceShape[d] : 1;
        uniforms[24 + d] = (d < rank && d != normAxis)
            ? (stridesSrc[d] & 0xFFFFFFFF)
            : 0;
        uniforms[32 + d] = (d < rank && d != normAxis)
            ? (effectiveValStrides[d] & 0xFFFFFFFF)
            : 0;
        uniforms[40 + d] = (d < rank && d != normAxis)
            ? (effectiveIndicesStrides[d] & 0xFFFFFFFF)
            : 0;
      }

      src.device.backend.dispatchComputePipeline(
        shaderModule: shaderModule,
        buffers: [src, effectiveValBuffer, effectiveIndicesBuffer],
        uniforms: uniforms,
        workgroupsX: dispatch.workgroupsX,
        workgroupsY: dispatch.workgroupsY,
        workgroupsZ: dispatch.workgroupsZ,
      );

      if (tempValues != null) {
        copyStrided(
          src: tempValues,
          shape: outValShape!,
          strides: effectiveValStrides,
          offsetSrc: 0,
          dtypeSrc: dtypeSrc,
          dst: dstValues!,
          outStrides: outValStrides!,
          offsetDst: offsetDstValues,
          dtypeDst: dtypeSrc,
        );
      }
      if (tempIndices != null) {
        copyStrided(
          src: tempIndices,
          shape: outIndicesShape!,
          strides: effectiveIndicesStrides,
          offsetSrc: 0,
          dtypeSrc: DType.int64,
          dst: dstIndices!,
          outStrides: outIndicesStrides!,
          offsetDst: offsetDstIndices,
          dtypeDst: DType.int64,
        );
      }
    } finally {
      tempValues?.dispose();
      tempIndices?.dispose();
      dummyValues?.dispose();
      dummyIndices?.dispose();
    }
  }

  /// Executes binary search (`searchsorted`) on sorted 1-D buffer [arr] for
  /// every element of [values] on the GPU.
  static void executeSearchSorted({
    required GpuBuffer arr,
    required int lengthA,
    required int strideA,
    required int offsetA,
    required DType dtype,
    required GpuBuffer values,
    required List<int> shapeV,
    required List<int> stridesV,
    required int offsetV,
    required GpuBuffer dst,
    required List<int> outStrides,
    required int offsetDst,
    required int side,
    GpuBuffer? sorter,
    int strideSorter = 0,
    int offsetSorter = 0,
    DType sorterDType = DType.int64,
  }) {
    final totalV = computeSize(shapeV);
    if (totalV == 0) return;

    final rankV = shapeV.length;
    final inputs = <GpuBuffer>[arr, values, ?sorter];

    _withAliasSafeDst(
      inputs: inputs,
      dst: dst,
      outShape: shapeV,
      outStrides: outStrides,
      offsetDst: offsetDst,
      dtypeDst: DType.int64,
      action: (safeDst, safeOutStrides, safeOffsetDst) {
        GpuBuffer? dummySorter;
        try {
          final hasSorter = sorter != null;
          final effectiveSorter =
              sorter ?? (dummySorter = arr.device.createBuffer(sizeInBytes: 8));
          final shaderModule = WgslIndexingKernels.searchSortedShader(
            dtype: dtype,
            sorterDType: sorterDType,
          );
          final dispatch = shaderModule.calculateDispatch1D(totalV);
          final uniforms = List<int>.filled(36, 0);
          uniforms[0] = totalV;
          uniforms[1] = rankV;
          uniforms[2] = lengthA;
          uniforms[3] = side;
          uniforms[4] = hasSorter ? 1 : 0;
          uniforms[5] = offsetA;
          uniforms[6] = offsetV;
          uniforms[7] = offsetSorter;
          uniforms[8] = safeOffsetDst;
          uniforms[9] = strideA & 0xFFFFFFFF;
          uniforms[10] = strideSorter & 0xFFFFFFFF;
          for (var d = 0; d < 8; d++) {
            uniforms[12 + d] = d < rankV ? shapeV[d] : 1;
            uniforms[20 + d] = d < rankV ? (stridesV[d] & 0xFFFFFFFF) : 0;
            uniforms[28 + d] = d < rankV ? (safeOutStrides[d] & 0xFFFFFFFF) : 0;
          }
          arr.device.backend.dispatchComputePipeline(
            shaderModule: shaderModule,
            buffers: [arr, values, effectiveSorter, safeDst],
            uniforms: uniforms,
            workgroupsX: dispatch.workgroupsX,
            workgroupsY: dispatch.workgroupsY,
            workgroupsZ: dispatch.workgroupsZ,
          );
        } finally {
          dummySorter?.dispose();
        }
      },
    );
  }

  /// Executes GPU stream-compaction `unique` over [numRows] rows of length
  /// [rowLength] in contiguous buffer [src] of [dtype].
  static (
    GpuBuffer values,
    GpuBuffer indices,
    GpuBuffer inverse,
    GpuBuffer counts,
    int numUnique,
  )
  executeUnique({
    required GpuBuffer src,
    required int numRows,
    required int rowLength,
    required DType dtype,
  }) {
    final device = src.device;
    if (numRows == 0 || rowLength == 0) {
      final valuesBuffer = device.createBuffer(sizeInBytes: 4);
      final indicesBuffer = device.createBuffer(sizeInBytes: 8);
      final inverseBuffer = device.createBuffer(
        sizeInBytes: math.max(numRows * 8, 8),
      );
      if (numRows > 0) {
        executeFill(
          dst: inverseBuffer,
          outShape: [numRows],
          outStrides: const [1],
          offsetDst: 0,
          dtypeDst: DType.int64,
          value: 0,
        );
      }
      final countsBuffer = device.createBuffer(sizeInBytes: 8);
      return (valuesBuffer, indicesBuffer, inverseBuffer, countsBuffer, 0);
    }

    final sortedIndexBuffer = device.createBuffer(
      sizeInBytes: math.max(numRows * 4, 4),
    );
    final groupIdBuffer = device.createBuffer(
      sizeInBytes: math.max(numRows * 4, 4),
    );
    final headPositionBuffer = device.createBuffer(
      sizeInBytes: math.max(numRows * 4, 4),
    );
    final countBuffer = device.createBuffer(sizeInBytes: 4);

    try {
      final sortShader = WgslIndexingKernels.uniqueRowSortShader(dtype);
      device.backend.dispatchComputePipeline(
        shaderModule: sortShader,
        buffers: [src, sortedIndexBuffer],
        uniforms: [numRows, rowLength, 0, 0],
        workgroupsX: 1,
        workgroupsY: 1,
        workgroupsZ: 1,
      );

      final scanShader = WgslIndexingKernels.uniqueMarkScanShader(dtype);
      device.backend.dispatchComputePipeline(
        shaderModule: scanShader,
        buffers: [
          src,
          sortedIndexBuffer,
          groupIdBuffer,
          headPositionBuffer,
          countBuffer,
        ],
        uniforms: [numRows, rowLength, 0, 0],
        workgroupsX: 1,
        workgroupsY: 1,
        workgroupsZ: 1,
      );

      final numUnique = readBufferAny(countBuffer, DType.uint32, 0) as int;
      final valuesBuffer = device.createBuffer(
        sizeInBytes: math.max(numUnique * rowLength * dtype.byteWidth, 4),
      );
      final indicesBuffer = device.createBuffer(
        sizeInBytes: math.max(numUnique * 8, 8),
      );
      final inverseBuffer = device.createBuffer(
        sizeInBytes: math.max(numRows * 8, 8),
      );
      final countsBuffer = device.createBuffer(
        sizeInBytes: math.max(numUnique * 8, 8),
      );

      final totalThreads = math.max(numRows, numUnique * rowLength);
      if (totalThreads > 0) {
        final scatterShader = WgslIndexingKernels.uniqueScatterShader(dtype);
        final dispatch = scatterShader.calculateDispatch1D(totalThreads);
        device.backend.dispatchComputePipeline(
          shaderModule: scatterShader,
          buffers: [
            src,
            sortedIndexBuffer,
            groupIdBuffer,
            headPositionBuffer,
            valuesBuffer,
            indicesBuffer,
            inverseBuffer,
            countsBuffer,
          ],
          uniforms: [numRows, numUnique, rowLength, 0],
          workgroupsX: dispatch.workgroupsX,
          workgroupsY: dispatch.workgroupsY,
          workgroupsZ: dispatch.workgroupsZ,
        );
      }

      return (
        valuesBuffer,
        indicesBuffer,
        inverseBuffer,
        countsBuffer,
        numUnique,
      );
    } finally {
      sortedIndexBuffer.dispose();
      groupIdBuffer.dispose();
      headPositionBuffer.dispose();
      countBuffer.dispose();
    }
  }

  /// Executes `bincount` on the GPU, writing `Int64` counts or `Float64`
  /// weighted sums into [dst].
  static void executeBincount({
    required GpuBuffer x,
    required int lengthX,
    required int strideX,
    required int offsetX,
    required DType dtypeX,
    required GpuBuffer dst,
    required int outLength,
    required int strideOut,
    required int offsetDst,
    GpuBuffer? weights,
    int strideWeights = 0,
    int offsetWeights = 0,
    DType? dtypeWeights,
  }) {
    if (outLength == 0) return;

    final outDType = dtypeWeights != null ? DType.float64 : DType.int64;
    final inputs = <GpuBuffer>[x, ?weights];

    _withAliasSafeDst(
      inputs: inputs,
      dst: dst,
      outShape: [outLength],
      outStrides: [strideOut],
      offsetDst: offsetDst,
      dtypeDst: outDType,
      action: (safeDst, safeOutStrides, safeOffsetDst) {
        final shaderModule = WgslIndexingKernels.bincountShader(
          xDType: dtypeX,
          weightsDType: dtypeWeights,
        );
        final dispatch = shaderModule.calculateDispatch1D(outLength);
        final uniforms = <int>[
          outLength,
          lengthX,
          offsetX,
          offsetWeights,
          safeOffsetDst,
          strideX & 0xFFFFFFFF,
          strideWeights & 0xFFFFFFFF,
          safeOutStrides[0] & 0xFFFFFFFF,
        ];
        x.device.backend.dispatchComputePipeline(
          shaderModule: shaderModule,
          buffers: [x, ?weights, safeDst],
          uniforms: uniforms,
          workgroupsX: dispatch.workgroupsX,
          workgroupsY: dispatch.workgroupsY,
          workgroupsZ: dispatch.workgroupsZ,
        );
      },
    );
  }

  /// Executes an inclusive cumulative scan (`cumsum` or `cumprod`) along [axis]
  /// on the GPU.
  static void executeCumulativeScan({
    required String op,
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
    required int axis,
  }) {
    final totalElements = computeSize(outShape);
    if (totalElements == 0) return;

    final rank = outShape.length;
    final normAxis = axis < 0 ? axis + rank : axis;
    final axisSize = outShape[normAxis];
    if (axisSize == 0) return;
    final numSlices = totalElements ~/ axisSize;
    if (numSlices == 0) return;

    final sliceShape = List<int>.of(outShape);
    sliceShape[normAxis] = 1;

    _withAliasSafeDst(
      inputs: [src],
      dst: dst,
      outShape: outShape,
      outStrides: outStrides,
      offsetDst: offsetDst,
      dtypeDst: dtypeDst,
      action: (safeDst, safeOutStrides, safeOffsetDst) {
        GpuBuffer? castSrc;
        try {
          var effectiveSrc = src;
          var effectiveStrides = stridesSrc;
          var effectiveOffset = offsetSrc;
          if (dtypeSrc != dtypeDst) {
            castSrc = src.device.createBuffer(
              sizeInBytes: math.max(totalElements * dtypeDst.byteWidth, 4),
            );
            final contiguousStrides = computeCStrides(shapeSrc);
            copyStrided(
              src: src,
              shape: shapeSrc,
              strides: stridesSrc,
              offsetSrc: offsetSrc,
              dtypeSrc: dtypeSrc,
              dst: castSrc,
              outStrides: contiguousStrides,
              offsetDst: 0,
              dtypeDst: dtypeDst,
            );
            effectiveSrc = castSrc;
            effectiveStrides = contiguousStrides;
            effectiveOffset = 0;
          }

          final shaderModule = WgslIndexingKernels.cumulativeScanShader(
            op: op,
            dtype: dtypeDst,
          );
          final dispatch = shaderModule.calculateDispatch1D(numSlices * 256);
          final uniforms = List<int>.filled(32, 0);
          uniforms[0] = numSlices;
          uniforms[1] = rank;
          uniforms[2] = axisSize;
          uniforms[3] = effectiveOffset;
          uniforms[4] = safeOffsetDst;
          uniforms[5] = effectiveStrides[normAxis] & 0xFFFFFFFF;
          uniforms[6] = safeOutStrides[normAxis] & 0xFFFFFFFF;
          for (var d = 0; d < 8; d++) {
            uniforms[8 + d] = d < rank ? sliceShape[d] : 1;
            uniforms[16 + d] = (d < rank && d != normAxis)
                ? (effectiveStrides[d] & 0xFFFFFFFF)
                : 0;
            uniforms[24 + d] = (d < rank && d != normAxis)
                ? (safeOutStrides[d] & 0xFFFFFFFF)
                : 0;
          }
          src.device.backend.dispatchComputePipeline(
            shaderModule: shaderModule,
            buffers: [effectiveSrc, safeDst],
            uniforms: uniforms,
            workgroupsX: dispatch.workgroupsX,
            workgroupsY: dispatch.workgroupsY,
            workgroupsZ: dispatch.workgroupsZ,
          );
        } finally {
          castSrc?.dispose();
        }
      },
    );
  }

  /// Executes a single-order discrete difference along [axis] on the GPU.
  static void executeDiff1({
    required GpuBuffer src,
    required List<int> stridesSrc,
    required int offsetSrc,
    required DType dtype,
    required GpuBuffer dst,
    required List<int> outShape,
    required List<int> outStrides,
    required int offsetDst,
    required int axis,
  }) {
    final totalElements = computeSize(outShape);
    if (totalElements == 0) return;

    final rank = outShape.length;
    final normAxis = axis < 0 ? axis + rank : axis;

    _withAliasSafeDst(
      inputs: [src],
      dst: dst,
      outShape: outShape,
      outStrides: outStrides,
      offsetDst: offsetDst,
      dtypeDst: dtype,
      action: (safeDst, safeOutStrides, safeOffsetDst) {
        final shaderModule = WgslIndexingKernels.diffShader(dtype);
        final dispatch = shaderModule.calculateDispatch1D(totalElements);
        final uniforms = List<int>.filled(32, 0);
        uniforms[0] = totalElements;
        uniforms[1] = rank;
        uniforms[2] = offsetSrc;
        uniforms[3] = safeOffsetDst;
        uniforms[4] = stridesSrc[normAxis] & 0xFFFFFFFF;
        for (var d = 0; d < 8; d++) {
          uniforms[8 + d] = d < rank ? outShape[d] : 1;
          uniforms[16 + d] = d < rank ? (stridesSrc[d] & 0xFFFFFFFF) : 0;
          uniforms[24 + d] = d < rank ? (safeOutStrides[d] & 0xFFFFFFFF) : 0;
        }
        src.device.backend.dispatchComputePipeline(
          shaderModule: shaderModule,
          buffers: [src, safeDst],
          uniforms: uniforms,
          workgroupsX: dispatch.workgroupsX,
          workgroupsY: dispatch.workgroupsY,
          workgroupsZ: dispatch.workgroupsZ,
        );
      },
    );
  }
}
