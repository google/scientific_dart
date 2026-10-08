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
import '../device.dart';
import '../dtype.dart';
import '../exceptions.dart';
import '../gpu_array.dart';
import 'fft_post_wgsl.dart';
import 'fft_wgsl.dart';

/// Normalization mode for Fast Fourier Transform operations.
enum FftNorm {
  /// No normalization on forward transforms; scales inverse transforms by `1/n`.
  backward,

  /// Scales both forward and inverse transforms by `1/sqrt(n)`.
  ortho,

  /// Scales forward transforms by `1/n`; no normalization on inverse transforms.
  forward;

  /// Normalization multiplier for a forward transform of length [n].
  double forwardFactor(int n) => switch (this) {
    FftNorm.backward => 1.0,
    FftNorm.ortho => 1.0 / math.sqrt(n),
    FftNorm.forward => 1.0 / n,
  };

  /// Normalization multiplier for an inverse transform of length [n].
  double inverseFactor(int n) => switch (this) {
    FftNorm.backward => 1.0 / n,
    FftNorm.ortho => 1.0 / math.sqrt(n),
    FftNorm.forward => 1.0,
  };
}

const int _directDftMaxLength = 256;
const int _workgroupSize = 64;

int _workgroupsFor(int totalItems) =>
    totalItems <= 0 ? 1 : (totalItems + _workgroupSize - 1) ~/ _workgroupSize;

int _float32Bits(double value) {
  final byteData = ByteData(4)..setFloat32(0, value, Endian.little);
  return byteData.getUint32(0, Endian.little);
}

bool _isPowerOfTwo(int value) => value > 0 && (value & (value - 1)) == 0;

int _log2Exact(int value) {
  var power = 0;
  var current = value;
  while (current > 1) {
    current >>= 1;
    power++;
  }
  return power;
}

int _nextPowerOfTwo(int value) {
  var power = 1;
  while (power < value) {
    power <<= 1;
  }
  return power;
}

void _checkInputAlive(GpuArray array, String name, [GpuArray? out]) {
  if (array.isDisposed) {
    throw StateError('Cannot execute FFT operation on disposed input $name.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Output GpuArray has already been disposed.');
  }
  if (array.ndim == 0) {
    throw ArgumentError.value(
      array.shape,
      name,
      'Must have at least 1 dimension',
    );
  }
  if (array.ndim > 8) {
    throw ArgumentError.value(
      array.shape,
      name,
      'Must have rank at most 8 for GPU FFT shaders',
    );
  }
}

int _resolveAxis(int axis, int rank) {
  final resolved = axis < 0 ? rank + axis : axis;
  if (resolved < 0 || resolved >= rank) {
    throw GpuAxisOutOfBoundsException(axis, rank);
  }
  return resolved;
}

List<int> _resolveAxes(List<int> axes, int rank) {
  if (axes.isEmpty) {
    throw ArgumentError.value(axes, 'axes', 'Must not be empty');
  }
  return [for (final axis in axes) _resolveAxis(axis, rank)];
}

void _validateOut<T extends DTypeTag>(
  GpuArray<T>? out,
  List<int> expectedShape,
  DType<T> expectedDType,
  GpuDevice expectedDevice,
) {
  if (out == null) return;
  if (out.isDisposed) {
    throw StateError('Output GpuArray has already been disposed.');
  }
  if (out.size > 1 && out.strides.contains(0)) {
    throw UnsupportedError('Cannot write FFT output into a broadcasted view.');
  }
  if (!identical(out.device, expectedDevice)) {
    throw ArgumentError.value(
      out,
      'out',
      'Must reside on the same GpuDevice as input (${expectedDevice.name})',
    );
  }
  if (out.dtype != expectedDType) {
    throw ArgumentError.value(
      out.dtype,
      'out',
      'Must have dtype ${expectedDType.name}, got ${out.dtype.name}',
    );
  }
  if (out.shape.length != expectedShape.length) {
    throw GpuShapeMismatchException('fft', expectedShape, out.shape);
  }
  for (var i = 0; i < expectedShape.length; i++) {
    if (out.shape[i] != expectedShape[i]) {
      throw GpuShapeMismatchException('fft', expectedShape, out.shape);
    }
  }
}

({
  int batchCount,
  int outerRank,
  List<int> outerShape,
  List<int> inputOuterStrides,
  int inputAxisStride,
})
_computeAxisLayout(GpuArray input, int resolvedAxis) {
  final outerShape = <int>[];
  final inputOuterStrides = <int>[];
  var batchCount = 1;
  for (var d = 0; d < input.ndim; d++) {
    if (d == resolvedAxis) continue;
    final dimSize = input.shape[d];
    outerShape.add(dimSize);
    inputOuterStrides.add(input.strides[d]);
    batchCount *= dimSize;
  }
  if (outerShape.isEmpty) {
    outerShape.add(1);
    inputOuterStrides.add(0);
  }
  return (
    batchCount: batchCount,
    outerRank: outerShape.length,
    outerShape: outerShape,
    inputOuterStrides: inputOuterStrides,
    inputAxisStride: input.strides[resolvedAxis],
  );
}

List<int> _computeOutOuterStrides(GpuArray out, int resolvedAxis) {
  final outOuterStrides = <int>[];
  for (var d = 0; d < out.ndim; d++) {
    if (d == resolvedAxis) continue;
    outOuterStrides.add(out.strides[d]);
  }
  if (outOuterStrides.isEmpty) {
    outOuterStrides.add(0);
  }
  return outOuterStrides;
}

List<int> _packVec8U32(List<int> values, {int defaultFill = 1}) {
  return [
    for (var i = 0; i < 8; i++)
      (i < values.length ? values[i] : defaultFill) & 0xFFFFFFFF,
  ];
}

List<int> _packVec8I32(List<int> values) {
  return [
    for (var i = 0; i < 8; i++)
      (i < values.length ? values[i] : 0) & 0xFFFFFFFF,
  ];
}

bool _isSinglePrecisionFftDType(DType dtype) => switch (dtype) {
  DType.float32 || DType.complex64 => true,
  _ => false,
};

DType<C> _fftComplexDType<C extends DTypeTag>(DType dtype) => switch (dtype) {
  DType.float32 || DType.complex64 => DType.complex64 as DType<C>,
  _ => DType.complex128 as DType<C>,
};

DType<F> _fftFloatDType<F extends DTypeTag>(DType dtype) => switch (dtype) {
  DType.float32 || DType.complex64 => DType.float32 as DType<F>,
  _ => DType.float64 as DType<F>,
};

/// Executes a 1D complex DFT between contiguous `[batchCount, transformLength]`
/// complex buffers on [device], returning the buffer holding the unscaled result.
GpuBuffer _executeContiguous1dComplexDft({
  required GpuDevice device,
  required GpuBuffer inputBuffer,
  required GpuBuffer scratchBuffer,
  required int batchCount,
  required int transformLength,
  required bool inverse,
  bool singlePrecision = false,
}) {
  final totalElements = batchCount * transformLength;
  if (totalElements == 0 || transformLength <= 1) {
    return inputBuffer;
  }
  final signDir = inverse ? 1.0 : -1.0;
  final signBits = _float32Bits(signDir);

  if (_isPowerOfTwo(transformLength)) {
    final log2N = _log2Exact(transformLength);
    device.backend.dispatchComputePipeline(
      shaderModule: buildFftBitReverseShader(singlePrecision: singlePrecision),
      buffers: [inputBuffer, scratchBuffer],
      uniforms: [batchCount, transformLength, log2N, 0],
      workgroupsX: _workgroupsFor(totalElements),
    );
    var currentSource = scratchBuffer;
    var currentTarget = inputBuffer;
    final butterflyShader = buildFftButterflyStageShader(
      singlePrecision: singlePrecision,
    );
    final totalPairs = batchCount * (transformLength >> 1);
    final pairWorkgroups = _workgroupsFor(totalPairs);

    for (var stage = 0; stage < log2N; stage++) {
      final halfSpan = 1 << stage;
      device.backend.dispatchComputePipeline(
        shaderModule: butterflyShader,
        buffers: [currentSource, currentTarget],
        uniforms: [batchCount, transformLength, halfSpan, signBits],
        workgroupsX: pairWorkgroups,
      );
      final swap = currentSource;
      currentSource = currentTarget;
      currentTarget = swap;
    }
    return currentSource;
  }

  if (transformLength <= _directDftMaxLength) {
    device.backend.dispatchComputePipeline(
      shaderModule: buildFftDirectDftShader(singlePrecision: singlePrecision),
      buffers: [inputBuffer, scratchBuffer],
      uniforms: [batchCount, transformLength, signBits, 0],
      workgroupsX: _workgroupsFor(totalElements),
    );
    return scratchBuffer;
  }

  // Bluestein's Chirp-Z algorithm on GPU for large non-power-of-2 lengths.
  final chirpM = _nextPowerOfTwo(2 * transformLength - 1);
  final chirpElements = batchCount * chirpM;
  final elemBytes = singlePrecision
      ? DType.complex64.byteWidth
      : DType.complex128.byteWidth;
  final chirpBytes = chirpElements * elemBytes;
  final aPadBuffer = GpuBuffer.allocate(
    sizeInBytes: chirpBytes,
    device: device,
  );
  final bPadBuffer = GpuBuffer.allocate(
    sizeInBytes: chirpBytes,
    device: device,
  );
  final workPadBuffer = GpuBuffer.allocate(
    sizeInBytes: chirpBytes,
    device: device,
  );
  try {
    device.backend.dispatchComputePipeline(
      shaderModule: buildFftBluesteinPreShader(
        singlePrecision: singlePrecision,
      ),
      buffers: [inputBuffer, aPadBuffer, bPadBuffer],
      uniforms: [batchCount, transformLength, chirpM, signBits],
      workgroupsX: _workgroupsFor(chirpElements),
    );

    final aFftBuffer = _executeContiguous1dComplexDft(
      device: device,
      inputBuffer: aPadBuffer,
      scratchBuffer: workPadBuffer,
      batchCount: batchCount,
      transformLength: chirpM,
      inverse: false,
      singlePrecision: singlePrecision,
    );
    if (identical(aFftBuffer, workPadBuffer)) {
      device.backend.copyBufferToBuffer(workPadBuffer, aPadBuffer, chirpBytes);
    }

    final bFftBuffer = _executeContiguous1dComplexDft(
      device: device,
      inputBuffer: bPadBuffer,
      scratchBuffer: workPadBuffer,
      batchCount: batchCount,
      transformLength: chirpM,
      inverse: false,
      singlePrecision: singlePrecision,
    );

    final convFftTarget = identical(bFftBuffer, bPadBuffer)
        ? workPadBuffer
        : bPadBuffer;
    device.backend.dispatchComputePipeline(
      shaderModule: buildFftComplexMulShader(singlePrecision: singlePrecision),
      buffers: [aPadBuffer, bFftBuffer, convFftTarget],
      uniforms: [chirpElements, 0, 0, 0],
      workgroupsX: _workgroupsFor(chirpElements),
    );

    final convTimeBuffer = _executeContiguous1dComplexDft(
      device: device,
      inputBuffer: convFftTarget,
      scratchBuffer: aPadBuffer,
      batchCount: batchCount,
      transformLength: chirpM,
      inverse: true,
      singlePrecision: singlePrecision,
    );

    final (invMHi, invMLo) = encodeDoubleFloatUniform(1.0 / chirpM);
    device.backend.dispatchComputePipeline(
      shaderModule: buildFftBluesteinPostShader(
        singlePrecision: singlePrecision,
      ),
      buffers: [convTimeBuffer, scratchBuffer],
      uniforms: [
        batchCount,
        transformLength,
        chirpM,
        signBits,
        invMHi,
        invMLo,
        0,
        0,
      ],
      workgroupsX: _workgroupsFor(totalElements),
    );
    return scratchBuffer;
  } finally {
    aPadBuffer.dispose();
    bPadBuffer.dispose();
    workPadBuffer.dispose();
  }
}

GpuArray<C> _fft1dComplexInternal<C extends DTypeTag>(
  GpuArray a, {
  required int? n,
  required int axis,
  required double scaleFactor,
  required bool inverse,
  required int? truncateBins,
  required bool conjugateInput,
  required bool conjugateOutput,
  required GpuArray<C>? out,
}) {
  _checkInputAlive(a, 'a');
  final resolvedAxis = _resolveAxis(axis, a.ndim);
  final targetN = n ?? a.shape[resolvedAxis];
  if (targetN <= 0) {
    throw ArgumentError.value(n, 'n', 'Must be positive');
  }
  final outAxisLength = truncateBins ?? targetN;
  final outShape = List<int>.of(a.shape)..[resolvedAxis] = outAxisLength;
  final single = _isSinglePrecisionFftDType(a.dtype);
  final outDType = _fftComplexDType<C>(a.dtype);
  _validateOut(out, outShape, outDType, a.device);

  final destination =
      out ?? GpuArray.empty(outShape, outDType, device: a.device);
  if (destination.size == 0) {
    return destination;
  }

  final layout = _computeAxisLayout(a, resolvedAxis);
  final workElements = layout.batchCount * targetN;
  final workBytes = workElements * outDType.byteWidth;

  final gatheredBuffer = GpuBuffer.allocate(
    sizeInBytes: workBytes,
    device: a.device,
  );
  final scratchBuffer = GpuBuffer.allocate(
    sizeInBytes: workBytes,
    device: a.device,
  );
  try {
    final copyLength = math.min(a.shape[resolvedAxis], targetN);
    final gatherUniforms = <int>[
      layout.batchCount,
      targetN,
      copyLength,
      layout.outerRank,
      a.offsetElements,
      layout.inputAxisStride & 0xFFFFFFFF,
      conjugateInput ? 1 : 0,
      0,
      ..._packVec8U32(layout.outerShape),
      ..._packVec8I32(layout.inputOuterStrides),
    ];
    a.device.backend.dispatchComputePipeline(
      shaderModule: buildFftGatherShader(a.dtype, singlePrecision: single),
      buffers: [a.buffer, gatheredBuffer],
      uniforms: gatherUniforms,
      workgroupsX: _workgroupsFor(workElements),
    );

    final resultBuffer = _executeContiguous1dComplexDft(
      device: a.device,
      inputBuffer: gatheredBuffer,
      scratchBuffer: scratchBuffer,
      batchCount: layout.batchCount,
      transformLength: targetN,
      inverse: inverse,
      singlePrecision: single,
    );

    final (scaleHi, scaleLo) = single
        ? (_float32Bits(scaleFactor), 0)
        : encodeDoubleFloatUniform(scaleFactor);
    final outOuterStrides = _computeOutOuterStrides(destination, resolvedAxis);
    final scatterUniforms = <int>[
      layout.batchCount,
      targetN,
      outAxisLength,
      layout.outerRank,
      destination.offsetElements,
      destination.strides[resolvedAxis] & 0xFFFFFFFF,
      scaleHi,
      scaleLo,
      conjugateOutput ? 1 : 0,
      0,
      0,
      0,
      ..._packVec8U32(layout.outerShape),
      ..._packVec8I32(outOuterStrides),
    ];
    a.device.backend.dispatchComputePipeline(
      shaderModule: buildFftScatterComplexShader(singlePrecision: single),
      buffers: [resultBuffer, destination.buffer],
      uniforms: scatterUniforms,
      workgroupsX: _workgroupsFor(layout.batchCount * outAxisLength),
    );
    return destination;
  } finally {
    gatheredBuffer.dispose();
    scratchBuffer.dispose();
  }
}

GpuArray<F> _hermitianToReal1dInternal<F extends DTypeTag>(
  GpuArray a, {
  required int? n,
  required int axis,
  required double scaleFactor,
  required bool conjugateInput,
  required GpuArray<F>? out,
}) {
  _checkInputAlive(a, 'a');
  final resolvedAxis = _resolveAxis(axis, a.ndim);
  final inputAxisLength = a.shape[resolvedAxis];
  final targetN = n ?? (2 * (inputAxisLength - 1));
  if (targetN <= 0) {
    throw ArgumentError.value(n, 'n', 'Must be positive');
  }
  final outShape = List<int>.of(a.shape)..[resolvedAxis] = targetN;
  final single = _isSinglePrecisionFftDType(a.dtype);
  final outDType = _fftFloatDType<F>(a.dtype);
  _validateOut(out, outShape, outDType, a.device);

  final destination =
      out ?? GpuArray.empty(outShape, outDType, device: a.device);
  if (destination.size == 0) {
    return destination;
  }

  final layout = _computeAxisLayout(a, resolvedAxis);
  final halfN = (targetN ~/ 2) + 1;
  final copyBins = math.min(inputAxisLength, halfN);
  final gatherBins = math.max(1, copyBins);
  final gatherElements = layout.batchCount * gatherBins;
  final workElements = layout.batchCount * targetN;
  final complexBytes = single
      ? DType.complex64.byteWidth
      : DType.complex128.byteWidth;

  final gatherBuffer = GpuBuffer.allocate(
    sizeInBytes: gatherElements * complexBytes,
    device: a.device,
  );
  final spectrumBuffer = GpuBuffer.allocate(
    sizeInBytes: workElements * complexBytes,
    device: a.device,
  );
  final scratchBuffer = GpuBuffer.allocate(
    sizeInBytes: workElements * complexBytes,
    device: a.device,
  );
  try {
    final gatherUniforms = <int>[
      layout.batchCount,
      gatherBins,
      copyBins,
      layout.outerRank,
      a.offsetElements,
      layout.inputAxisStride & 0xFFFFFFFF,
      conjugateInput ? 1 : 0,
      0,
      ..._packVec8U32(layout.outerShape),
      ..._packVec8I32(layout.inputOuterStrides),
    ];
    a.device.backend.dispatchComputePipeline(
      shaderModule: buildFftGatherShader(a.dtype, singlePrecision: single),
      buffers: [a.buffer, gatherBuffer],
      uniforms: gatherUniforms,
      workgroupsX: _workgroupsFor(gatherElements),
    );

    a.device.backend.dispatchComputePipeline(
      shaderModule: buildFftHermitianExpandShader(singlePrecision: single),
      buffers: [gatherBuffer, spectrumBuffer],
      uniforms: [layout.batchCount, gatherBins, targetN, halfN],
      workgroupsX: _workgroupsFor(workElements),
    );

    final resultBuffer = _executeContiguous1dComplexDft(
      device: a.device,
      inputBuffer: spectrumBuffer,
      scratchBuffer: scratchBuffer,
      batchCount: layout.batchCount,
      transformLength: targetN,
      inverse: true,
      singlePrecision: single,
    );

    final (scaleHi, scaleLo) = single
        ? (_float32Bits(scaleFactor), 0)
        : encodeDoubleFloatUniform(scaleFactor);
    final outOuterStrides = _computeOutOuterStrides(destination, resolvedAxis);
    final scatterUniforms = <int>[
      layout.batchCount,
      targetN,
      layout.outerRank,
      destination.offsetElements,
      destination.strides[resolvedAxis] & 0xFFFFFFFF,
      scaleHi,
      scaleLo,
      0,
      ..._packVec8U32(layout.outerShape),
      ..._packVec8I32(outOuterStrides),
    ];
    a.device.backend.dispatchComputePipeline(
      shaderModule: buildFftScatterRealShader(singlePrecision: single),
      buffers: [resultBuffer, destination.buffer],
      uniforms: scatterUniforms,
      workgroupsX: _workgroupsFor(workElements),
    );
    return destination;
  } finally {
    gatherBuffer.dispose();
    spectrumBuffer.dispose();
    scratchBuffer.dispose();
  }
}

/// Computes the 1D discrete Fourier Transform along [axis] on the GPU.
///
/// The transform length [n] must be positive when provided.
/// If [out] is provided, the result is written into [out] and returned.
GpuArray<C> fft<
  R extends DTypeTag,
  E,
  F extends DTypeTag,
  C extends DTypeTag,
  M extends DTypeTag,
  S extends DTypeTag,
  D extends DTypeTag
>(
  GpuArray<DTypeSpec<R, E, F, C, M, S, D, DTypeTag>> a, {
  int? n,
  int axis = -1,
  FftNorm norm = FftNorm.backward,
  GpuArray<C>? out,
}) {
  _checkInputAlive(a, 'a', out);
  final resolvedAxis = _resolveAxis(axis, a.ndim);
  final targetN = n ?? a.shape[resolvedAxis];
  if (targetN <= 0) {
    throw ArgumentError.value(n, 'n', 'Must be positive');
  }
  return _fft1dComplexInternal<C>(
    a,
    n: targetN,
    axis: resolvedAxis,
    scaleFactor: norm.forwardFactor(targetN),
    inverse: false,
    truncateBins: null,
    conjugateInput: false,
    conjugateOutput: false,
    out: out,
  );
}

/// Computes the 1D inverse discrete Fourier Transform along [axis] on the GPU.
///
/// The transform length [n] must be positive when provided.
/// If [out] is provided, the result is written into [out] and returned.
GpuArray<C> ifft<
  R extends DTypeTag,
  E,
  F extends DTypeTag,
  C extends DTypeTag,
  M extends DTypeTag,
  S extends DTypeTag,
  D extends DTypeTag
>(
  GpuArray<DTypeSpec<R, E, F, C, M, S, D, DTypeTag>> a, {
  int? n,
  int axis = -1,
  FftNorm norm = FftNorm.backward,
  GpuArray<C>? out,
}) {
  _checkInputAlive(a, 'a', out);
  final resolvedAxis = _resolveAxis(axis, a.ndim);
  final targetN = n ?? a.shape[resolvedAxis];
  if (targetN <= 0) {
    throw ArgumentError.value(n, 'n', 'Must be positive');
  }
  return _fft1dComplexInternal<C>(
    a,
    n: targetN,
    axis: resolvedAxis,
    scaleFactor: norm.inverseFactor(targetN),
    inverse: true,
    truncateBins: null,
    conjugateInput: false,
    conjugateOutput: false,
    out: out,
  );
}

/// Computes the 1D discrete Fourier Transform of a real-valued input [a] along [axis].
///
/// The input [a] must have a real-valued data type, and [n] must be positive when provided.
/// Produces `(n ~/ 2) + 1` non-redundant Hermitian frequency bins along [axis].
GpuArray<C> rfft<
  R extends DTypeTag,
  E,
  F extends DTypeTag,
  C extends DTypeTag,
  M extends DTypeTag,
  S extends DTypeTag,
  D extends DTypeTag
>(
  GpuArray<DTypeSpec<R, E, F, C, M, S, D, DTypeTag>> a, {
  int? n,
  int axis = -1,
  FftNorm norm = FftNorm.backward,
  GpuArray<C>? out,
}) {
  _checkInputAlive(a, 'a', out);
  if (a.dtype.isComplex) {
    throw ArgumentError.value(
      a.dtype,
      'a',
      'Must be a real-valued array for rfft',
    );
  }
  final resolvedAxis = _resolveAxis(axis, a.ndim);
  final targetN = n ?? a.shape[resolvedAxis];
  if (targetN <= 0) {
    throw ArgumentError.value(n, 'n', 'Must be positive');
  }
  final outBins = (targetN ~/ 2) + 1;
  return _fft1dComplexInternal<C>(
    a,
    n: targetN,
    axis: resolvedAxis,
    scaleFactor: norm.forwardFactor(targetN),
    inverse: false,
    truncateBins: outBins,
    conjugateInput: false,
    conjugateOutput: false,
    out: out,
  );
}

/// Computes the inverse of [rfft], reconstructing a real-valued signal of length [n] along [axis].
///
/// When [n] is omitted, defaults to `2 * (a.shape[axis] - 1)`. The resulting [n] must be positive.
GpuArray<F> irfft<
  R extends DTypeTag,
  E,
  F extends DTypeTag,
  C extends DTypeTag,
  M extends DTypeTag,
  S extends DTypeTag,
  D extends DTypeTag
>(
  GpuArray<DTypeSpec<R, E, F, C, M, S, D, DTypeTag>> a, {
  int? n,
  int axis = -1,
  FftNorm norm = FftNorm.backward,
  GpuArray<F>? out,
}) {
  _checkInputAlive(a, 'a', out);
  final resolvedAxis = _resolveAxis(axis, a.ndim);
  final targetN = n ?? (2 * (a.shape[resolvedAxis] - 1));
  if (targetN <= 0) {
    throw ArgumentError.value(n, 'n', 'Must be positive');
  }
  return _hermitianToReal1dInternal<F>(
    a,
    n: targetN,
    axis: resolvedAxis,
    scaleFactor: norm.inverseFactor(targetN),
    conjugateInput: false,
    out: out,
  );
}

/// Computes the 1D FFT of a signal [a] that has Hermitian symmetry in the time domain,
/// producing a real-valued spectrum of length [n] along [axis].
///
/// When [n] is omitted, defaults to `2 * (a.shape[axis] - 1)`. The resulting [n] must be positive.
GpuArray<F> hfft<
  R extends DTypeTag,
  E,
  F extends DTypeTag,
  C extends DTypeTag,
  M extends DTypeTag,
  S extends DTypeTag,
  D extends DTypeTag
>(
  GpuArray<DTypeSpec<R, E, F, C, M, S, D, DTypeTag>> a, {
  int? n,
  int axis = -1,
  FftNorm norm = FftNorm.backward,
  GpuArray<F>? out,
}) {
  _checkInputAlive(a, 'a', out);
  final resolvedAxis = _resolveAxis(axis, a.ndim);
  final targetN = n ?? (2 * (a.shape[resolvedAxis] - 1));
  if (targetN <= 0) {
    throw ArgumentError.value(n, 'n', 'Must be positive');
  }
  return _hermitianToReal1dInternal<F>(
    a,
    n: targetN,
    axis: resolvedAxis,
    scaleFactor: norm.forwardFactor(targetN),
    conjugateInput: true,
    out: out,
  );
}

/// Computes the inverse FFT of a real-valued spectrum [a], producing `(n ~/ 2) + 1`
/// Hermitian-symmetric complex coefficients along [axis].
///
/// The input [a] must have a real-valued data type, and [n] must be positive when provided.
GpuArray<C> ihfft<
  R extends DTypeTag,
  E,
  F extends DTypeTag,
  C extends DTypeTag,
  M extends DTypeTag,
  S extends DTypeTag,
  D extends DTypeTag
>(
  GpuArray<DTypeSpec<R, E, F, C, M, S, D, DTypeTag>> a, {
  int? n,
  int axis = -1,
  FftNorm norm = FftNorm.backward,
  GpuArray<C>? out,
}) {
  _checkInputAlive(a, 'a', out);
  if (a.dtype.isComplex) {
    throw ArgumentError.value(
      a.dtype,
      'a',
      'Must be a real-valued array for ihfft',
    );
  }
  final resolvedAxis = _resolveAxis(axis, a.ndim);
  final targetN = n ?? a.shape[resolvedAxis];
  if (targetN <= 0) {
    throw ArgumentError.value(n, 'n', 'Must be positive');
  }
  final outBins = (targetN ~/ 2) + 1;
  return _fft1dComplexInternal<C>(
    a,
    n: targetN,
    axis: resolvedAxis,
    scaleFactor: norm.inverseFactor(targetN),
    inverse: false,
    truncateBins: outBins,
    conjugateInput: false,
    conjugateOutput: true,
    out: out,
  );
}

({List<int> resolvedAxes, List<int> lengths}) _resolveNdTransformAxesAndLengths(
  GpuArray a, {
  required List<int>? s,
  required List<int>? axes,
  required int? requiredCount,
  required bool lastAxisIsHermitianInverse,
}) {
  if (requiredCount != null && a.ndim < requiredCount) {
    throw ArgumentError.value(
      a.shape,
      'a',
      'Must have at least $requiredCount dimensions',
    );
  }
  if (axes != null && requiredCount != null && axes.length != requiredCount) {
    throw ArgumentError.value(
      axes,
      'axes',
      'Must contain exactly $requiredCount axes',
    );
  }
  if (s != null && requiredCount != null && s.length != requiredCount) {
    throw ArgumentError.value(
      s,
      's',
      'Must contain exactly $requiredCount lengths',
    );
  }

  final List<int> resolvedAxes;
  if (axes != null) {
    resolvedAxes = _resolveAxes(axes, a.ndim);
  } else if (s != null) {
    if (s.isEmpty || s.length > a.ndim) {
      throw ArgumentError.value(
        s,
        's',
        'Must have between 1 and ${a.ndim} lengths',
      );
    }
    final startAxis = a.ndim - s.length;
    resolvedAxes = [for (var i = 0; i < s.length; i++) startAxis + i];
  } else {
    resolvedAxes = [for (var i = 0; i < a.ndim; i++) i];
  }

  if (s != null && s.length != resolvedAxes.length) {
    throw ArgumentError.value(
      s,
      's',
      'Must match length of axes (${resolvedAxes.length})',
    );
  }

  final lengths = <int>[];
  for (var i = 0; i < resolvedAxes.length; i++) {
    final ax = resolvedAxes[i];
    final int length;
    if (s != null) {
      length = s[i];
    } else if (lastAxisIsHermitianInverse && i == resolvedAxes.length - 1) {
      length = 2 * (a.shape[ax] - 1);
    } else {
      length = a.shape[ax];
    }
    if (length <= 0) {
      throw ArgumentError.value(s, 's', 'Must contain positive lengths');
    }
    lengths.add(length);
  }
  return (resolvedAxes: resolvedAxes, lengths: lengths);
}

/// Computes the N-dimensional discrete Fourier Transform over [axes] on the GPU.
///
/// When [axes] is omitted, transforms all axes (or the last `s.length` axes when [s] is given).
GpuArray<C> fftn<
  R extends DTypeTag,
  E,
  F extends DTypeTag,
  C extends DTypeTag,
  M extends DTypeTag,
  S extends DTypeTag,
  D extends DTypeTag
>(
  GpuArray<DTypeSpec<R, E, F, C, M, S, D, DTypeTag>> a, {
  List<int>? s,
  List<int>? axes,
  FftNorm norm = FftNorm.backward,
  GpuArray<C>? out,
}) {
  _checkInputAlive(a, 'a', out);
  final plan = _resolveNdTransformAxesAndLengths(
    a,
    s: s,
    axes: axes,
    requiredCount: null,
    lastAxisIsHermitianInverse: false,
  );
  final outShape = List<int>.of(a.shape);
  for (var i = 0; i < plan.resolvedAxes.length; i++) {
    outShape[plan.resolvedAxes[i]] = plan.lengths[i];
  }
  final outDType = _fftComplexDType<C>(a.dtype);
  _validateOut(out, outShape, outDType, a.device);

  return ResourceScope.scope(() {
    GpuArray<DTypeTag> current = a;
    for (var i = 0; i < plan.resolvedAxes.length; i++) {
      final isLast = i == plan.resolvedAxes.length - 1;
      current = _fft1dComplexInternal<C>(
        current,
        n: plan.lengths[i],
        axis: plan.resolvedAxes[i],
        scaleFactor: norm.forwardFactor(plan.lengths[i]),
        inverse: false,
        truncateBins: null,
        conjugateInput: false,
        conjugateOutput: false,
        out: isLast ? out : null,
      );
    }
    final result = current as GpuArray<C>;
    if (out == null) {
      result.detachToParentScope();
    }
    return result;
  });
}

/// Computes the N-dimensional inverse discrete Fourier Transform over [axes] on the GPU.
///
/// When [axes] is omitted, transforms all axes (or the last `s.length` axes when [s] is given).
GpuArray<C> ifftn<
  R extends DTypeTag,
  E,
  F extends DTypeTag,
  C extends DTypeTag,
  M extends DTypeTag,
  S extends DTypeTag,
  D extends DTypeTag
>(
  GpuArray<DTypeSpec<R, E, F, C, M, S, D, DTypeTag>> a, {
  List<int>? s,
  List<int>? axes,
  FftNorm norm = FftNorm.backward,
  GpuArray<C>? out,
}) {
  _checkInputAlive(a, 'a', out);
  final plan = _resolveNdTransformAxesAndLengths(
    a,
    s: s,
    axes: axes,
    requiredCount: null,
    lastAxisIsHermitianInverse: false,
  );
  final outShape = List<int>.of(a.shape);
  for (var i = 0; i < plan.resolvedAxes.length; i++) {
    outShape[plan.resolvedAxes[i]] = plan.lengths[i];
  }
  final outDType = _fftComplexDType<C>(a.dtype);
  _validateOut(out, outShape, outDType, a.device);

  return ResourceScope.scope(() {
    GpuArray<DTypeTag> current = a;
    for (var i = 0; i < plan.resolvedAxes.length; i++) {
      final isLast = i == plan.resolvedAxes.length - 1;
      current = _fft1dComplexInternal<C>(
        current,
        n: plan.lengths[i],
        axis: plan.resolvedAxes[i],
        scaleFactor: norm.inverseFactor(plan.lengths[i]),
        inverse: true,
        truncateBins: null,
        conjugateInput: false,
        conjugateOutput: false,
        out: isLast ? out : null,
      );
    }
    final result = current as GpuArray<C>;
    if (out == null) {
      result.detachToParentScope();
    }
    return result;
  });
}

/// Computes the N-dimensional discrete Fourier Transform of a real-valued input [a] on the GPU.
///
/// Transforms the last axis in [axes] via [rfft] and all preceding axes via [fft].
GpuArray<C> rfftn<
  R extends DTypeTag,
  E,
  F extends DTypeTag,
  C extends DTypeTag,
  M extends DTypeTag,
  S extends DTypeTag,
  D extends DTypeTag
>(
  GpuArray<DTypeSpec<R, E, F, C, M, S, D, DTypeTag>> a, {
  List<int>? s,
  List<int>? axes,
  FftNorm norm = FftNorm.backward,
  GpuArray<C>? out,
}) {
  _checkInputAlive(a, 'a', out);
  if (a.dtype.isComplex) {
    throw ArgumentError.value(
      a.dtype,
      'a',
      'Must be a real-valued array for rfftn',
    );
  }
  final plan = _resolveNdTransformAxesAndLengths(
    a,
    s: s,
    axes: axes,
    requiredCount: null,
    lastAxisIsHermitianInverse: false,
  );
  final outShape = List<int>.of(a.shape);
  for (var i = 0; i < plan.resolvedAxes.length - 1; i++) {
    outShape[plan.resolvedAxes[i]] = plan.lengths[i];
  }
  final lastAxisPosition = plan.resolvedAxes.length - 1;
  outShape[plan.resolvedAxes[lastAxisPosition]] =
      (plan.lengths[lastAxisPosition] ~/ 2) + 1;
  final outDType = _fftComplexDType<C>(a.dtype);
  _validateOut(out, outShape, outDType, a.device);

  return ResourceScope.scope(() {
    GpuArray<C> current = rfft(
      a,
      n: plan.lengths[lastAxisPosition],
      axis: plan.resolvedAxes[lastAxisPosition],
      norm: norm,
      out: plan.resolvedAxes.length == 1 ? out : null,
    );
    for (var i = 0; i < lastAxisPosition; i++) {
      final isFinalPass = i == lastAxisPosition - 1;
      current = _fft1dComplexInternal<C>(
        current,
        n: plan.lengths[i],
        axis: plan.resolvedAxes[i],
        scaleFactor: norm.forwardFactor(plan.lengths[i]),
        inverse: false,
        truncateBins: null,
        conjugateInput: false,
        conjugateOutput: false,
        out: isFinalPass ? out : null,
      );
    }
    if (out == null) {
      current.detachToParentScope();
    }
    return current;
  });
}

/// Computes the inverse of [rfftn], reconstructing a real-valued N-D array on the GPU.
///
/// Transforms all axes in [axes] except the last via [ifft], and the last axis via [irfft].
GpuArray<F> irfftn<
  R extends DTypeTag,
  E,
  F extends DTypeTag,
  C extends DTypeTag,
  M extends DTypeTag,
  S extends DTypeTag,
  D extends DTypeTag
>(
  GpuArray<DTypeSpec<R, E, F, C, M, S, D, DTypeTag>> a, {
  List<int>? s,
  List<int>? axes,
  FftNorm norm = FftNorm.backward,
  GpuArray<F>? out,
}) {
  _checkInputAlive(a, 'a', out);
  final plan = _resolveNdTransformAxesAndLengths(
    a,
    s: s,
    axes: axes,
    requiredCount: null,
    lastAxisIsHermitianInverse: true,
  );
  final outShape = List<int>.of(a.shape);
  for (var i = 0; i < plan.resolvedAxes.length; i++) {
    outShape[plan.resolvedAxes[i]] = plan.lengths[i];
  }
  final outDType = _fftFloatDType<F>(a.dtype);
  _validateOut(out, outShape, outDType, a.device);

  return ResourceScope.scope(() {
    GpuArray<DTypeTag> current = a;
    final lastAxisPosition = plan.resolvedAxes.length - 1;
    for (var i = 0; i < lastAxisPosition; i++) {
      current = _fft1dComplexInternal<C>(
        current,
        n: plan.lengths[i],
        axis: plan.resolvedAxes[i],
        scaleFactor: norm.inverseFactor(plan.lengths[i]),
        inverse: true,
        truncateBins: null,
        conjugateInput: false,
        conjugateOutput: false,
        out: null,
      );
    }
    final result = _hermitianToReal1dInternal<F>(
      current,
      n: plan.lengths[lastAxisPosition],
      axis: plan.resolvedAxes[lastAxisPosition],
      scaleFactor: norm.inverseFactor(plan.lengths[lastAxisPosition]),
      conjugateInput: false,
      out: out,
    );
    if (out == null) {
      result.detachToParentScope();
    }
    return result;
  });
}

/// Computes the 2D discrete Fourier Transform over [axes] on the GPU.
///
/// The input [a] must have at least 2 dimensions and [axes] must contain 2 axes.
GpuArray<C> fft2<
  R extends DTypeTag,
  E,
  F extends DTypeTag,
  C extends DTypeTag,
  M extends DTypeTag,
  S extends DTypeTag,
  D extends DTypeTag
>(
  GpuArray<DTypeSpec<R, E, F, C, M, S, D, DTypeTag>> a, {
  List<int>? s,
  List<int> axes = const [-2, -1],
  FftNorm norm = FftNorm.backward,
  GpuArray<C>? out,
}) {
  _checkInputAlive(a, 'a', out);
  _resolveNdTransformAxesAndLengths(
    a,
    s: s,
    axes: axes,
    requiredCount: 2,
    lastAxisIsHermitianInverse: false,
  );
  return fftn(a, s: s, axes: axes, norm: norm, out: out);
}

/// Computes the 2D inverse discrete Fourier Transform over [axes] on the GPU.
///
/// The input [a] must have at least 2 dimensions and [axes] must contain 2 axes.
GpuArray<C> ifft2<
  R extends DTypeTag,
  E,
  F extends DTypeTag,
  C extends DTypeTag,
  M extends DTypeTag,
  S extends DTypeTag,
  D extends DTypeTag
>(
  GpuArray<DTypeSpec<R, E, F, C, M, S, D, DTypeTag>> a, {
  List<int>? s,
  List<int> axes = const [-2, -1],
  FftNorm norm = FftNorm.backward,
  GpuArray<C>? out,
}) {
  _checkInputAlive(a, 'a', out);
  _resolveNdTransformAxesAndLengths(
    a,
    s: s,
    axes: axes,
    requiredCount: 2,
    lastAxisIsHermitianInverse: false,
  );
  return ifftn(a, s: s, axes: axes, norm: norm, out: out);
}

/// Computes the 2D discrete Fourier Transform of a real-valued array [a] over [axes] on the GPU.
///
/// The input [a] must be real-valued with at least 2 dimensions, and [axes] must contain 2 axes.
GpuArray<C> rfft2<
  R extends DTypeTag,
  E,
  F extends DTypeTag,
  C extends DTypeTag,
  M extends DTypeTag,
  S extends DTypeTag,
  D extends DTypeTag
>(
  GpuArray<DTypeSpec<R, E, F, C, M, S, D, DTypeTag>> a, {
  List<int>? s,
  List<int> axes = const [-2, -1],
  FftNorm norm = FftNorm.backward,
  GpuArray<C>? out,
}) {
  _checkInputAlive(a, 'a', out);
  _resolveNdTransformAxesAndLengths(
    a,
    s: s,
    axes: axes,
    requiredCount: 2,
    lastAxisIsHermitianInverse: false,
  );
  return rfftn(a, s: s, axes: axes, norm: norm, out: out);
}

/// Computes the inverse of [rfft2], reconstructing a 2D real-valued array over [axes] on the GPU.
///
/// The input [a] must have at least 2 dimensions and [axes] must contain 2 axes.
GpuArray<F> irfft2<
  R extends DTypeTag,
  E,
  F extends DTypeTag,
  C extends DTypeTag,
  M extends DTypeTag,
  S extends DTypeTag,
  D extends DTypeTag
>(
  GpuArray<DTypeSpec<R, E, F, C, M, S, D, DTypeTag>> a, {
  List<int>? s,
  List<int> axes = const [-2, -1],
  FftNorm norm = FftNorm.backward,
  GpuArray<F>? out,
}) {
  _checkInputAlive(a, 'a', out);
  _resolveNdTransformAxesAndLengths(
    a,
    s: s,
    axes: axes,
    requiredCount: 2,
    lastAxisIsHermitianInverse: true,
  );
  return irfftn(a, s: s, axes: axes, norm: norm, out: out);
}

/// Discrete Fourier Transform sample frequencies for a window of length [n] and spacing [d].
///
/// Both [n] and [d] must be positive. Executes directly on the GPU.
GpuArray<Float64> fftfreq(
  int n, {
  double d = 1.0,
  GpuDevice? device,
  GpuArray<Float64>? out,
}) {
  if (out != null && out.isDisposed) {
    throw StateError('Output GpuArray has already been disposed.');
  }
  if (n <= 0) {
    throw ArgumentError.value(n, 'n', 'Must be positive');
  }
  if (d <= 0.0 || d.isNaN) {
    throw ArgumentError.value(d, 'd', 'Must be positive');
  }
  final targetDevice = device ?? out?.device ?? GpuDevice.defaultDevice;
  _validateOut(out, [n], DType.float64, targetDevice);
  final destination =
      out ?? GpuArray.empty([n], DType.float64, device: targetDevice);
  final positiveCount = (n - 1) ~/ 2 + 1;
  final (denomHi, denomLo) = encodeDoubleFloatUniform(d * n);
  targetDevice.backend.dispatchComputePipeline(
    shaderModule: buildFftFreqShader(),
    buffers: [destination.buffer],
    uniforms: [
      n,
      n,
      positiveCount,
      0,
      destination.offsetElements,
      destination.strides[0] & 0xFFFFFFFF,
      denomHi,
      denomLo,
    ],
    workgroupsX: _workgroupsFor(n),
  );
  return destination;
}

/// Discrete Fourier Transform sample frequencies for [rfft] of window length [n] and spacing [d].
///
/// Both [n] and [d] must be positive. Executes directly on the GPU.
GpuArray<Float64> rfftfreq(
  int n, {
  double d = 1.0,
  GpuDevice? device,
  GpuArray<Float64>? out,
}) {
  if (out != null && out.isDisposed) {
    throw StateError('Output GpuArray has already been disposed.');
  }
  if (n <= 0) {
    throw ArgumentError.value(n, 'n', 'Must be positive');
  }
  if (d <= 0.0 || d.isNaN) {
    throw ArgumentError.value(d, 'd', 'Must be positive');
  }
  final outLength = (n ~/ 2) + 1;
  final targetDevice = device ?? out?.device ?? GpuDevice.defaultDevice;
  _validateOut(out, [outLength], DType.float64, targetDevice);
  final destination =
      out ?? GpuArray.empty([outLength], DType.float64, device: targetDevice);
  final (denomHi, denomLo) = encodeDoubleFloatUniform(d * n);
  targetDevice.backend.dispatchComputePipeline(
    shaderModule: buildFftFreqShader(),
    buffers: [destination.buffer],
    uniforms: [
      outLength,
      n,
      outLength,
      1,
      destination.offsetElements,
      destination.strides[0] & 0xFFFFFFFF,
      denomHi,
      denomLo,
    ],
    workgroupsX: _workgroupsFor(outLength),
  );
  return destination;
}

List<int> _normalizeShiftAxes(Object? axes, int rank) {
  if (axes == null) {
    return [for (var i = 0; i < rank; i++) i];
  }
  if (axes is int) {
    return [_resolveAxis(axes, rank)];
  }
  if (axes is List<int>) {
    return _resolveAxes(axes, rank);
  }
  throw ArgumentError.value(
    axes,
    'axes',
    'Must be null, an int, or a List<int>',
  );
}

GpuArray<T> _dispatchShift<T extends DTypeTag>(
  GpuArray<T> a, {
  required Object? axes,
  required bool inverse,
  required GpuArray<T>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot shift a disposed GpuArray.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Output GpuArray has already been disposed.');
  }
  if (a.ndim > 8) {
    throw ArgumentError.value(
      a.shape,
      'a',
      'Must have rank at most 8 for GPU shift shaders',
    );
  }
  _validateOut(out, a.shape, a.dtype, a.device);
  final destination = out ?? GpuArray.zeros(a.shape, a.dtype, device: a.device);
  if (a.size == 0) {
    return destination;
  }
  if (a.ndim == 0) {
    a.copy(out: destination);
    return destination;
  }
  final resolvedAxes = _normalizeShiftAxes(axes, a.ndim);
  final shiftsPerDim = List<int>.filled(a.ndim, 0);
  for (final ax in resolvedAxes) {
    final dimSize = a.shape[ax];
    if (dimSize > 1) {
      final shiftAmount = inverse ? (dimSize - (dimSize ~/ 2)) : (dimSize ~/ 2);
      shiftsPerDim[ax] = (shiftsPerDim[ax] + shiftAmount) % dimSize;
    }
  }
  if (out != null && a.dtype.byteWidth < 4) {
    final zeroFill = GpuArray.zeros(a.shape, a.dtype, device: a.device);
    try {
      zeroFill.copy(out: destination);
    } finally {
      zeroFill.dispose();
    }
  }
  final uniforms = <int>[
    a.size,
    a.ndim,
    a.offsetElements,
    destination.offsetElements,
    ..._packVec8U32(a.shape),
    ..._packVec8U32(shiftsPerDim, defaultFill: 0),
    ..._packVec8I32(a.strides),
    ..._packVec8I32(destination.strides),
  ];
  a.device.backend.dispatchComputePipeline(
    shaderModule: buildFftShiftShader(a.dtype),
    buffers: [a.buffer, destination.buffer],
    uniforms: uniforms,
    workgroupsX: _workgroupsFor(a.size),
  );
  return destination;
}

/// Shifts the zero-frequency component to the center of the spectrum along [axes] on the GPU.
///
/// The [axes] parameter may be `null` (all axes), an `int`, or a `List<int>`.
GpuArray<T> fftshift<T extends DTypeTag>(
  GpuArray<T> a, {
  Object? axes,
  GpuArray<T>? out,
}) => _dispatchShift(a, axes: axes, inverse: false, out: out);

/// Inverse of [fftshift], shifting the zero-frequency component back to index 0 along [axes] on the GPU.
///
/// The [axes] parameter may be `null` (all axes), an `int`, or a `List<int>`.
GpuArray<T> ifftshift<T extends DTypeTag>(
  GpuArray<T> a, {
  Object? axes,
  GpuArray<T>? out,
}) => _dispatchShift(a, axes: axes, inverse: true, out: out);
