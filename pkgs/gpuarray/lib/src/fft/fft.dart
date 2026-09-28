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
import '../dtype.dart';
import '../gpu_array.dart';
import '../slice.dart';
import '../backend/compute_engine.dart';
import '../operations/manipulation.dart' as manip;

/// Cooley-Tukey Radix-2 1D FFT implementation.
void _fftRadix2(
  List<double> real,
  List<double> imag,
  int n, {
  required bool inverse,
}) {
  if (n <= 1) return;

  // Bit reversal permutation
  var j = 0;
  for (var i = 0; i < n - 1; i++) {
    if (i < j) {
      final tr = real[i];
      real[i] = real[j];
      real[j] = tr;
      final ti = imag[i];
      imag[i] = imag[j];
      imag[j] = ti;
    }
    var k = n >> 1;
    while (k <= j) {
      j -= k;
      k >>= 1;
    }
    j += k;
  }

  // Cooley-Tukey radix-2 butterfly
  for (var len = 2; len <= n; len <<= 1) {
    final half = len >> 1;
    final angle = (inverse ? 2.0 : -2.0) * math.pi / len;
    final wstepR = math.cos(angle);
    final wstepI = math.sin(angle);

    for (var i = 0; i < n; i += len) {
      var wr = 1.0;
      var wi = 0.0;
      for (var k = 0; k < half; k++) {
        final uR = real[i + k];
        final uI = imag[i + k];
        final vR = real[i + k + half] * wr - imag[i + k + half] * wi;
        final vI = real[i + k + half] * wi + imag[i + k + half] * wr;

        real[i + k] = uR + vR;
        imag[i + k] = uI + vI;
        real[i + k + half] = uR - vR;
        imag[i + k + half] = uI - vI;

        final nextWr = wr * wstepR - wi * wstepI;
        wi = wr * wstepI + wi * wstepR;
        wr = nextWr;
      }
    }
  }

  if (inverse) {
    for (var i = 0; i < n; i++) {
      real[i] /= n;
      imag[i] /= n;
    }
  }
}

/// Bluestein Chirp-Z transform for arbitrary non-power-of-2 FFT (O(N log N)).
void _fftBluestein(
  List<double> real,
  List<double> imag,
  int n, {
  required bool inverse,
}) {
  // Find power of 2 M >= 2*n - 1
  var m = 1;
  while (m < 2 * n - 1) {
    m <<= 1;
  }

  final aReal = List<double>.filled(m, 0.0);
  final aImag = List<double>.filled(m, 0.0);
  final bReal = List<double>.filled(m, 0.0);
  final bImag = List<double>.filled(m, 0.0);

  final chirpReal = List<double>.filled(n, 0.0);
  final chirpImag = List<double>.filled(n, 0.0);

  final sign = inverse ? 1.0 : -1.0;
  final twoN = 2 * n;

  for (var i = 0; i < n; i++) {
    final angle = sign * math.pi * ((i * i) % twoN) / n;
    final c = math.cos(angle);
    final s = math.sin(angle);
    chirpReal[i] = c;
    chirpImag[i] = s;

    // a[i] = input[i] * chirp[i]
    aReal[i] = real[i] * c - imag[i] * s;
    aImag[i] = real[i] * s + imag[i] * c;

    // b[i] = conj(chirp[i])
    bReal[i] = c;
    bImag[i] = -s;
    if (i > 0) {
      bReal[m - i] = c;
      bImag[m - i] = -s;
    }
  }

  // Forward FFTs on length M
  _fftRadix2(aReal, aImag, m, inverse: false);
  _fftRadix2(bReal, bImag, m, inverse: false);

  // Pointwise multiply: FC = FA * FB
  for (var i = 0; i < m; i++) {
    final r = aReal[i] * bReal[i] - aImag[i] * bImag[i];
    final im = aReal[i] * bImag[i] + aImag[i] * bReal[i];
    aReal[i] = r;
    aImag[i] = im;
  }

  // Inverse FFT on length M
  _fftRadix2(aReal, aImag, m, inverse: true);

  // result[k] = C[k] * chirp[k]
  for (var k = 0; k < n; k++) {
    final c = chirpReal[k];
    final s = chirpImag[k];
    real[k] = aReal[k] * c - aImag[k] * s;
    imag[k] = aReal[k] * s + aImag[k] * c;
    if (inverse) {
      real[k] /= n;
      imag[k] /= n;
    }
  }
}

/// Internal helper for 1D Complex FFT (Cooley-Tukey / Bluestein).
void _fft1d(
  List<double> real,
  List<double> imag,
  int n, {
  required bool inverse,
}) {
  if (n <= 1) return;

  if ((n & (n - 1)) == 0) {
    _fftRadix2(real, imag, n, inverse: inverse);
  } else {
    _fftBluestein(real, imag, n, inverse: inverse);
  }
}

(double, double) _readComplexOrReal(dynamic val) {
  if (val is Complex) {
    return (val.real, val.imag);
  }
  if (val is num) {
    return (val.toDouble(), 0.0);
  }
  return (0.0, 0.0);
}

/// Computes the 1D Discrete Fourier Transform of [a] along [axis].
GpuArray<T> fft<T extends DTypeTag>(
  GpuArray a, {
  int? n,
  int axis = -1,
  String? norm,
}) {
  final normAxis = axis < 0 ? axis + a.rank : axis;
  final length = n ?? a.shape[normAxis];

  final outDType = (a.dtype == DType.complex64)
      ? DType.complex64
      : DType.complex128;
  final outShape = List<int>.from(a.shape);
  outShape[normAxis] = length;

  final result = GpuArray<T>.empty(
    outShape,
    outDType as DType<T>,
    device: a.device,
  );

  final inLength = a.shape[normAxis];
  final outerSize = a.shape.sublist(0, normAxis).fold(1, (r, e) => r * e);
  final innerSize = a.shape.sublist(normAxis + 1).fold(1, (r, e) => r * e);

  final aFlat = a.toNDArray();
  final aList = aFlat.toList();

  final realBuf = List<double>.filled(length, 0.0);
  final imagBuf = List<double>.filled(length, 0.0);

  for (var o = 0; o < outerSize; o++) {
    for (var i = 0; i < innerSize; i++) {
      for (var k = 0; k < length; k++) {
        if (k < inLength) {
          final srcIdx = (o * inLength + k) * innerSize + i;
          final (r, im) = _readComplexOrReal(aList[srcIdx]);
          realBuf[k] = r;
          imagBuf[k] = im;
        } else {
          realBuf[k] = 0.0;
          imagBuf[k] = 0.0;
        }
      }

      _fft1d(realBuf, imagBuf, length, inverse: false);

      if (norm == 'ortho') {
        final scale = 1.0 / math.sqrt(length);
        for (var k = 0; k < length; k++) {
          realBuf[k] *= scale;
          imagBuf[k] *= scale;
        }
      }

      for (var k = 0; k < length; k++) {
        final dstIdx = (o * length + k) * innerSize + i;
        ComputeEngine.writeAny(
          result.buffer,
          outDType,
          dstIdx,
          Complex(realBuf[k], imagBuf[k]),
        );
      }
    }
  }

  aFlat.dispose();
  return result;
}

/// Computes the 1D Inverse Discrete Fourier Transform of [a] along [axis].
GpuArray<T> ifft<T extends DTypeTag>(
  GpuArray a, {
  int? n,
  int axis = -1,
  String? norm,
}) {
  final normAxis = axis < 0 ? axis + a.rank : axis;
  final length = n ?? a.shape[normAxis];

  final outDType = (a.dtype == DType.complex64)
      ? DType.complex64
      : DType.complex128;
  final outShape = List<int>.from(a.shape);
  outShape[normAxis] = length;

  final result = GpuArray<T>.empty(
    outShape,
    outDType as DType<T>,
    device: a.device,
  );

  final inLength = a.shape[normAxis];
  final outerSize = a.shape.sublist(0, normAxis).fold(1, (r, e) => r * e);
  final innerSize = a.shape.sublist(normAxis + 1).fold(1, (r, e) => r * e);

  final aFlat = a.toNDArray();
  final aList = aFlat.toList();

  final realBuf = List<double>.filled(length, 0.0);
  final imagBuf = List<double>.filled(length, 0.0);

  for (var o = 0; o < outerSize; o++) {
    for (var i = 0; i < innerSize; i++) {
      for (var k = 0; k < length; k++) {
        if (k < inLength) {
          final srcIdx = (o * inLength + k) * innerSize + i;
          final (r, im) = _readComplexOrReal(aList[srcIdx]);
          realBuf[k] = r;
          imagBuf[k] = im;
        } else {
          realBuf[k] = 0.0;
          imagBuf[k] = 0.0;
        }
      }

      _fft1d(realBuf, imagBuf, length, inverse: true);

      if (norm == 'ortho') {
        final scale = math.sqrt(length);
        for (var k = 0; k < length; k++) {
          realBuf[k] *= scale;
          imagBuf[k] *= scale;
        }
      }

      for (var k = 0; k < length; k++) {
        final dstIdx = (o * length + k) * innerSize + i;
        ComputeEngine.writeAny(
          result.buffer,
          outDType,
          dstIdx,
          Complex(realBuf[k], imagBuf[k]),
        );
      }
    }
  }

  aFlat.dispose();
  return result;
}

/// Computes 1D real Discrete Fourier Transform of [a].
GpuArray<T> rfft<T extends DTypeTag>(
  GpuArray a, {
  int? n,
  int axis = -1,
  String? norm,
}) {
  final normAxis = axis < 0 ? axis + a.rank : axis;
  final length = n ?? a.shape[normAxis];
  final outLen = (length ~/ 2) + 1;

  final fullFft = fft(a, n: length, axis: normAxis, norm: norm);
  final sliceSpecs = List<dynamic>.generate(
    a.rank,
    (d) => d == normAxis ? Slice(0, outLen) : const All(),
  );
  return fullFft.slice(sliceSpecs) as GpuArray<T>;
}

/// Computes inverse of 1D real Discrete Fourier Transform.
GpuArray<T> irfft<T extends DTypeTag>(
  GpuArray a, {
  int? n,
  int axis = -1,
  String? norm,
}) {
  final normAxis = axis < 0 ? axis + a.rank : axis;
  final inLen = a.shape[normAxis];
  final outLen = n ?? ((inLen - 1) * 2);

  final fullComplexShape = List<int>.from(a.shape);
  fullComplexShape[normAxis] = outLen;

  final fullComplex = GpuArray.zeros(
    fullComplexShape,
    DType.complex128,
    device: a.device,
  );

  final aFlat = a.toNDArray();
  final aList = aFlat.toList();
  final outerSize = a.shape.sublist(0, normAxis).fold(1, (r, e) => r * e);
  final innerSize = a.shape.sublist(normAxis + 1).fold(1, (r, e) => r * e);

  final numPos = math.min(inLen, (outLen ~/ 2) + 1);
  final maxMirror = (outLen - 1) ~/ 2;

  for (var o = 0; o < outerSize; o++) {
    for (var i = 0; i < innerSize; i++) {
      for (var k = 0; k < numPos; k++) {
        final srcIdx = (o * inLen + k) * innerSize + i;
        final (r, im) = _readComplexOrReal(aList[srcIdx]);
        final dstIdx = (o * outLen + k) * innerSize + i;
        ComputeEngine.writeAny(
          fullComplex.buffer,
          DType.complex128,
          dstIdx,
          Complex(r, im),
        );
      }
      for (var k = 1; k <= maxMirror; k++) {
        if (k < inLen) {
          final srcIdx = (o * inLen + k) * innerSize + i;
          final (r, im) = _readComplexOrReal(aList[srcIdx]);
          final dstIdx = (o * outLen + (outLen - k)) * innerSize + i;
          ComputeEngine.writeAny(
            fullComplex.buffer,
            DType.complex128,
            dstIdx,
            Complex(r, -im),
          );
        }
      }
    }
  }

  aFlat.dispose();

  final ifftRes = ifft(fullComplex, n: outLen, axis: normAxis, norm: norm);
  fullComplex.dispose();

  final outDType = (a.dtype == DType.complex64) ? DType.float32 : DType.float64;
  final result = GpuArray<T>.empty(
    fullComplexShape,
    outDType as DType<T>,
    device: a.device,
  );

  final ifftFlat = ifftRes.toNDArray();
  final ifftList = ifftFlat.toList();
  final totalElements = ShapeUtils.computeSize(fullComplexShape);

  for (var i = 0; i < totalElements; i++) {
    final (r, _) = _readComplexOrReal(ifftList[i]);
    ComputeEngine.writeValue(result.buffer, outDType, i, r);
  }

  ifftFlat.dispose();
  ifftRes.dispose();
  return result;
}

/// Computes 2D Discrete Fourier Transform of [a].
GpuArray<T> fft2<T extends DTypeTag>(
  GpuArray a, {
  List<int>? s,
  List<int> axes = const [-2, -1],
  String? norm,
}) {
  final ax1 = axes[0] < 0 ? axes[0] + a.rank : axes[0];
  final ax2 = axes[1] < 0 ? axes[1] + a.rank : axes[1];
  final s1 = s != null ? s[0] : null;
  final s2 = s != null ? s[1] : null;

  final res1 = fft(a, n: s1, axis: ax1, norm: norm);
  final res2 = fft(res1, n: s2, axis: ax2, norm: norm);
  res1.dispose();
  return res2 as GpuArray<T>;
}

/// Computes 2D Inverse Discrete Fourier Transform of [a].
GpuArray<T> ifft2<T extends DTypeTag>(
  GpuArray a, {
  List<int>? s,
  List<int> axes = const [-2, -1],
  String? norm,
}) {
  final ax1 = axes[0] < 0 ? axes[0] + a.rank : axes[0];
  final ax2 = axes[1] < 0 ? axes[1] + a.rank : axes[1];
  final s1 = s != null ? s[0] : null;
  final s2 = s != null ? s[1] : null;

  final res1 = ifft(a, n: s1, axis: ax1, norm: norm);
  final res2 = ifft(res1, n: s2, axis: ax2, norm: norm);
  res1.dispose();
  return res2 as GpuArray<T>;
}

/// Returns the DFT sample frequencies.
GpuArray<Float64> fftfreq(int n, {double d = 1.0}) {
  final freqs = List<double>.filled(n, 0.0);
  final factor = 1.0 / (d * n);
  final half = (n - 1) ~/ 2 + 1;

  for (var i = 0; i < half; i++) {
    freqs[i] = i * factor;
  }
  for (var i = half; i < n; i++) {
    freqs[i] = -(n - i) * factor;
  }

  return GpuArray.fromList(freqs, [n], DType.float64);
}

/// Returns the Real DFT sample frequencies.
GpuArray<Float64> rfftfreq(int n, {double d = 1.0}) {
  final outLen = (n ~/ 2) + 1;
  final freqs = List<double>.filled(outLen, 0.0);
  final factor = 1.0 / (d * n);

  for (var i = 0; i < outLen; i++) {
    freqs[i] = i * factor;
  }

  return GpuArray.fromList(freqs, [outLen], DType.float64);
}

/// Shifts zero-frequency component to center of spectrum.
///
/// [axes] can be an [int], a `List<int>`, or `null` (shifts all axes).
GpuArray<T> fftshift<T extends DTypeTag>(GpuArray<T> x, {Object? axes}) {
  if (axes == null) {
    var curr = x;
    for (var dim = 0; dim < x.rank; dim++) {
      final shift = x.shape[dim] ~/ 2;
      final prev = curr;
      curr = manip.roll(curr, shift, axis: dim);
      if (!identical(prev, x)) {
        prev.dispose();
      }
    }
    return curr;
  }
  if (axes is int) {
    final shift = x.shape[axes < 0 ? axes + x.rank : axes] ~/ 2;
    return manip.roll(x, shift, axis: axes);
  }
  if (axes is Iterable<int>) {
    var curr = x;
    for (final ax in axes) {
      final normAx = ax < 0 ? ax + x.rank : ax;
      final shift = x.shape[normAx] ~/ 2;
      final prev = curr;
      curr = manip.roll(curr, shift, axis: normAx);
      if (!identical(prev, x)) {
        prev.dispose();
      }
    }
    return curr;
  }
  throw ArgumentError(
    'Invalid axes for fftshift: $axes. Expected int, List<int>, or null.',
  );
}

/// Inverse of [fftshift].
///
/// [axes] can be an [int], a `List<int>`, or `null` (shifts all axes).
GpuArray<T> ifftshift<T extends DTypeTag>(GpuArray<T> x, {Object? axes}) {
  if (axes == null) {
    var curr = x;
    for (var dim = 0; dim < x.rank; dim++) {
      final shift = -((x.shape[dim]) ~/ 2);
      final prev = curr;
      curr = manip.roll(curr, shift, axis: dim);
      if (!identical(prev, x)) {
        prev.dispose();
      }
    }
    return curr;
  }
  if (axes is int) {
    final shift = -((x.shape[axes < 0 ? axes + x.rank : axes]) ~/ 2);
    return manip.roll(x, shift, axis: axes);
  }
  if (axes is Iterable<int>) {
    var curr = x;
    for (final ax in axes) {
      final normAx = ax < 0 ? ax + x.rank : ax;
      final shift = -((x.shape[normAx]) ~/ 2);
      final prev = curr;
      curr = manip.roll(curr, shift, axis: normAx);
      if (!identical(prev, x)) {
        prev.dispose();
      }
    }
    return curr;
  }
  throw ArgumentError(
    'Invalid axes for ifftshift: $axes. Expected int, List<int>, or null.',
  );
}
