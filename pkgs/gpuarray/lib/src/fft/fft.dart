import 'dart:ffi' as ffi;
import 'dart:math' as math;

import 'package:ndarray/ndarray.dart'
    show Complex128, DType, DTypeTag, Float64, NDArray;
import 'package:ndarray/operations.dart' as nd;
import 'package:resource_scope/resource_scope.dart';

import '../device.dart';
import '../exceptions.dart';
import '../gpu_array.dart';
import '../operations/manipulation.dart' as manip;

/// Normalization mode for Discrete Fourier Transform operations.
enum FftNorm {
  /// Unnormalized forward transform; inverse transform scaled by $1/N$.
  backward,

  /// Unitary (orthonormal) transform; both forward and inverse transforms
  /// scaled by $1/\sqrt{N}$.
  ortho,

  /// Forward transform scaled by $1/N$; unnormalized inverse transform.
  forward;

  /// Multiplier applied to the output of `package:ndarray`'s forward FFT
  /// (which is unnormalized by default) for transform size [length].
  double forwardFactor(int length) => switch (this) {
    FftNorm.backward => 1.0,
    FftNorm.ortho => 1.0 / math.sqrt(length),
    FftNorm.forward => 1.0 / length,
  };

  /// Multiplier applied to the output of `package:ndarray`'s inverse FFT
  /// (which already includes a $1/N$ factor) for transform size [length].
  double inverseFactor(int length) => switch (this) {
    FftNorm.backward => 1.0,
    FftNorm.ortho => math.sqrt(length),
    FftNorm.forward => length.toDouble(),
  };
}

bool _shapesEqual(List<int> first, List<int> second) {
  if (first.length != second.length) return false;
  for (var i = 0; i < first.length; i++) {
    if (first[i] != second[i]) return false;
  }
  return true;
}

void _scaleArrayInPlace<T extends DTypeTag>(NDArray<T> array, double factor) {
  if (factor == 1.0 || array.size == 0) return;
  final totalElements = array.size;
  switch (array.dtype) {
    case DType.complex128:
      final doubles = array.pointer.cast<ffi.Double>();
      final count = totalElements * 2;
      for (var i = 0; i < count; i++) {
        doubles[i] = doubles[i] * factor;
      }
    case DType.float64:
      final doubles = array.pointer.cast<ffi.Double>();
      for (var i = 0; i < totalElements; i++) {
        doubles[i] = doubles[i] * factor;
      }
    case DType.complex64:
      final floats = array.pointer.cast<ffi.Float>();
      final count = totalElements * 2;
      for (var i = 0; i < count; i++) {
        floats[i] = floats[i] * factor;
      }
    case DType.float32:
      final floats = array.pointer.cast<ffi.Float>();
      for (var i = 0; i < totalElements; i++) {
        floats[i] = floats[i] * factor;
      }
    default:
      break;
  }
}

GpuArray<R> _writeOrWrapResult<R extends DTypeTag>(
  NDArray<R> hostResult,
  GpuDevice device,
  GpuArray<R>? out,
) {
  if (out != null) {
    if (!_shapesEqual(out.shape, hostResult.shape) ||
        out.dtype != hostResult.dtype) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have shape ${hostResult.shape} and dtype ${hostResult.dtype}, '
            'got shape ${out.shape} and dtype ${out.dtype}.',
      );
    }
    if (out.isContiguous) {
      final contiguous = hostResult.isContiguous
          ? hostResult
          : hostResult.copy();
      if (out.byteSize > 0) {
        out.buffer.copyFromHost(
          contiguous.pointer.cast<ffi.Void>(),
          out.byteSize,
          offset: out.offsetElements * out.dtype.byteWidth,
        );
      }
    } else {
      out.buffer.ensureHostSynced();
      final outView = NDArray<R>.fromBuffer(
        out.buffer.address.cast<ffi.Void>(),
        offsetBytes: out.offsetElements * out.dtype.byteWidth,
        shape: out.shape,
        strides: out.strides,
        dtype: out.dtype,
      );
      hostResult.copy(out: outView);
      out.buffer.markHostModified();
    }
    return out;
  }
  return ResourceScope.scope(() {
    final gpuResult = GpuArray<R>.fromNDArray(hostResult, device: device);
    gpuResult.detachToParentScope();
    return gpuResult;
  });
}

int _resolveAxis(int axis, int rank) {
  final normalized = axis < 0 ? axis + rank : axis;
  if (normalized < 0 || normalized >= rank) {
    throw GpuAxisOutOfBoundsException(axis, rank);
  }
  return normalized;
}

/// Computes the 1D Discrete Fourier Transform of [a] along [axis].
///
/// If [n] is provided, the input along [axis] is truncated or zero-padded to
/// length [n] before computing the transform. The normalization convention is
/// controlled by [norm] (defaulting to [FftNorm.backward]).
///
/// Both [a] and [out] (if provided) must not be disposed, [a] must have at
/// least 1 dimension, and [n] (if provided) must be positive.
GpuArray<Complex128> fft<T extends DTypeTag>(
  GpuArray<T> a, {
  int? n,
  int axis = -1,
  FftNorm norm = FftNorm.backward,
  GpuArray<Complex128>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute fft on a disposed GpuArray.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write fft result to a disposed output GpuArray.');
  }
  if (a.rank == 0) {
    throw ArgumentError.value(
      a.shape,
      'a',
      'Must have at least 1 dimension for fft.',
    );
  }
  if (n != null && n <= 0) {
    throw ArgumentError.value(n, 'n', 'Must be positive.');
  }
  final normalizedAxis = _resolveAxis(axis, a.rank);
  final transformLength = n ?? a.shape[normalizedAxis];

  return NDArray.scope(() {
    final hostInput = a.toNDArray();
    final hostResult = nd.fft(hostInput, n: n, axis: normalizedAxis);
    _scaleArrayInPlace(hostResult, norm.forwardFactor(transformLength));
    return _writeOrWrapResult(hostResult, a.device, out);
  });
}

/// Computes the 1D Inverse Discrete Fourier Transform of [a] along [axis].
///
/// If [n] is provided, the input along [axis] is truncated or zero-padded to
/// length [n] before computing the inverse transform. The normalization
/// convention is controlled by [norm] (defaulting to [FftNorm.backward]).
///
/// Both [a] and [out] (if provided) must not be disposed, [a] must have at
/// least 1 dimension, and [n] (if provided) must be positive.
GpuArray<Complex128> ifft<T extends DTypeTag>(
  GpuArray<T> a, {
  int? n,
  int axis = -1,
  FftNorm norm = FftNorm.backward,
  GpuArray<Complex128>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute ifft on a disposed GpuArray.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write ifft result to a disposed output GpuArray.');
  }
  if (a.rank == 0) {
    throw ArgumentError.value(
      a.shape,
      'a',
      'Must have at least 1 dimension for ifft.',
    );
  }
  if (n != null && n <= 0) {
    throw ArgumentError.value(n, 'n', 'Must be positive.');
  }
  final normalizedAxis = _resolveAxis(axis, a.rank);
  final transformLength = n ?? a.shape[normalizedAxis];

  return NDArray.scope(() {
    final hostInput = a.toNDArray();
    final hostResult = nd.ifft(hostInput, n: n, axis: normalizedAxis);
    _scaleArrayInPlace(hostResult, norm.inverseFactor(transformLength));
    return _writeOrWrapResult(hostResult, a.device, out);
  });
}

/// Computes the 1D Discrete Fourier Transform of a real-valued array [a] along
/// [axis], returning the non-redundant positive frequency terms of length
/// `(n ~/ 2) + 1`.
///
/// Both [a] and [out] (if provided) must not be disposed, [a] must have at
/// least 1 dimension and a real dtype, and [n] (if provided) must be positive.
GpuArray<Complex128> rfft<T extends DTypeTag>(
  GpuArray<T> a, {
  int? n,
  int axis = -1,
  FftNorm norm = FftNorm.backward,
  GpuArray<Complex128>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute rfft on a disposed GpuArray.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write rfft result to a disposed output GpuArray.');
  }
  if (a.rank == 0) {
    throw ArgumentError.value(
      a.shape,
      'a',
      'Must have at least 1 dimension for rfft.',
    );
  }
  if (n != null && n <= 0) {
    throw ArgumentError.value(n, 'n', 'Must be positive.');
  }
  final normalizedAxis = _resolveAxis(axis, a.rank);
  final transformLength = n ?? a.shape[normalizedAxis];

  return NDArray.scope(() {
    final hostInput = a.toNDArray();
    final hostResult = nd.rfft(hostInput, n: n, axis: normalizedAxis);
    _scaleArrayInPlace(hostResult, norm.forwardFactor(transformLength));
    return _writeOrWrapResult(hostResult, a.device, out);
  });
}

/// Computes the inverse of [rfft], transforming a Hermitian-symmetric complex
/// spectrum [a] into a real-valued [GpuArray] of [Float64].
///
/// If [n] is omitted, the output length along [axis] defaults to
/// `2 * (a.shape[axis] - 1)`.
///
/// Both [a] and [out] (if provided) must not be disposed, [a] must have at
/// least 1 dimension, and the resolved output length [n] must be positive.
GpuArray<Float64> irfft<T extends DTypeTag>(
  GpuArray<T> a, {
  int? n,
  int axis = -1,
  FftNorm norm = FftNorm.backward,
  GpuArray<Float64>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute irfft on a disposed GpuArray.');
  }
  if (out != null && out.isDisposed) {
    throw StateError(
      'Cannot write irfft result to a disposed output GpuArray.',
    );
  }
  if (a.rank == 0) {
    throw ArgumentError.value(
      a.shape,
      'a',
      'Must have at least 1 dimension for irfft.',
    );
  }
  if (n != null && n <= 0) {
    throw ArgumentError.value(n, 'n', 'Must be positive.');
  }
  final normalizedAxis = _resolveAxis(axis, a.rank);
  final inputLength = a.shape[normalizedAxis];
  final transformLength = n ?? (2 * (inputLength - 1));
  if (transformLength <= 0) {
    throw ArgumentError.value(
      transformLength,
      'n',
      'Must be positive (specify n explicitly when input axis length is 1).',
    );
  }

  return NDArray.scope(() {
    final hostInput = a.toNDArray();
    final hostResult = nd.irfft(hostInput, n: n, axis: normalizedAxis);
    _scaleArrayInPlace(hostResult, norm.inverseFactor(transformLength));
    return _writeOrWrapResult(hostResult, a.device, out);
  });
}

/// Computes the 2D Discrete Fourier Transform of [a] along [axes].
///
/// Both [a] and [out] (if provided) must not be disposed, [a] must have at
/// least 2 dimensions, and [axes] must contain exactly 2 axis indices.
GpuArray<Complex128> fft2<T extends DTypeTag>(
  GpuArray<T> a, {
  List<int>? s,
  List<int> axes = const <int>[-2, -1],
  FftNorm norm = FftNorm.backward,
  GpuArray<Complex128>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute fft2 on a disposed GpuArray.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write fft2 result to a disposed output GpuArray.');
  }
  if (a.rank < 2) {
    throw ArgumentError.value(
      a.shape,
      'a',
      'Must have at least 2 dimensions for fft2.',
    );
  }
  if (axes.length != 2) {
    throw ArgumentError.value(axes, 'axes', 'Must contain exactly 2 axes.');
  }
  if (s != null) {
    if (s.length != 2) {
      throw ArgumentError.value(s, 's', 'Must contain exactly 2 lengths.');
    }
    if (s[0] <= 0 || s[1] <= 0) {
      throw ArgumentError.value(s, 's', 'Must contain positive lengths.');
    }
  }
  final axis0 = _resolveAxis(axes[0], a.rank);
  final axis1 = _resolveAxis(axes[1], a.rank);
  final length0 = s != null ? s[0] : a.shape[axis0];
  final length1 = s != null ? s[1] : a.shape[axis1];
  final totalLength = length0 * length1;

  return NDArray.scope(() {
    final hostInput = a.toNDArray();
    final hostResult = nd.fft2(hostInput, s: s, axes: <int>[axis0, axis1]);
    _scaleArrayInPlace(hostResult, norm.forwardFactor(totalLength));
    return _writeOrWrapResult(hostResult, a.device, out);
  });
}

/// Computes the 2D Inverse Discrete Fourier Transform of [a] along [axes].
///
/// Both [a] and [out] (if provided) must not be disposed, [a] must have at
/// least 2 dimensions, and [axes] must contain exactly 2 axis indices.
GpuArray<Complex128> ifft2<T extends DTypeTag>(
  GpuArray<T> a, {
  List<int>? s,
  List<int> axes = const <int>[-2, -1],
  FftNorm norm = FftNorm.backward,
  GpuArray<Complex128>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute ifft2 on a disposed GpuArray.');
  }
  if (out != null && out.isDisposed) {
    throw StateError(
      'Cannot write ifft2 result to a disposed output GpuArray.',
    );
  }
  if (a.rank < 2) {
    throw ArgumentError.value(
      a.shape,
      'a',
      'Must have at least 2 dimensions for ifft2.',
    );
  }
  if (axes.length != 2) {
    throw ArgumentError.value(axes, 'axes', 'Must contain exactly 2 axes.');
  }
  if (s != null) {
    if (s.length != 2) {
      throw ArgumentError.value(s, 's', 'Must contain exactly 2 lengths.');
    }
    if (s[0] <= 0 || s[1] <= 0) {
      throw ArgumentError.value(s, 's', 'Must contain positive lengths.');
    }
  }
  final axis0 = _resolveAxis(axes[0], a.rank);
  final axis1 = _resolveAxis(axes[1], a.rank);
  final length0 = s != null ? s[0] : a.shape[axis0];
  final length1 = s != null ? s[1] : a.shape[axis1];
  final totalLength = length0 * length1;

  return NDArray.scope(() {
    final hostInput = a.toNDArray();
    final hostResult = nd.ifft2(hostInput, s: s, axes: <int>[axis0, axis1]);
    _scaleArrayInPlace(hostResult, norm.inverseFactor(totalLength));
    return _writeOrWrapResult(hostResult, a.device, out);
  });
}

/// Returns the Discrete Fourier Transform sample frequencies for a window of
/// length [n] and sample spacing [d].
///
/// The window length [n] must be positive and [d] must be non-zero. If [out]
/// is provided, it must not be disposed and must have shape `[n]` and dtype
/// [DType.float64].
GpuArray<Float64> fftfreq(
  int n, {
  double d = 1.0,
  GpuDevice? device,
  GpuArray<Float64>? out,
}) {
  if (out != null && out.isDisposed) {
    throw StateError(
      'Cannot write fftfreq result to a disposed output GpuArray.',
    );
  }
  if (n <= 0) {
    throw ArgumentError.value(n, 'n', 'Must be positive.');
  }
  if (d == 0.0 || d.isNaN) {
    throw ArgumentError.value(d, 'd', 'Must be non-zero and finite.');
  }
  final targetDevice = out?.device ?? device ?? GpuDevice.defaultDevice;
  return NDArray.scope(() {
    final hostFreqs = nd.fftfreq(n, d: d);
    return _writeOrWrapResult(hostFreqs, targetDevice, out);
  });
}

/// Returns the Discrete Fourier Transform sample frequencies for real-input
/// transforms ([rfft]) of window length [n] and sample spacing [d].
///
/// The window length [n] must be positive and [d] must be non-zero. If [out]
/// is provided, it must not be disposed and must have shape `[(n ~/ 2) + 1]`
/// and dtype [DType.float64].
GpuArray<Float64> rfftfreq(
  int n, {
  double d = 1.0,
  GpuDevice? device,
  GpuArray<Float64>? out,
}) {
  if (out != null && out.isDisposed) {
    throw StateError(
      'Cannot write rfftfreq result to a disposed output GpuArray.',
    );
  }
  if (n <= 0) {
    throw ArgumentError.value(n, 'n', 'Must be positive.');
  }
  if (d == 0.0 || d.isNaN) {
    throw ArgumentError.value(d, 'd', 'Must be non-zero and finite.');
  }
  final targetDevice = out?.device ?? device ?? GpuDevice.defaultDevice;
  return NDArray.scope(() {
    final hostFreqs = nd.rfftfreq(n, d: d);
    return _writeOrWrapResult(hostFreqs, targetDevice, out);
  });
}

List<int> _normalizeShiftAxes(Object? axes, int rank) {
  if (axes == null) {
    return List<int>.generate(rank, (index) => index);
  }
  if (axes is int) {
    return <int>[_resolveAxis(axes, rank)];
  }
  if (axes is Iterable<int>) {
    return <int>[for (final axis in axes) _resolveAxis(axis, rank)];
  }
  throw ArgumentError.value(
    axes,
    'axes',
    'Must be an int, Iterable<int>, or null.',
  );
}

/// Shifts the zero-frequency component to the center of the spectrum along
/// [axes] (or all axes when [axes] is `null`).
///
/// Neither [x] nor [out] (if provided) may be disposed.
GpuArray<T> fftshift<T extends DTypeTag>(
  GpuArray<T> x, {
  Object? axes,
  GpuArray<T>? out,
}) {
  if (x.isDisposed) {
    throw StateError('Cannot execute fftshift on a disposed GpuArray.');
  }
  if (out != null && out.isDisposed) {
    throw StateError(
      'Cannot write fftshift result to a disposed output GpuArray.',
    );
  }
  final resolvedAxes = _normalizeShiftAxes(axes, x.rank);
  if (resolvedAxes.isEmpty) {
    return NDArray.scope(() {
      final copied = x.toNDArray().copy();
      return _writeOrWrapResult(copied, x.device, out);
    });
  }
  return ResourceScope.scope(() {
    var current = x;
    for (final axis in resolvedAxes) {
      final shiftAmount = x.shape[axis] ~/ 2;
      current = manip.roll(current, shiftAmount, axis: axis);
    }
    if (out != null) {
      return NDArray.scope(() {
        final hostResult = current.toNDArray();
        return _writeOrWrapResult(hostResult, x.device, out);
      });
    }
    current.detachToParentScope();
    return current;
  });
}

/// Inverse of [fftshift], shifting the zero-frequency component back to the
/// beginning of the spectrum along [axes] (or all axes when [axes] is `null`).
///
/// Neither [x] nor [out] (if provided) may be disposed.
GpuArray<T> ifftshift<T extends DTypeTag>(
  GpuArray<T> x, {
  Object? axes,
  GpuArray<T>? out,
}) {
  if (x.isDisposed) {
    throw StateError('Cannot execute ifftshift on a disposed GpuArray.');
  }
  if (out != null && out.isDisposed) {
    throw StateError(
      'Cannot write ifftshift result to a disposed output GpuArray.',
    );
  }
  final resolvedAxes = _normalizeShiftAxes(axes, x.rank);
  if (resolvedAxes.isEmpty) {
    return NDArray.scope(() {
      final copied = x.toNDArray().copy();
      return _writeOrWrapResult(copied, x.device, out);
    });
  }
  return ResourceScope.scope(() {
    var current = x;
    for (final axis in resolvedAxes) {
      final shiftAmount = -(x.shape[axis] ~/ 2);
      current = manip.roll(current, shiftAmount, axis: axis);
    }
    if (out != null) {
      return NDArray.scope(() {
        final hostResult = current.toNDArray();
        return _writeOrWrapResult(hostResult, x.device, out);
      });
    }
    current.detachToParentScope();
    return current;
  });
}

