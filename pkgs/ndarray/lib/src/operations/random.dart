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
import 'dart:math' show Random;
import '../ndarray.dart';
import 'dart:ffi' as ffi;
import '../ndarray_bindings.dart';
import '../scratch_arena.dart';

// Standalone operational relative cross-imports
import 'helpers.dart';
import 'math.dart';
import 'linalg.dart';

int _defaultSeedCounter = 0;

int _nextDefaultSeed() {
  final now = DateTime.now().microsecondsSinceEpoch;
  final counter = _defaultSeedCounter++;
  return (now ^ (counter * 0x9e3779b9)) & 0xFFFFFFFF;
}

/// A stateful pseudo-random number generator that produces reproducible
/// sequences of draws from various probability distributions.
final class RandomGenerator {
  final Random _rand;

  /// Creates a stateful random number generator with an optional [seed].
  ///
  /// If [seed] is omitted, a unique timestamp-and-counter-based seed is generated.
  RandomGenerator([int? seed])
    : _rand = seed != null ? Random(seed) : Random(_nextDefaultSeed());

  int _nextSeedVal(bool secure) {
    if (secure) return Random.secure().nextInt(4294967296);
    return _rand.nextInt(4294967296);
  }

  /// Generates an array with random values uniformly distributed in the half-open interval `[0.0, 1.0)`.
  NDArray<T> uniform<T extends AnySpec>(
    List<int> shape, {
    DType<T>? dtype,
    NDArray<T>? out,
    bool secure = false,
  }) {
    return _uniformImpl(
      shape,
      dtype: dtype,
      seedVal: _nextSeedVal(secure),
      out: out,
      secure: secure,
    );
  }

  /// Returns random integers from the half-open interval `[low, high)`.
  NDArray<T> randint<T extends AnySpec>(
    List<int> shape, {
    required int low,
    required int high,
    DType<T>? dtype,
    NDArray<T>? out,
    bool secure = false,
  }) {
    return _randintImpl(
      shape,
      low: low,
      high: high,
      dtype: dtype,
      seedVal: _nextSeedVal(secure),
      out: out,
      secure: secure,
    );
  }

  /// Draws random samples from a normal (Gaussian) distribution.
  NDArray<T> normal<T extends AnySpec>(
    List<int> shape, {
    double loc = 0.0,
    double scale = 1.0,
    DType<T>? dtype,
    NDArray<T>? out,
    bool secure = false,
  }) {
    return _normalImpl(
      shape,
      loc: loc,
      scale: scale,
      dtype: dtype,
      seedVal: _nextSeedVal(secure),
      out: out,
      secure: secure,
    );
  }

  /// Draws samples from an exponential distribution.
  NDArray<T> exponential<T extends AnySpec>(
    List<int> shape, {
    double scale = 1.0,
    double? lam,
    DType<T>? dtype,
    NDArray<T>? out,
    bool secure = false,
  }) {
    return _exponentialImpl(
      shape,
      scale: scale,
      lam: lam,
      dtype: dtype,
      seedVal: _nextSeedVal(secure),
      out: out,
      secure: secure,
    );
  }
}

void _validateOutBuffer<T extends DTypeTag>(
  NDArray<T>? out,
  List<int> expectedShape,
  DType<T> expectedDType,
) {
  if (out == null) return;
  if (out.isDisposed) {
    throw StateError('Cannot write random result to a disposed output array.');
  }
  validateOutBuffer(out);
  if (!listEquals(out.shape, expectedShape) || out.dtype != expectedDType) {
    throw ArgumentError.value(
      out,
      'out',
      'Must have compatible shape $expectedShape and dtype $expectedDType (incompatible out buffer shape or dtype, got shape ${out.shape} and dtype ${out.dtype})',
    );
  }
}

NDArray<T> _uniformImpl<T extends DTypeTag>(
  List<int> shape, {
  DType<T>? dtype,
  required int seedVal,
  NDArray<T>? out,
  bool secure = false,
}) {
  final resolvedDType = dtype ?? (out?.dtype ?? DType.float64 as DType<T>);
  if (!identical(resolvedDType, DType.float64) &&
      !identical(resolvedDType, DType.float32)) {
    throw ArgumentError.value(
      resolvedDType,
      'dtype',
      'Must be a float dtype (uniform only supports float types for now, got $resolvedDType)',
    );
  }
  _validateOutBuffer(out, shape, resolvedDType);

  if (out != null && !out.isContiguous) {
    return NDArray.scope(() {
      final temp = NDArray<T>.create(shape, resolvedDType);
      final len = temp.size;
      switch (resolvedDType) {
        case DType.float64:
          if (secure) {
            v_secure_uniform_double(temp.pointer.cast<ffi.Double>(), len);
          } else {
            v_uniform_double(temp.pointer.cast<ffi.Double>(), len, seedVal);
          }
        case DType.float32:
          if (secure) {
            v_secure_uniform_float(temp.pointer.cast<ffi.Float>(), len);
          } else {
            v_uniform_float(temp.pointer.cast<ffi.Float>(), len, seedVal);
          }
        case DType.float16:
        case DType.bfloat16:
        case DType.int64:
        case DType.int32:
        case DType.int16:
        case DType.int8:
        case DType.uint64:
        case DType.uint32:
        case DType.uint16:
        case DType.uint8:
        case DType.boolean:
        case DType.complex128:
        case DType.complex64:
          throw ArgumentError.value(
            resolvedDType,
            'resolvedDType',
            'Must be a float dtype (uniform only supports float types for now, got $resolvedDType)',
          );
      }
      temp.copy(out: out);
      return out;
    });
  }

  final arr = out ?? NDArray<T>.create(shape, resolvedDType);
  final len = arr.size;

  switch (resolvedDType) {
    case DType.float64:
      if (secure) {
        v_secure_uniform_double(arr.pointer.cast<ffi.Double>(), len);
      } else {
        v_uniform_double(arr.pointer.cast<ffi.Double>(), len, seedVal);
      }
    case DType.float32:
      if (secure) {
        v_secure_uniform_float(arr.pointer.cast<ffi.Float>(), len);
      } else {
        v_uniform_float(arr.pointer.cast<ffi.Float>(), len, seedVal);
      }
    case DType.float16:
    case DType.bfloat16:
    case DType.int64:
    case DType.int32:
    case DType.int16:
    case DType.int8:
    case DType.uint64:
    case DType.uint32:
    case DType.uint16:
    case DType.uint8:
    case DType.boolean:
    case DType.complex128:
    case DType.complex64:
      throw ArgumentError.value(
        resolvedDType,
        'resolvedDType',
        'Must be a float dtype (uniform only supports float types for now, got $resolvedDType)',
      );
  }
  return arr;
}

NDArray<T> _randintImpl<T extends DTypeTag>(
  List<int> shape, {
  required int low,
  required int high,
  DType<T>? dtype,
  required int seedVal,
  NDArray<T>? out,
  bool secure = false,
}) {
  if (low >= high) {
    throw ArgumentError.value(
      low,
      'low',
      'Must be less than high (low must be less than high, got low=$low, high=$high)',
    );
  }
  final resolvedDType = dtype ?? (out?.dtype ?? DType.int64 as DType<T>);
  if (!identical(resolvedDType, DType.int64) &&
      !identical(resolvedDType, DType.int32) &&
      !identical(resolvedDType, DType.uint8) &&
      !identical(resolvedDType, DType.int16)) {
    throw ArgumentError.value(
      resolvedDType,
      'dtype',
      'Must be an integer dtype (randint only supports integer types: int64, int32, int16, uint8, got $resolvedDType)',
    );
  }
  if (identical(resolvedDType, DType.int32)) {
    if (low < -2147483648) {
      throw ArgumentError.value(
        low,
        'low',
        'Must be >= -2147483648 for dtype int32 (got $low)',
      );
    }
    if (high > 2147483648) {
      throw ArgumentError.value(
        high,
        'high',
        'Must be <= 2147483648 for dtype int32 (got $high)',
      );
    }
  } else if (identical(resolvedDType, DType.int16)) {
    if (low < -32768) {
      throw ArgumentError.value(
        low,
        'low',
        'Must be >= -32768 for dtype int16 (got $low)',
      );
    }
    if (high > 32768) {
      throw ArgumentError.value(
        high,
        'high',
        'Must be <= 32768 for dtype int16 (got $high)',
      );
    }
  } else if (identical(resolvedDType, DType.uint8)) {
    if (low < 0) {
      throw ArgumentError.value(
        low,
        'low',
        'Must be >= 0 for dtype uint8 (got $low)',
      );
    }
    if (high > 256) {
      throw ArgumentError.value(
        high,
        'high',
        'Must be <= 256 for dtype uint8 (got $high)',
      );
    }
  }
  _validateOutBuffer(out, shape, resolvedDType);

  if (out != null && !out.isContiguous) {
    return NDArray.scope(() {
      final temp = NDArray<T>.create(shape, resolvedDType);
      final len = temp.size;
      switch (resolvedDType) {
        case DType.int64:
          if (secure) {
            v_secure_randint_int64(
              temp.pointer.cast<ffi.Int64>(),
              len,
              low,
              high,
            );
          } else {
            v_randint_int64(
              temp.pointer.cast<ffi.Int64>(),
              len,
              low,
              high,
              seedVal,
            );
          }
        case DType.int32:
          if (secure) {
            v_secure_randint_int32(
              temp.pointer.cast<ffi.Int32>(),
              len,
              low,
              high,
            );
          } else {
            v_randint_int32(
              temp.pointer.cast<ffi.Int32>(),
              len,
              low,
              high,
              seedVal,
            );
          }
        case DType.uint8:
          if (secure) {
            v_secure_randint_uint8(
              temp.pointer.cast<ffi.Uint8>(),
              len,
              low,
              high,
            );
          } else {
            v_randint_uint8(
              temp.pointer.cast<ffi.Uint8>(),
              len,
              low,
              high,
              seedVal,
            );
          }
        case DType.int16:
          if (secure) {
            v_secure_randint_int16(
              temp.pointer.cast<ffi.Int16>(),
              len,
              low,
              high,
            );
          } else {
            v_randint_int16(
              temp.pointer.cast<ffi.Int16>(),
              len,
              low,
              high,
              seedVal,
            );
          }
        case DType.float64:
        case DType.float32:
        case DType.float16:
        case DType.bfloat16:
        case DType.int8:
        case DType.uint64:
        case DType.uint32:
        case DType.uint16:
        case DType.boolean:
        case DType.complex128:
        case DType.complex64:
          throw ArgumentError.value(
            resolvedDType,
            'resolvedDType',
            'Must be an integer dtype (randint only supports integer types: int64, int32, int16, uint8, got $resolvedDType)',
          );
      }
      temp.copy(out: out);
      return out;
    });
  }

  final arr = out ?? NDArray<T>.create(shape, resolvedDType);
  final len = arr.size;

  switch (resolvedDType) {
    case DType.int64:
      if (secure) {
        v_secure_randint_int64(arr.pointer.cast<ffi.Int64>(), len, low, high);
      } else {
        v_randint_int64(arr.pointer.cast<ffi.Int64>(), len, low, high, seedVal);
      }
    case DType.int32:
      if (secure) {
        v_secure_randint_int32(arr.pointer.cast<ffi.Int32>(), len, low, high);
      } else {
        v_randint_int32(arr.pointer.cast<ffi.Int32>(), len, low, high, seedVal);
      }
    case DType.uint8:
      if (secure) {
        v_secure_randint_uint8(arr.pointer.cast<ffi.Uint8>(), len, low, high);
      } else {
        v_randint_uint8(arr.pointer.cast<ffi.Uint8>(), len, low, high, seedVal);
      }
    case DType.int16:
      if (secure) {
        v_secure_randint_int16(arr.pointer.cast<ffi.Int16>(), len, low, high);
      } else {
        v_randint_int16(arr.pointer.cast<ffi.Int16>(), len, low, high, seedVal);
      }
    case DType.float64:
    case DType.float32:
    case DType.float16:
    case DType.bfloat16:
    case DType.int8:
    case DType.uint64:
    case DType.uint32:
    case DType.uint16:
    case DType.boolean:
    case DType.complex128:
    case DType.complex64:
      throw ArgumentError.value(
        resolvedDType,
        'resolvedDType',
        'Must be an integer dtype (randint only supports integer types: int64, int32, int16, uint8, got $resolvedDType)',
      );
  }
  return arr;
}

NDArray<T> _normalImpl<T extends DTypeTag>(
  List<int> shape, {
  double loc = 0.0,
  double scale = 1.0,
  DType<T>? dtype,
  required int seedVal,
  NDArray<T>? out,
  bool secure = false,
}) {
  if (scale <= 0.0) {
    throw ArgumentError.value(
      scale,
      'scale',
      'Must be strictly positive (scale / standard deviation must be strictly positive, was $scale)',
    );
  }
  final resolvedDType = dtype ?? (out?.dtype ?? DType.float64 as DType<T>);
  if (!identical(resolvedDType, DType.float64) &&
      !identical(resolvedDType, DType.float32)) {
    throw ArgumentError.value(
      resolvedDType,
      'dtype',
      'Must be a floating point dtype (normal only supports floating point dtypes: float32/float64, got $resolvedDType)',
    );
  }
  _validateOutBuffer(out, shape, resolvedDType);

  if (out != null && !out.isContiguous) {
    return NDArray.scope(() {
      final temp = NDArray<T>.create(shape, resolvedDType);
      final len = temp.size;
      switch (resolvedDType) {
        case DType.float64:
          if (secure) {
            v_secure_normal_double(
              temp.pointer.cast<ffi.Double>(),
              len,
              loc,
              scale,
            );
          } else {
            v_normal_double(
              temp.pointer.cast<ffi.Double>(),
              len,
              loc,
              scale,
              seedVal,
            );
          }
        case DType.float32:
          if (secure) {
            v_secure_normal_float(
              temp.pointer.cast<ffi.Float>(),
              len,
              loc,
              scale,
            );
          } else {
            v_normal_float(
              temp.pointer.cast<ffi.Float>(),
              len,
              loc,
              scale,
              seedVal,
            );
          }
        case DType.float16:
        case DType.bfloat16:
        case DType.int64:
        case DType.int32:
        case DType.int16:
        case DType.int8:
        case DType.uint64:
        case DType.uint32:
        case DType.uint16:
        case DType.uint8:
        case DType.boolean:
        case DType.complex128:
        case DType.complex64:
          throw ArgumentError.value(
            resolvedDType,
            'resolvedDType',
            'Must be a floating point dtype (normal only supports floating point dtypes: float32/float64, got $resolvedDType)',
          );
      }
      temp.copy(out: out);
      return out;
    });
  }

  final arr = out ?? NDArray<T>.create(shape, resolvedDType);
  final len = arr.size;

  switch (resolvedDType) {
    case DType.float64:
      if (secure) {
        v_secure_normal_double(arr.pointer.cast<ffi.Double>(), len, loc, scale);
      } else {
        v_normal_double(
          arr.pointer.cast<ffi.Double>(),
          len,
          loc,
          scale,
          seedVal,
        );
      }
    case DType.float32:
      if (secure) {
        v_secure_normal_float(arr.pointer.cast<ffi.Float>(), len, loc, scale);
      } else {
        v_normal_float(arr.pointer.cast<ffi.Float>(), len, loc, scale, seedVal);
      }
    case DType.float16:
    case DType.bfloat16:
    case DType.int64:
    case DType.int32:
    case DType.int16:
    case DType.int8:
    case DType.uint64:
    case DType.uint32:
    case DType.uint16:
    case DType.uint8:
    case DType.boolean:
    case DType.complex128:
    case DType.complex64:
      throw ArgumentError.value(
        resolvedDType,
        'resolvedDType',
        'Must be a floating point dtype (normal only supports floating point dtypes: float32/float64, got $resolvedDType)',
      );
  }
  return arr;
}

NDArray<T> _exponentialImpl<T extends DTypeTag>(
  List<int> shape, {
  double scale = 1.0,
  double? lam,
  DType<T>? dtype,
  required int seedVal,
  NDArray<T>? out,
  bool secure = false,
}) {
  final targetScale = lam != null ? 1.0 / lam : scale;
  if (targetScale <= 0.0) {
    throw ArgumentError.value(
      targetScale,
      'scale / lam',
      'Must be strictly positive (scale parameter or 1 / lam was $targetScale)',
    );
  }
  final resolvedDType = dtype ?? (out?.dtype ?? DType.float64 as DType<T>);
  if (!identical(resolvedDType, DType.float64) &&
      !identical(resolvedDType, DType.float32)) {
    throw ArgumentError.value(
      resolvedDType,
      'dtype',
      'Must be a floating point dtype (exponential only supports floating point dtypes: float32/float64, got $resolvedDType)',
    );
  }
  _validateOutBuffer(out, shape, resolvedDType);

  if (out != null && !out.isContiguous) {
    return NDArray.scope(() {
      final temp = NDArray<T>.create(shape, resolvedDType);
      final len = temp.size;
      switch (resolvedDType) {
        case DType.float64:
          if (secure) {
            v_secure_uniform_double(temp.pointer.cast<ffi.Double>(), len);
          } else {
            v_uniform_double(temp.pointer.cast<ffi.Double>(), len, seedVal);
          }
          final ptr = temp.pointer.cast<ffi.Double>();
          for (var i = 0; i < len; i++) {
            var u = ptr[i];
            if (u >= 1.0) u = 0.9999999999999999;
            ptr[i] = -targetScale * math.log(1.0 - u);
          }
        case DType.float32:
          if (secure) {
            v_secure_uniform_float(temp.pointer.cast<ffi.Float>(), len);
          } else {
            v_uniform_float(temp.pointer.cast<ffi.Float>(), len, seedVal);
          }
          final ptr = temp.pointer.cast<ffi.Float>();
          for (var i = 0; i < len; i++) {
            var u = ptr[i];
            if (u >= 1.0) u = 0.999999;
            ptr[i] = -targetScale * math.log(1.0 - u);
          }
        case DType.float16:
        case DType.bfloat16:
        case DType.int64:
        case DType.int32:
        case DType.int16:
        case DType.int8:
        case DType.uint64:
        case DType.uint32:
        case DType.uint16:
        case DType.uint8:
        case DType.boolean:
        case DType.complex128:
        case DType.complex64:
          throw ArgumentError.value(
            resolvedDType,
            'resolvedDType',
            'Must be a floating point dtype (exponential only supports floating point dtypes: float32/float64, got $resolvedDType)',
          );
      }
      temp.copy(out: out);
      return out;
    });
  }

  final arr = out ?? NDArray<T>.create(shape, resolvedDType);
  final len = arr.size;

  switch (resolvedDType) {
    case DType.float64:
      if (secure) {
        v_secure_uniform_double(arr.pointer.cast<ffi.Double>(), len);
      } else {
        v_uniform_double(arr.pointer.cast<ffi.Double>(), len, seedVal);
      }
      final ptr = arr.pointer.cast<ffi.Double>();
      for (var i = 0; i < len; i++) {
        var u = ptr[i];
        if (u >= 1.0) u = 0.9999999999999999;
        ptr[i] = -targetScale * math.log(1.0 - u);
      }
    case DType.float32:
      if (secure) {
        v_secure_uniform_float(arr.pointer.cast<ffi.Float>(), len);
      } else {
        v_uniform_float(arr.pointer.cast<ffi.Float>(), len, seedVal);
      }
      final ptr = arr.pointer.cast<ffi.Float>();
      for (var i = 0; i < len; i++) {
        var u = ptr[i];
        if (u >= 1.0) u = 0.999999;
        ptr[i] = -targetScale * math.log(1.0 - u);
      }
    case DType.float16:
    case DType.bfloat16:
    case DType.int64:
    case DType.int32:
    case DType.int16:
    case DType.int8:
    case DType.uint64:
    case DType.uint32:
    case DType.uint16:
    case DType.uint8:
    case DType.boolean:
    case DType.complex128:
    case DType.complex64:
      throw ArgumentError.value(
        resolvedDType,
        'resolvedDType',
        'Must be a floating point dtype (exponential only supports floating point dtypes: float32/float64, got $resolvedDType)',
      );
  }
  return arr;
}

/// Generates an array with random values uniformly distributed in the half-open interval `[0.0, 1.0)`.
///
/// **Preconditions:**
/// - [dtype] must be a floating point type (DType.float32 or DType.float64).
///
/// - It is an error if the provided [dtype] is not a supported floating point type.
///
/// **Performance considerations:**
/// - Algorithmic time complexity is $O(N)$ and space complexity is $O(N)$, where $N$ is the total size of
///   the generated array (product of [shape] dimensions).
/// - Uses native C vector functions (`v_uniform_double` / `v_uniform_float`) for element generation.
///
/// **Example:**
/// {@example /example/random_example.dart lang=dart}
///
/// Refer to the [Uniform Distribution Reference](https://en.wikipedia.org/wiki/Continuous_uniform_distribution)
/// for details on continuous uniform distributions.
///
/// By default, uses Dart's standard [Random] class, which is not cryptographically secure.
/// You can request cryptographically secure generation via the [secure] parameter if needed.
NDArray<T> uniform<T extends AnySpec>(
  List<int> shape, {
  DType<T>? dtype,
  int? seed,
  NDArray<T>? out,
  bool secure = false,
}) {
  if (secure && seed != null) {
    throw ArgumentError.value(seed, 'seed', 'Must be null when secure is true');
  }
  final seedVal = secure ? 0 : (seed ?? _nextDefaultSeed());
  return _uniformImpl(
    shape,
    dtype: dtype,
    seedVal: seedVal,
    out: out,
    secure: secure,
  );
}

/// Returns random integers from the half-open interval `[low, high)`.
///
/// Generates uniformly distributed random integers of the specified integer [dtype]
/// in the range `[low, high)`.
///
/// **Preconditions:**
/// - [low] must be strictly less than [high].
/// - [dtype] must be a supported integer type (`int64`, `int32`, `int16`, or `uint8`).
/// - If provided, the [out] recycler array must exactly match the shape and compatible dtype.
///
/// - It is an error if [low] is greater than or equal to [high].
/// - It is an error if [dtype] is not a supported integer DType.
/// - It is an error if [out] has mismatched shape or dtype.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ in both time and space, where $N$ is the total size of the generated array.
/// - Uses native C vector functions (`v_randint_int64`, `v_randint_int32` etc.).
///
/// **Memory Ownership & Recycle:**
/// - If the optional [out] recycler buffer is provided, it is populated in-place, avoiding heap allocations.
/// - Otherwise, allocates a new array on the unmanaged C heap. **The caller takes full ownership** of this memory page and **must explicitly call [dispose]** to prevent native memory leaks, unless executing inside a managed [NDArray.scope()].
///
/// **NumPy Counterpart:**
/// - Equates directly to NumPy's `np.random.randint`.
///
/// **Example:**
/// {@example /example/random_example.dart lang=dart}
NDArray<T> randint<T extends AnySpec>(
  List<int> shape, {
  required int low,
  required int high,
  DType<T>? dtype,
  int? seed,
  NDArray<T>? out,
  bool secure = false,
}) {
  if (secure && seed != null) {
    throw ArgumentError.value(seed, 'seed', 'Must be null when secure is true');
  }
  final seedVal = secure ? 0 : (seed ?? _nextDefaultSeed());
  return _randintImpl(
    shape,
    low: low,
    high: high,
    dtype: dtype,
    seedVal: seedVal,
    out: out,
    secure: secure,
  );
}

/// Draws random samples from a normal (Gaussian) distribution.
///
/// This function corresponds to NumPy's `random.normal` function.
///
/// **Box-Muller Transform**:
/// It utilizes the Box-Muller transform to generate two independent
/// standard normal random scalars simultaneously, reducing transcendental function calls.
///
/// **Preconditions:**
/// - [scale] (standard deviation) must be strictly positive.
/// - [dtype] must be a floating point type (`float32` or `float64`).
///
/// - It is an error if [dtype] is not a supported floating point type.
/// - It is an error if [scale] is less than or equal to 0.0.
///
/// **Performance considerations:**
/// - Algorithmic time complexity is $O(N)$ and space complexity is $O(N)$, where $N$ is the total size of
///   the generated array.
/// - Uses native C vector functions (`v_normal_double` / `v_normal_float`) with Box-Muller transform.
///
/// **Example:**
/// {@example /example/random_example.dart lang=dart}
///
/// Refer to the [Normal Distribution Reference](https://en.wikipedia.org/wiki/Normal_distribution)
/// for details on standard Gaussian distributions.
NDArray<T> normal<T extends AnySpec>(
  List<int> shape, {
  double loc = 0.0,
  double scale = 1.0,
  DType<T>? dtype,
  int? seed,
  NDArray<T>? out,
  bool secure = false,
}) {
  if (secure && seed != null) {
    throw ArgumentError.value(seed, 'seed', 'Must be null when secure is true');
  }
  final seedVal = secure ? 0 : (seed ?? _nextDefaultSeed());
  return _normalImpl(
    shape,
    loc: loc,
    scale: scale,
    dtype: dtype,
    seedVal: seedVal,
    out: out,
    secure: secure,
  );
}

/// Draws samples from an exponential distribution.
///
/// This function corresponds to NumPy's `random.exponential` function.
/// It uses Inverse Transform Sampling to extract exponential variables.
///
/// **Preconditions:**
/// - [scale] (the inverse of the rate parameter lambda, i.e., 1/lambda) must be strictly positive.
/// - [dtype] must be a floating point type (`float32` or `float64`).
///
/// - It is an error if [dtype] is not a supported floating point type.
/// - It is an error if [scale] (or 1 / lam) is non-positive.
///
/// **Performance considerations:**
/// - Algorithmic time complexity is $O(N)$ and space complexity is $O(N)$, where $N$ is the total size of
///   the generated array.
///
/// **Example:**
/// {@example /example/random_example.dart lang=dart}
///
/// Refer to the [Exponential Distribution Reference](https://en.wikipedia.org/wiki/Exponential_distribution)
/// for details on exponential variables.
NDArray<T> exponential<T extends AnySpec>(
  List<int> shape, {
  double scale = 1.0,
  double? lam,
  DType<T>? dtype,
  int? seed,
  NDArray<T>? out,
  bool secure = false,
}) {
  if (secure && seed != null) {
    throw ArgumentError.value(seed, 'seed', 'Must be null when secure is true');
  }
  final seedVal = secure ? 0 : (seed ?? _nextDefaultSeed());
  return _exponentialImpl(
    shape,
    scale: scale,
    lam: lam,
    dtype: dtype,
    seedVal: seedVal,
    out: out,
    secure: secure,
  );
}

/// Draws samples from a Poisson distribution.
///
/// This function corresponds to NumPy's `random.poisson` function.
///
/// **Dual-Track Algorithms**:
/// - For small lambda (`lam < 30.0`), it executes Knuth's precise inversion algorithm.
/// - For large lambda (`lam >= 30.0`), Knuth's method averages `lam` steps per element
///   and suffers severe numerical float underflow. To avoid stalls and underflows, it
///   automatically switches to **Gaussian Approximation** with continuity correction.
///
/// **Preconditions:**
/// - [lam] (lambda, the rate/mean) must be strictly positive.
/// - [dtype] must be an integer type (`int32` or `int64`).
///
/// - It is an error if [dtype] is not a supported integer type.
/// - It is an error if [lam] is less than or equal to 0.0.
///
/// **Performance considerations:**
/// - Algorithmic time complexity is $O(N)$ and space complexity is $O(N)$, where $N$ is the total size of
///   the generated array.
/// - For small [lam] (< 30.0), Knuth's method iterates an average of `lam` times per element, making the runtime
///   dependent on the rate, whereas the Gaussian approximation runs in stable $O(1)$ steps per element.
///
/// **Example:**
/// {@example /example/random_example.dart lang=dart}
///
/// Refer to the [Poisson Distribution Reference](https://en.wikipedia.org/wiki/Poisson_distribution)
/// for details on Poisson processes.
NDArray<T> poisson<T extends AnySpec>(
  List<int> shape, {
  double lam = 1.0,
  DType<T>? dtype,
  int? seed,
  NDArray<T>? out,
  bool secure = false,
}) {
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write poisson result to a disposed output array.');
  }
  if (secure && seed != null) {
    throw ArgumentError.value(seed, 'seed', 'Must be null when secure is true');
  }
  if (lam <= 0.0) {
    throw ArgumentError.value(
      lam,
      'lam',
      'Must be strictly positive (lambda was $lam)',
    );
  }
  final resolvedDType = dtype ?? (out?.dtype ?? DType.int64 as DType<T>);
  if (!identical(resolvedDType, DType.int64) &&
      !identical(resolvedDType, DType.int32)) {
    throw ArgumentError.value(
      resolvedDType,
      'dtype',
      'Must be an integer dtype (poisson only supports integer dtypes: int32/int64, got $resolvedDType)',
    );
  }
  _validateOutBuffer(out, shape, resolvedDType);
  final useSecureCsprng = secure;
  final seedVal = useSecureCsprng ? 0 : (seed ?? _nextDefaultSeed());

  if (out != null && !out.isContiguous) {
    return NDArray.scope(() {
      final temp = NDArray<T>.create(shape, resolvedDType);
      final len = temp.size;
      switch (resolvedDType) {
        case DType.int64:
          if (useSecureCsprng) {
            v_secure_poisson_int64(temp.pointer.cast<ffi.Int64>(), len, lam);
          } else {
            v_poisson_int64(temp.pointer.cast<ffi.Int64>(), len, lam, seedVal);
          }
        case DType.int32:
          if (useSecureCsprng) {
            v_secure_poisson_int32(temp.pointer.cast<ffi.Int32>(), len, lam);
          } else {
            v_poisson_int32(temp.pointer.cast<ffi.Int32>(), len, lam, seedVal);
          }
        case DType.float64:
        case DType.float32:
        case DType.float16:
        case DType.bfloat16:
        case DType.int16:
        case DType.int8:
        case DType.uint64:
        case DType.uint32:
        case DType.uint16:
        case DType.uint8:
        case DType.boolean:
        case DType.complex128:
        case DType.complex64:
          throw ArgumentError.value(
            resolvedDType,
            'resolvedDType',
            'Must be an integer dtype (poisson only supports integer dtypes: int32/int64, got $resolvedDType)',
          );
      }
      temp.copy(out: out);
      return out;
    });
  }

  final arr = out ?? NDArray<T>.create(shape, resolvedDType);
  final len = arr.size;

  switch (resolvedDType) {
    case DType.int64:
      if (useSecureCsprng) {
        v_secure_poisson_int64(arr.pointer.cast<ffi.Int64>(), len, lam);
      } else {
        v_poisson_int64(arr.pointer.cast<ffi.Int64>(), len, lam, seedVal);
      }
    case DType.int32:
      if (useSecureCsprng) {
        v_secure_poisson_int32(arr.pointer.cast<ffi.Int32>(), len, lam);
      } else {
        v_poisson_int32(arr.pointer.cast<ffi.Int32>(), len, lam, seedVal);
      }
    case DType.float64:
    case DType.float32:
    case DType.float16:
    case DType.bfloat16:
    case DType.int16:
    case DType.int8:
    case DType.uint64:
    case DType.uint32:
    case DType.uint16:
    case DType.uint8:
    case DType.boolean:
    case DType.complex128:
    case DType.complex64:
      throw ArgumentError.value(
        resolvedDType,
        'resolvedDType',
        'Must be an integer dtype (poisson only supports integer dtypes: int32/int64, got $resolvedDType)',
      );
  }
  return arr;
}

/// Draws samples from a Binomial distribution.
///
/// This function corresponds to NumPy's `random.binomial` function.
///
/// **Dual-Track Algorithms**:
/// - For small `n < 50`, it directly simulates the Bernoulli trials (counts successes of [n]
///   independent random tests).
/// - For large `n >= 50`, counting `n` trials gets slow ($O(n)$). It triggers an optimized
///   **Normal Distribution Approximation** with mean `n*p` and standard deviation `sqrt(n*p*(1-p))`
///   for probabilistic simulations.
///
/// **Preconditions:**
/// - [n] (number of trials) must be non-negative.
/// - [p] (success probability) must be in the interval `[0.0, 1.0]`.
/// - [dtype] must be an integer type (`int32` or `int64`).
///
/// - It is an error if [dtype] is not a supported integer type.
/// - It is an error if [n] is negative.
/// - It is an error if [p] is less than 0.0 or greater than 1.0.
///
/// **Performance considerations:**
/// - Algorithmic time complexity is $O(N)$ and space complexity is $O(N)$, where $N$ is the total size of
///   the generated array.
/// - For small [n] (< 50), Bernoulli simulation runs in $O(n)$ loops per element. For large [n], the Normal
///   distribution approximation executes in stable $O(1)$ steps per element.
///
/// **Example:**
/// {@example /example/random_example.dart lang=dart}
///
/// Refer to the [Binomial Distribution Reference](https://en.wikipedia.org/wiki/Binomial_distribution)
/// for details on independent Bernoulli trials.
NDArray<T> binomial<T extends AnySpec>(
  List<int> shape, {
  required int n,
  required double p,
  DType<T>? dtype,
  int? seed,
  NDArray<T>? out,
  bool secure = false,
}) {
  if (out != null && out.isDisposed) {
    throw StateError(
      'Cannot write binomial result to a disposed output array.',
    );
  }
  if (secure && seed != null) {
    throw ArgumentError.value(seed, 'seed', 'Must be null when secure is true');
  }
  if (n < 0) {
    throw ArgumentError.value(
      n,
      'n',
      'Must be non-negative (number of trials n was $n)',
    );
  }
  if (p < 0.0 || p > 1.0 || p.isNaN) {
    throw ArgumentError.value(
      p,
      'p',
      'Must be between 0.0 and 1.0 (success probability p was $p)',
    );
  }
  final resolvedDType = dtype ?? (out?.dtype ?? DType.int64 as DType<T>);
  if (!identical(resolvedDType, DType.int64) &&
      !identical(resolvedDType, DType.int32)) {
    throw ArgumentError.value(
      resolvedDType,
      'dtype',
      'Must be an integer dtype (binomial only supports integer dtypes: int32/int64, got $resolvedDType)',
    );
  }
  _validateOutBuffer(out, shape, resolvedDType);
  final useSecureCsprng = secure;
  final seedVal = useSecureCsprng ? 0 : (seed ?? _nextDefaultSeed());

  if (out != null && !out.isContiguous) {
    return NDArray.scope(() {
      final temp = NDArray<T>.create(shape, resolvedDType);
      final len = temp.size;
      switch (resolvedDType) {
        case DType.int64:
          if (useSecureCsprng) {
            v_secure_binomial_int64(temp.pointer.cast<ffi.Int64>(), len, n, p);
          } else {
            v_binomial_int64(
              temp.pointer.cast<ffi.Int64>(),
              len,
              n,
              p,
              seedVal,
            );
          }
        case DType.int32:
          if (useSecureCsprng) {
            v_secure_binomial_int32(temp.pointer.cast<ffi.Int32>(), len, n, p);
          } else {
            v_binomial_int32(
              temp.pointer.cast<ffi.Int32>(),
              len,
              n,
              p,
              seedVal,
            );
          }
        case DType.float64:
        case DType.float32:
        case DType.float16:
        case DType.bfloat16:
        case DType.int16:
        case DType.int8:
        case DType.uint64:
        case DType.uint32:
        case DType.uint16:
        case DType.uint8:
        case DType.boolean:
        case DType.complex128:
        case DType.complex64:
          throw ArgumentError.value(
            resolvedDType,
            'resolvedDType',
            'Must be an integer dtype (binomial only supports integer dtypes: int32/int64, got $resolvedDType)',
          );
      }
      temp.copy(out: out);
      return out;
    });
  }

  final arr = out ?? NDArray<T>.create(shape, resolvedDType);
  final len = arr.size;

  switch (resolvedDType) {
    case DType.int64:
      if (useSecureCsprng) {
        v_secure_binomial_int64(arr.pointer.cast<ffi.Int64>(), len, n, p);
      } else {
        v_binomial_int64(arr.pointer.cast<ffi.Int64>(), len, n, p, seedVal);
      }
    case DType.int32:
      if (useSecureCsprng) {
        v_secure_binomial_int32(arr.pointer.cast<ffi.Int32>(), len, n, p);
      } else {
        v_binomial_int32(arr.pointer.cast<ffi.Int32>(), len, n, p, seedVal);
      }
    case DType.float64:
    case DType.float32:
    case DType.float16:
    case DType.bfloat16:
    case DType.int16:
    case DType.int8:
    case DType.uint64:
    case DType.uint32:
    case DType.uint16:
    case DType.uint8:
    case DType.boolean:
    case DType.complex128:
    case DType.complex64:
      throw ArgumentError.value(
        resolvedDType,
        'resolvedDType',
        'Must be an integer dtype (binomial only supports integer dtypes: int32/int64, got $resolvedDType)',
      );
  }
  return arr;
}

/// Draws random samples from a multivariate normal (Gaussian) distribution.
///
/// This corresponds to NumPy's `random.multivariate_normal` function.
///
/// **Mathematical Mechanics**:
/// The multivariate normal distribution is defined by a mean vector [mean] ($\mu$) of size $D$
/// and a symmetric, positive-semidefinite covariance matrix [cov] ($\Sigma$) of size $D \times D$.
///
/// To draw a sample $X \sim \mathcal{N}(\mu, \Sigma)$:
/// 1. Computes a factor $L$ of the covariance matrix $\Sigma = L \cdot L^T$ (via Cholesky decomposition
///    when strictly positive-definite, falling back to symmetric eigendecomposition for positive-semidefinite matrices).
/// 2. Draws standard independent normal vectors $Z \sim \mathcal{N}(0, I)$ of size $D$.
/// 3. Returns the linearly transformed sample $X = \mu + Z \cdot L^T$ natively using
///    zero-copy BLAS matrix multiplication (`matmul()`) and broadcasted upcast addition (`add()`)!
///
/// **Preconditions:**
/// - [mean] must be a 1-dimensional vector of size $D$.
/// - [cov] must be a square 2-dimensional symmetric, positive-semidefinite covariance matrix of size $D \times D$.
/// - If provided, [size] must be a valid shape list (e.g. `[N]`).
///
/// - It is an error if [mean] is not 1D or [cov] is not 2D and square.
/// - It is an error if [mean] first dimension does not match [cov] dimensions.
/// - It is an error if [cov] is not symmetric positive-semidefinite.
///
/// **Performance considerations:**
/// - Uses LAPACK Cholesky / eigensolver and CBLAS matrix multiplication.
///
/// **Example:**
/// {@example /example/random_example.dart lang=dart}
NDArray<T> multivariateNormal<T extends SelfOf<DTypeTag>>(
  NDArray<T> mean,
  NDArray<T> cov, {
  List<int>? size,
  DType<T>? dtype,
  int? seed,
  NDArray<T>? out,
  bool secure = false,
}) {
  if (mean.isDisposed || cov.isDisposed) {
    throw StateError(
      'Cannot execute multivariateNormal() on a disposed array.',
    );
  }
  if (out != null && out.isDisposed) {
    throw StateError(
      'Cannot write multivariateNormal result to a disposed output array.',
    );
  }
  if (secure && seed != null) {
    throw ArgumentError.value(seed, 'seed', 'Must be null when secure is true');
  }
  if (mean.dtype != cov.dtype) {
    throw ArgumentError.value(
      cov,
      'cov',
      'Must have the same dtype as mean (${mean.dtype}), got ${cov.dtype}',
    );
  }
  if (mean.shape.length != 1) {
    throw ArgumentError.value(
      mean.shape,
      'mean',
      'Must be a 1-dimensional vector (mean must be a 1-dimensional vector, was ${mean.shape})',
    );
  }
  if (cov.shape.length != 2 || cov.shape[0] != cov.shape[1]) {
    throw ArgumentError.value(
      cov.shape,
      'cov',
      'Must be a 2-dimensional square matrix (cov must be a 2-dimensional square matrix, was ${cov.shape})',
    );
  }
  final d = mean.shape[0];
  if (cov.shape[0] != d) {
    throw ArgumentError.value(
      cov.shape,
      'cov',
      'Must match mean dimension ($d) (mean dimension $d must match cov dimensions ${cov.shape[0]}x${cov.shape[1]})',
    );
  }

  final resolvedDType = dtype ?? (out?.dtype ?? mean.dtype);
  if (!identical(resolvedDType, DType.float32) &&
      !identical(resolvedDType, DType.float64)) {
    throw ArgumentError.value(
      resolvedDType,
      'dtype',
      'Must be a floating point dtype (multivariateNormal only supports floating point dtypes: float32/float64, got $resolvedDType)',
    );
  }

  final sampleShape = <int>[];
  if (size != null) {
    sampleShape.addAll(size);
  }
  final sampleCount = checkTotalSize(sampleShape);
  final finalShape = [...sampleShape, d];
  _validateOutBuffer(out, finalShape, resolvedDType);

  return NDArray.scope(() {
    final meanCast = mean.dtype == resolvedDType
        ? mean
        : castNDArray<T>(mean, resolvedDType);
    final covCast = cov.dtype == resolvedDType
        ? cov
        : castNDArray<T>(cov, resolvedDType);

    for (var i = 0; i < d; i++) {
      final diagVal = (covCast.getCell([i, i]) as num).toDouble();
      if (!diagVal.isFinite || diagVal < -1e-8) {
        throw ArgumentError.value(
          cov,
          'cov',
          'Must be a symmetric positive-semidefinite matrix',
        );
      }
      for (var j = i + 1; j < d; j++) {
        final vij = (covCast.getCell([i, j]) as num).toDouble();
        final vji = (covCast.getCell([j, i]) as num).toDouble();
        if (!vij.isFinite ||
            !vji.isFinite ||
            (vij - vji).abs() > 1e-6 * (1.0 + vij.abs())) {
          throw ArgumentError.value(
            cov,
            'cov',
            'Must be a symmetric positive-semidefinite matrix',
          );
        }
      }
    }

    late final NDArray<T> l;
    if (d == 0) {
      l = NDArray<T>.create([0, 0], resolvedDType);
    } else {
      NDArray<T>? chol;
      try {
        chol = cholesky(covCast);
      } on Object {
        chol = null;
      }
      if (chol != null) {
        l = chol;
      } else {
        final covF64 = covCast.dtype == DType.float64
            ? covCast as NDArray<Float64>
            : castNDArray<Float64>(covCast, DType.float64);
        final eig = eigh(covF64);
        final w = eig.eigenvalues;
        final v = eig.eigenvectors;
        final factorF64 = NDArray<Float64>.create([d, d], DType.float64);
        for (var j = 0; j < d; j++) {
          final wj = w.getCell([j]);
          if (wj.isNaN || wj < -1e-6) {
            throw ArgumentError.value(
              cov,
              'cov',
              'Must be a symmetric positive-semidefinite matrix',
            );
          }
          final scaleJ = wj <= 0.0 ? 0.0 : math.sqrt(wj);
          for (var i = 0; i < d; i++) {
            factorF64.setCell([i, j], v.getCell([i, j]) * scaleJ);
          }
        }
        l = resolvedDType == DType.float64
            ? factorF64 as NDArray<T>
            : castNDArray<T>(factorF64, resolvedDType);
      }
    }

    final zShape = [...sampleShape, d];
    final z = normal(
      zShape,
      dtype: resolvedDType.asAnySpec,
      seed: seed,
      secure: secure,
    );
    final lT = l.transpose();

    final z2D = z.reshape([sampleCount, d]);
    final bool useTempOut =
        out != null &&
        (!out.isContiguous ||
            sharesMemory(mean, out) ||
            sharesMemory(cov, out));
    final target = (out == null || useTempOut)
        ? NDArray<T>.create(finalShape, resolvedDType)
        : out;
    if (sampleCount > 0 && d > 0) {
      final x2D = target.reshape([sampleCount, d]);
      add(
        matmul(z2D.asAnySpec, lT.asAnySpec),
        meanCast.asAnySpec,
        out: x2D.asAnySpec,
      );
    }

    if (out != null) {
      if (useTempOut) {
        target.copy(out: out);
      }
      return out;
    }
    return target.detachToParentScope();
  });
}

/// Draws samples from a multinomial distribution.
///
/// This corresponds to NumPy's `random.multinomial` function.
///
/// **Mathematical Mechanics**:
/// The multinomial distribution is a generalization of the binomial distribution.
/// A trial has $K$ possible categorical outcomes, each with a probability $p_j$ specified in [pvals].
///
/// To draw a sample of shape `[...size, K]`:
/// 1. Computes the cumulative probability distribution (CDF) of [pvals].
/// 2. For each output coordinate block, performs [n] trials:
///    - Draws a standard uniform variable $U \sim \mathcal{U}(0, 1)$.
///    - Uses binary/linear search to find the first outcome index $j$ where $U \le \text{CDF}[j]$.
///    - Increments the count of category $j$ for that sample.
///
/// **Preconditions:**
/// - [n] must be strictly non-negative ($\ge 0$).
/// - [pvals] must be a 1-dimensional vector of probabilities. The probabilities must sum to approximately 1.0.
/// - If provided, [size] must be a valid shape list.
///
/// - It is an error if [n] is negative, or if [pvals] is not a 1D vector.
/// - It is an error if [pvals] contains negative probabilities, or if their sum exceeds 1.0 by a significant tolerance.
///
/// **Example:**
/// {@example /example/random_example.dart lang=dart}
NDArray<T> multinomial<T extends AnySpec, P extends DTypeTag>(
  int n,
  NDArray<P> pvals, {
  List<int>? size,
  DType<T>? dtype,
  int? seed,
  NDArray<T>? out,
  bool secure = false,
}) {
  if (pvals.isDisposed) {
    throw StateError('Cannot execute multinomial() on a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError(
      'Cannot write multinomial result to a disposed output array.',
    );
  }
  if (secure && seed != null) {
    throw ArgumentError.value(seed, 'seed', 'Must be null when secure is true');
  }
  if (n < 0) {
    throw ArgumentError.value(
      n,
      'n',
      'Must be non-negative (n trials must be non-negative, was $n)',
    );
  }
  if (pvals.dtype == DType.boolean || pvals.dtype.isComplex) {
    throw ArgumentError.value(
      pvals.dtype,
      'pvals',
      'Must be a real numeric dtype (got ${pvals.dtype})',
    );
  }
  if (pvals.shape.length != 1) {
    throw ArgumentError.value(
      pvals.shape,
      'pvals',
      'Must be a 1-dimensional probability vector (was ${pvals.shape})',
    );
  }

  final resolvedDType = dtype ?? (out?.dtype ?? DType.int64 as DType<T>);
  if (!identical(resolvedDType, DType.int32) &&
      !identical(resolvedDType, DType.int64)) {
    throw ArgumentError.value(
      resolvedDType,
      'dtype',
      'Must be an integer dtype (multinomial only supports integer dtypes: int32/int64, got $resolvedDType)',
    );
  }

  final k = pvals.shape[0];
  if (k == 0) {
    throw ArgumentError.value(pvals.shape, 'pvals', 'Must not be empty');
  }
  final rand = secure ? Random.secure() : Random(seed ?? _nextDefaultSeed());

  final cdf = List<double>.filled(k, 0.0);
  var sumP = 0.0;
  final isUint64 = (pvals.dtype as DType<DTypeTag>) == DType.uint64;
  for (var i = 0; i < k; i++) {
    final val = pvals.getCellFlat(i);
    final double p;
    if (isUint64 && val is int && val < 0) {
      p = BigInt.from(val).toUnsigned(64).toDouble();
    } else {
      p = (val as num).toDouble();
    }
    if (p.isNaN || p < 0.0 || p > 1.0 + 1e-5) {
      throw ArgumentError.value(
        p,
        'pvals',
        'Must contain probabilities in [0, 1]',
      );
    }
    sumP += p;
    cdf[i] = sumP;
  }

  if (sumP <= 0.0 || sumP.isNaN || sumP.isInfinite) {
    throw ArgumentError.value(
      sumP,
      'pvals',
      'Must have a positive sum of probabilities',
    );
  }

  if ((sumP - 1.0).abs() > 1e-5) {
    throw ArgumentError.value(
      pvals,
      'pvals',
      'Must sum to approximately 1.0 (probabilities do not sum to 1.0, got sum $sumP)',
    );
  }
  for (var i = 0; i < k - 1; i++) {
    cdf[i] /= sumP;
  }
  cdf[k - 1] = 1.0;

  final sampleShape = <int>[];
  if (size != null) {
    sampleShape.addAll(size);
  }
  final sampleCount = checkTotalSize(sampleShape);

  final finalShape = [...sampleShape, k];
  _validateOutBuffer(out, finalShape, resolvedDType);
  final result =
      out ?? NDArray<T>.create(finalShape, resolvedDType, zeroInit: true);
  if (out != null) {
    result.fill(0);
  }

  if (result.isContiguous) {
    if (resolvedDType == DType.int32) {
      final ptr = result.pointer.cast<ffi.Int32>();
      for (var s = 0; s < sampleCount; s++) {
        final offset = s * k;
        for (var t = 0; t < n; t++) {
          final u = rand.nextDouble();
          var outcome = k - 1;
          for (var j = 0; j < k; j++) {
            if (u <= cdf[j]) {
              outcome = j;
              break;
            }
          }
          ptr[offset + outcome]++;
        }
      }
    } else {
      final ptr = result.pointer.cast<ffi.Int64>();
      for (var s = 0; s < sampleCount; s++) {
        final offset = s * k;
        for (var t = 0; t < n; t++) {
          final u = rand.nextDouble();
          var outcome = k - 1;
          for (var j = 0; j < k; j++) {
            if (u <= cdf[j]) {
              outcome = j;
              break;
            }
          }
          ptr[offset + outcome]++;
        }
      }
    }
  } else {
    for (var s = 0; s < sampleCount; s++) {
      for (var t = 0; t < n; t++) {
        final u = rand.nextDouble();
        var outcome = k - 1;
        for (var j = 0; j < k; j++) {
          if (u <= cdf[j]) {
            outcome = j;
            break;
          }
        }
        final coords = <int>[];
        var rem = s;
        for (var d = sampleShape.length - 1; d >= 0; d--) {
          coords.insert(0, rem % sampleShape[d]);
          rem ~/= sampleShape[d];
        }
        coords.add(outcome);
        final currentVal = result.getCell(coords);
        result.setCell(coords, currentVal + 1);
      }
    }
  }

  return result;
}

/// Generates a random sample from a given 1-D array.
///
/// **Preconditions:**
/// - [a] must be a 1-D array and not disposed.
/// - [size] if specified must be a valid shape list.
/// - If [replace] is false, the total sample count must be $\le$ [a.size].
/// - If [p] is specified:
///   - It must be a 1-D array of the same size as [a].
///   - Its values must be non-negative probabilities summing to approximately 1.0.
///
/// - It is an error if [a] or [p] is disposed.
/// - It is an error if [a] is not 1-D.
/// - It is an error if [replace] is false and sample size exceeds [a.size].
/// - It is an error if [p] size is mismatched, negative, or does not sum to 1.0.
///
/// **Example:**
/// {@example /example/random_example.dart lang=dart}
///
/// Reference: [NumPy choice](https://numpy.org/doc/stable/reference/generated/numpy.random.choice.html)
NDArray<T> choice<T extends DTypeTag, Out extends T>(
  NDArray<T> a, {
  List<int>? size,
  bool replace = true,
  NDArray<Float64>? p,
  int? seed,
  bool secure = false,
  NDArray<Out>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute choice on a disposed array.');
  }
  if (secure && seed != null) {
    throw ArgumentError.value(seed, 'seed', 'Must be null when secure is true');
  }
  if (a.shape.length != 1) {
    throw ArgumentError.value(
      a.shape,
      'a',
      'Must be a 1-D array (choice only supports 1-D input arrays, got shape ${a.shape})',
    );
  }
  final sampleShape = size ?? <int>[];
  final sampleCount = checkTotalSize(sampleShape);

  _validateOutBuffer(out, sampleShape, a.dtype);

  double sumP = 0.0;
  if (p != null) {
    if (p.isDisposed) {
      throw StateError('Provided probability array p is disposed.');
    }
    if (p.shape.length != 1 || p.shape[0] != a.shape[0]) {
      throw ArgumentError.value(
        p.shape,
        'p',
        'Must be 1-D and match size of a (${a.shape[0]}) (probability array p was shape ${p.shape})',
      );
    }
    for (var i = 0; i < a.size; i++) {
      final prob = p.getCellFlat(i);
      if (prob.isNaN || prob < 0.0) {
        throw ArgumentError.value(
          prob,
          'p',
          'Must be non-negative and not NaN (pvals must contain non-negative probabilities, was $prob at index $i)',
        );
      }
      sumP += prob;
    }
    if (sumP <= 0.0 || sumP.isNaN || (sumP - 1.0).abs() > 1e-5) {
      throw ArgumentError.value(
        sumP,
        'p',
        'Must sum to approximately 1 (probabilities do not sum to 1, got sum $sumP)',
      );
    }
  }

  if (a.size == 0) {
    if (sampleCount > 0) {
      throw ArgumentError.value(
        sampleCount,
        'size',
        'Cannot choose $sampleCount elements from an empty array',
      );
    }
    return out ?? NDArray<T>.create(sampleShape, a.dtype);
  }

  if (!replace && sampleCount > a.size) {
    throw ArgumentError.value(
      sampleCount,
      'size',
      'Cannot choose $sampleCount elements without replacement from an array of size ${a.size}',
    );
  }

  if (sampleCount == 0) {
    return out ?? NDArray<T>.create(sampleShape, a.dtype);
  }

  final useSecureCsprng = secure;
  final seedVal = useSecureCsprng ? 0 : (seed ?? _nextDefaultSeed());

  return NDArray.scope(() {
    final bool useTempOut =
        out != null &&
        (!out.isContiguous ||
            sharesMemory(a, out) ||
            (p != null && sharesMemory(p, out)));
    final target = (out == null || useTempOut)
        ? NDArray<T>.create(sampleShape, a.dtype)
        : out;
    final result1D = target.reshape([sampleCount]);

    final srcPtr = a.pointer.cast<ffi.Uint8>().cast<ffi.Void>();
    final destPtr = result1D.pointer.cast<ffi.Uint8>().cast<ffi.Void>();
    final srcStride = a.strides.isEmpty ? 1 : a.strides[0];
    final destStride = result1D.strides.isEmpty ? 1 : result1D.strides[0];

    if (p == null) {
      if (replace) {
        if (useSecureCsprng) {
          native_secure_choice_uniform(
            srcPtr,
            srcStride,
            destPtr,
            destStride,
            a.size,
            sampleCount,
            a.dtype.byteWidth,
          );
        } else {
          native_choice_uniform(
            srcPtr,
            srcStride,
            destPtr,
            destStride,
            a.size,
            sampleCount,
            a.dtype.byteWidth,
            seedVal,
          );
        }
      } else {
        if (useSecureCsprng) {
          native_secure_choice_without_replacement(
            srcPtr,
            srcStride,
            destPtr,
            destStride,
            a.size,
            sampleCount,
            a.dtype.byteWidth,
          );
        } else {
          native_choice_without_replacement(
            srcPtr,
            srcStride,
            destPtr,
            destStride,
            a.size,
            sampleCount,
            a.dtype.byteWidth,
            seedVal,
          );
        }
        checkNativeOom();
      }
    } else {
      final marker = ScratchArena.marker;
      try {
        final nonNullP = p;
        if (replace) {
          final cdfPtr = ScratchArena.allocate<ffi.Double>(
            a.size * ffi.sizeOf<ffi.Double>(),
          );
          var runningSum = 0.0;
          for (var i = 0; i < a.size; i++) {
            runningSum += nonNullP.getCellFlat(i);
            cdfPtr[i] = runningSum;
          }
          for (var i = 0; i < a.size; i++) {
            cdfPtr[i] /= sumP;
          }
          cdfPtr[a.size - 1] = 1.0;
          if (useSecureCsprng) {
            native_secure_choice_weighted(
              srcPtr,
              srcStride,
              destPtr,
              destStride,
              cdfPtr,
              a.size,
              sampleCount,
              a.dtype.byteWidth,
            );
          } else {
            native_choice_weighted(
              srcPtr,
              srcStride,
              destPtr,
              destStride,
              cdfPtr,
              a.size,
              sampleCount,
              a.dtype.byteWidth,
              seedVal,
            );
          }
        } else {
          final probsPtr = ScratchArena.allocate<ffi.Double>(
            a.size * ffi.sizeOf<ffi.Double>(),
          );
          for (var i = 0; i < a.size; i++) {
            probsPtr[i] = nonNullP.getCellFlat(i) / sumP;
          }
          if (useSecureCsprng) {
            native_secure_choice_weighted_without_replacement(
              srcPtr,
              srcStride,
              destPtr,
              destStride,
              probsPtr,
              a.size,
              sampleCount,
              a.dtype.byteWidth,
            );
          } else {
            native_choice_weighted_without_replacement(
              srcPtr,
              srcStride,
              destPtr,
              destStride,
              probsPtr,
              a.size,
              sampleCount,
              a.dtype.byteWidth,
              seedVal,
            );
          }
          checkNativeOom();
        }
      } finally {
        ScratchArena.reset(marker);
      }
    }

    if (out != null) {
      if (useTempOut) {
        target.copy(out: out);
      }
      return out;
    }
    return target.detachToParentScope();
  });
}

/// Shuffles the array in-place along the first axis.
///
/// For N-Dimensional arrays, shuffles sub-arrays along axis 0.
/// For 1-Dimensional arrays, shuffles individual elements.
///
/// **Preconditions:**
/// - The array [a] must not be disposed.
///
/// - It is an error if [a] is disposed.
///
/// **Performance considerations:**
/// - Uses native C Fisher-Yates shuffle directly in unmanaged memory with zero Dart allocations.
/// - Performs $O(D_0)$ swaps where $D_0$ is the size of the first dimension (`shape[0]`).
/// - Time complexity is $O(D_0 \cdot S)$ where $S$ is the size of each sub-array slice.
///
/// **Example:**
/// {@example /example/random_example.dart lang=dart}
void shuffle<T extends DTypeTag>(
  NDArray<T> a, {
  int? seed,
  bool secure = false,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot shuffle a disposed array.');
  }
  if (secure && seed != null) {
    throw ArgumentError.value(seed, 'seed', 'Must be null when secure is true');
  }
  validateOutBuffer(a, 'a');

  final d0 = a.shape.isEmpty ? 1 : a.shape[0];
  if (d0 <= 1) return;

  final useSecureCsprng = secure;
  final seedVal = useSecureCsprng ? 0 : (seed ?? _nextDefaultSeed());

  final ptr = a.pointer.cast<ffi.Uint8>().cast<ffi.Void>();

  if (a.shape.length == 1) {
    if (useSecureCsprng) {
      native_secure_shuffle_1d(
        ptr,
        a.shape[0],
        a.strides[0],
        a.dtype.byteWidth,
      );
    } else {
      native_shuffle_1d(
        ptr,
        a.shape[0],
        a.strides[0],
        a.dtype.byteWidth,
        seedVal,
      );
    }
    checkNativeOom();
    return;
  }

  final marker = ScratchArena.marker;
  try {
    final cShape = ScratchArena.copyInt64s(a.shape);
    final cStrides = ScratchArena.copyInt64s(a.strides);
    if (useSecureCsprng) {
      native_secure_shuffle_nd(
        ptr,
        cShape,
        cStrides,
        a.rank,
        a.dtype.byteWidth,
      );
    } else {
      native_shuffle_nd(
        ptr,
        cShape,
        cStrides,
        a.rank,
        a.dtype.byteWidth,
        seedVal,
      );
    }
    checkNativeOom();
  } finally {
    ScratchArena.reset(marker);
  }
}

/// Returns a permuted copy of an array along axis 0.
///
/// **Preconditions:**
/// - The array [a] must not be disposed.
/// - If provided, [out] must not be disposed and must match the shape and dtype of [a].
///
/// - It is an error if [a] is disposed.
/// - It is an error if [out] is disposed or has incompatible shape or dtype.
///
/// **Performance considerations:**
/// - Returns a brand new contiguous deep copy of the array permuted along axis 0.
/// - Time complexity matches [shuffle] ($O(N)$).
///
/// **Example:**
/// {@example /example/random_example.dart lang=dart}
NDArray<T> permutation<T extends DTypeTag, Out extends T>(
  NDArray<T> a, {
  int? seed,
  bool secure = false,
  NDArray<Out>? out,
}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot permute a disposed array.');
  }
  if (secure && seed != null) {
    throw ArgumentError.value(seed, 'seed', 'Must be null when secure is true');
  }
  _validateOutBuffer(out, a.shape, a.dtype);
  if (out != null && (!out.isContiguous || sharesMemory(a, out))) {
    return NDArray.scope(() {
      final temp = a.copy();
      shuffle(temp, seed: seed, secure: secure);
      return temp.copy(out: out);
    });
  }
  final copyArr = a.copy(out: out);
  shuffle(copyArr, seed: seed, secure: secure);
  return copyArr;
}
