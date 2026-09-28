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

// ignore_for_file: non_constant_identifier_names
import 'dart:math' as math;
import '../dtype.dart';
import '../gpu_array.dart';
import '../device.dart';
import '../backend/compute_engine.dart';

/// Counter-based 4x32-10 Philox generator.
final class Philox4x32Engine {
  static const int _philoxM4x32A = 0xD2511F53;
  static const int _philoxM4x32B = 0xCD9E8D57;
  static const int _philoxW32A = 0x9E3779B9;
  static const int _philoxW32B = 0xBB67AE85;

  /// Multiplies two 32-bit unsigned integers into 64-bit (hi, lo) using 16-bit split arithmetic.
  ///
  /// Prevents signed 64-bit integer overflow and arithmetic sign extension.
  static (int, int) _mul32(int a, int b) {
    final aLo = a & 0xFFFF;
    final aHi = (a >>> 16) & 0xFFFF;
    final bLo = b & 0xFFFF;
    final bHi = (b >>> 16) & 0xFFFF;

    final p00 = aLo * bLo;
    final p01 = aLo * bHi;
    final p10 = aHi * bLo;
    final p11 = aHi * bHi;

    final mid1 = p01 + (p00 >>> 16);
    final mid2 = p10 + (mid1 & 0xFFFF);
    final lo = ((mid2 & 0xFFFF) << 16) | (p00 & 0xFFFF);
    final hi = p11 + (mid1 >>> 16) + (mid2 >>> 16);
    return (hi & 0xFFFFFFFF, lo & 0xFFFFFFFF);
  }

  static List<int> philox4x32TenRounds(List<int> ctr, List<int> key) {
    var c0 = ctr[0] & 0xFFFFFFFF;
    var c1 = ctr[1] & 0xFFFFFFFF;
    var c2 = ctr[2] & 0xFFFFFFFF;
    var c3 = ctr[3] & 0xFFFFFFFF;

    var k0 = key[0] & 0xFFFFFFFF;
    var k1 = key[1] & 0xFFFFFFFF;

    for (var r = 0; r < 10; r++) {
      // 32-bit high and low product via 16-bit split multiplication
      final (hi0, lo0) = _mul32(c0, _philoxM4x32A);
      final (hi2, lo2) = _mul32(c2, _philoxM4x32B);

      final nextC0 = (hi2 ^ c1 ^ k0) & 0xFFFFFFFF;
      final nextC1 = lo2;
      final nextC2 = (hi0 ^ c3 ^ k1) & 0xFFFFFFFF;
      final nextC3 = lo0;

      c0 = nextC0;
      c1 = nextC1;
      c2 = nextC2;
      c3 = nextC3;

      k0 = (k0 + _philoxW32A) & 0xFFFFFFFF;
      k1 = (k1 + _philoxW32B) & 0xFFFFFFFF;
    }

    return [c0, c1, c2, c3];
  }
}

/// Encapsulates on-device random generation state.
final class RandomState {
  int _seed;
  int _counter = 0;
  List<int> _buffer = const [];
  int _bufIndex = 0;

  RandomState([int? seed])
    : _seed = seed ?? DateTime.now().microsecondsSinceEpoch;

  /// Seeds this random state.
  void seed(int s) {
    _seed = s;
    _counter = 0;
    _buffer = const [];
    _bufIndex = 0;
  }

  /// Returns the next unsigned 32-bit integer from Philox PRNG.
  int _nextUint32() {
    if (_bufIndex >= _buffer.length) {
      final key = [_seed & 0xFFFFFFFF, (_seed >>> 32) & 0xFFFFFFFF];
      final ctr = [_counter & 0xFFFFFFFF, (_counter >>> 32) & 0xFFFFFFFF, 0, 0];
      _counter++;
      _buffer = Philox4x32Engine.philox4x32TenRounds(ctr, key);
      _bufIndex = 0;
    }
    return _buffer[_bufIndex++];
  }

  /// Returns the next uniform float in $[0.0, 1.0)$ from Philox PRNG.
  double _nextDouble() {
    return (_nextUint32() & 0xFFFFFFFF) / 4294967296.0;
  }

  /// Returns an unbiased random integer in $[0, span)$ from Philox PRNG.
  int _nextInt64(int span) {
    if (span <= 0) throw ArgumentError('span must be positive');
    if (span <= 0xFFFFFFFF) {
      final threshold = (0x100000000 - span) % span;
      while (true) {
        final r = _nextUint32() & 0xFFFFFFFF;
        if (r >= threshold) {
          return r % span;
        }
      }
    } else {
      while (true) {
        final hi = _nextUint32() & 0x7FFFFFFF;
        final lo = _nextUint32() & 0xFFFFFFFF;
        final val63 = (hi << 32) | lo;
        return val63 % span;
      }
    }
  }

  /// Generates uniform random floats in $[0.0, 1.0)$.
  GpuArray<Float64> rand([List<int> shape = const [], GpuDevice? device]) {
    final size = ShapeUtils.computeSize(shape);
    final data = List<double>.filled(size, 0.0);

    for (var i = 0; i < size; i++) {
      data[i] = _nextDouble();
    }

    return GpuArray.fromList(data, shape, DType.float64, device: device);
  }

  /// Generates standard normal random floats ($\\mathcal{N}(0, 1)$) using Box-Muller transform.
  GpuArray<Float64> randn([List<int> shape = const [], GpuDevice? device]) {
    final size = ShapeUtils.computeSize(shape);
    final data = List<double>.filled(size, 0.0);

    for (var i = 0; i < size; i += 2) {
      var u1 = _nextDouble();
      while (u1 <= 1e-15) {
        u1 = _nextDouble();
      }
      final u2 = _nextDouble();

      final mag = math.sqrt(-2.0 * math.log(u1));
      final z0 = mag * math.cos(2.0 * math.pi * u2);
      final z1 = mag * math.sin(2.0 * math.pi * u2);

      data[i] = z0;
      if (i + 1 < size) {
        data[i + 1] = z1;
      }
    }

    return GpuArray.fromList(data, shape, DType.float64, device: device);
  }

  /// Generates random integers from [low] (inclusive) to [high] (exclusive).
  GpuArray<T> randint<T extends DTypeTag>(
    int low, [
    int? high,
    List<int> shape = const [],
    DType<T>? dtype,
    GpuDevice? device,
  ]) {
    final effectiveLow = (high == null) ? 0 : low;
    final effectiveHigh = (high == null) ? low : high;

    if (effectiveHigh <= effectiveLow) {
      throw ArgumentError(
        'randint high ($effectiveHigh) must be > low ($effectiveLow).',
      );
    }

    final span = effectiveHigh - effectiveLow;
    final size = ShapeUtils.computeSize(shape);
    final targetDType = dtype ?? (DType.int64 as DType<T>);
    final result = GpuArray<T>.empty(shape, targetDType, device: device);

    for (var i = 0; i < size; i++) {
      final val = effectiveLow + _nextInt64(span);
      ComputeEngine.writeAny(result.buffer, targetDType, i, val);
    }

    return result;
  }

  /// Uniform distribution over $[low, high)$.
  GpuArray<Float64> uniform({
    double low = 0.0,
    double high = 1.0,
    List<int> shape = const [],
    GpuDevice? device,
  }) {
    final base = rand(shape, device);
    final span = high - low;
    final data = base
        .toList()
        .cast<double>()
        .map((v) => low + v * span)
        .toList();
    return GpuArray.fromList(data, shape, DType.float64, device: device);
  }

  /// Normal (Gaussian) distribution $\\mathcal{N}(\\mu, \\sigma^2)$.
  GpuArray<Float64> normal({
    double loc = 0.0,
    double scale = 1.0,
    List<int> shape = const [],
    GpuDevice? device,
  }) {
    final base = randn(shape, device);
    final data = base
        .toList()
        .cast<double>()
        .map((v) => loc + v * scale)
        .toList();
    return GpuArray.fromList(data, shape, DType.float64, device: device);
  }

  /// Standard normal distribution $\\mathcal{N}(0, 1)$.
  GpuArray<Float64> standard_normal([
    List<int> shape = const [],
    GpuDevice? device,
  ]) => randn(shape, device);

  /// Exponential distribution.
  GpuArray<Float64> exponential({
    double scale = 1.0,
    List<int> shape = const [],
    GpuDevice? device,
  }) {
    final size = ShapeUtils.computeSize(shape);
    final data = List<double>.filled(size, 0.0);
    for (var i = 0; i < size; i++) {
      var u = _nextDouble();
      while (u <= 1e-15) {
        u = _nextDouble();
      }
      data[i] = -scale * math.log(u);
    }
    return GpuArray.fromList(data, shape, DType.float64, device: device);
  }

  /// Randomly samples elements from [a].
  GpuArray<T> choice<T extends DTypeTag>(
    GpuArray<T> a, {
    int? size,
    bool replace = true,
    List<double>? p,
    GpuDevice? device,
  }) {
    final aList = a.toList();
    final n = aList.length;
    final count = size ?? 1;

    if (n == 0 && count > 0) {
      throw ArgumentError('Cannot sample from an empty array.');
    }

    if (!replace && count > n) {
      throw ArgumentError(
        'Cannot take a larger sample ($count) than population ($n) when replace=false.',
      );
    }

    if (p != null) {
      if (p.length != n) {
        throw ArgumentError(
          'p and a must have the same length: p has ${p.length}, a has $n.',
        );
      }
      var sumP = 0.0;
      var positiveCount = 0;
      for (var i = 0; i < n; i++) {
        if (p[i] < 0.0 || p[i].isNaN) {
          throw ArgumentError('Probabilities in p must be non-negative.');
        }
        sumP += p[i];
        if (p[i] > 0.0) {
          positiveCount++;
        }
      }
      if (sumP <= 0.0) {
        throw ArgumentError('Probabilities in p must sum to > 0.');
      }
      if (!replace && positiveCount < count) {
        throw ArgumentError(
          'Cannot take $count samples without replacement when only $positiveCount items have positive probability.',
        );
      }
    }

    final out = <dynamic>[];

    if (count > 0) {
      if (replace) {
        if (p == null) {
          // Uniform with replacement
          for (var i = 0; i < count; i++) {
            final idx = _nextInt64(n);
            out.add(aList[idx]);
          }
        } else {
          // Weighted with replacement
          final cdf = List<double>.filled(n, 0.0);
          var cum = 0.0;
          final sumP = p.reduce((a, b) => a + b);
          for (var i = 0; i < n; i++) {
            cum += p[i] / sumP;
            cdf[i] = cum;
          }
          cdf[n - 1] = 1.0;

          for (var i = 0; i < count; i++) {
            final u = _nextDouble();
            var low = 0;
            var high = n - 1;
            while (low < high) {
              final mid = (low + high) >> 1;
              if (u < cdf[mid]) {
                high = mid;
              } else {
                low = mid + 1;
              }
            }
            out.add(aList[low]);
          }
        }
      } else {
        // Without replacement
        if (p == null) {
          // Uniform without replacement via partial Fisher-Yates shuffle
          final indices = List<int>.generate(n, (i) => i);
          for (var i = 0; i < count; i++) {
            final j = i + _nextInt64(n - i);
            final tmp = indices[i];
            indices[i] = indices[j];
            indices[j] = tmp;
            out.add(aList[indices[i]]);
          }
        } else {
          // Weighted without replacement (Efraimidis-Spirakis algorithm)
          // Key = -ln(U_i) / p_i
          final indexedKeys = List<({int index, double key})>.generate(n, (i) {
            final prob = p[i];
            if (prob <= 0.0) {
              return (index: i, key: double.infinity);
            }
            var u = _nextDouble();
            while (u <= 1e-15) {
              u = _nextDouble();
            }
            final key = -math.log(u) / prob;
            return (index: i, key: key);
          });

          indexedKeys.sort((a, b) => a.key.compareTo(b.key));
          for (var i = 0; i < count; i++) {
            out.add(aList[indexedKeys[i].index]);
          }
        }
      }
    }

    final shape = size != null ? [size] : <int>[];
    return GpuArray.fromList(out, shape, a.dtype, device: device ?? a.device);
  }

  /// Randomly permutes a sequence or returns a permuted range.
  dynamic permutation(dynamic x, [GpuDevice? device]) {
    if (x is int) {
      final list = List.generate(x, (i) => i);
      _shuffleList(list);
      return GpuArray.fromList(list, [x], DType.int64, device: device);
    }
    if (x is GpuArray) {
      final list = x.toList();
      _shuffleList(list);
      return GpuArray.fromList(
        list,
        x.shape,
        x.dtype,
        device: device ?? x.device,
      );
    }
    throw ArgumentError('permutation requires an integer or a GpuArray.');
  }

  /// In-place shuffle of the contents of [x].
  void shuffle(GpuArray x) {
    final list = x.toList();
    _shuffleList(list);
    final total = list.length;
    final coords = List<int>.filled(x.rank, 0);
    for (var i = 0; i < total; i++) {
      var dstIdx = x.offsetElements;
      for (var d = 0; d < x.rank; d++) {
        dstIdx += coords[d] * x.strides[d];
      }
      ComputeEngine.writeAny(x.buffer, x.dtype, dstIdx, list[i]);
      for (var d = x.rank - 1; d >= 0; d--) {
        coords[d]++;
        if (coords[d] < x.shape[d]) break;
        coords[d] = 0;
      }
    }
  }

  void _shuffleList<E>(List<E> list) {
    for (var i = list.length - 1; i > 0; i--) {
      final j = _nextInt64(i + 1);
      final tmp = list[i];
      list[i] = list[j];
      list[j] = tmp;
    }
  }
}

/// Default global random state.
final RandomState defaultRandomState = RandomState();

/// Seeds the default random generator.
void seed(int s) => defaultRandomState.seed(s);

/// Uniform random numbers in $[0.0, 1.0)$.
GpuArray<Float64> rand([List<int> shape = const [], GpuDevice? device]) =>
    defaultRandomState.rand(shape, device);

/// Standard normal random numbers ($\\mathcal{N}(0, 1)$).
GpuArray<Float64> randn([List<int> shape = const [], GpuDevice? device]) =>
    defaultRandomState.randn(shape, device);

/// Random integers from [low] to [high].
GpuArray<T> randint<T extends DTypeTag>(
  int low, [
  int? high,
  List<int> shape = const [],
  DType<T>? dtype,
  GpuDevice? device,
]) => defaultRandomState.randint<T>(low, high, shape, dtype, device);

/// Uniform distribution over $[low, high)$.
GpuArray<Float64> uniform({
  double low = 0.0,
  double high = 1.0,
  List<int> shape = const [],
  GpuDevice? device,
}) => defaultRandomState.uniform(
  low: low,
  high: high,
  shape: shape,
  device: device,
);

/// Normal (Gaussian) distribution.
GpuArray<Float64> normal({
  double loc = 0.0,
  double scale = 1.0,
  List<int> shape = const [],
  GpuDevice? device,
}) => defaultRandomState.normal(
  loc: loc,
  scale: scale,
  shape: shape,
  device: device,
);

/// Standard normal distribution.
GpuArray<Float64> standard_normal([
  List<int> shape = const [],
  GpuDevice? device,
]) => defaultRandomState.standard_normal(shape, device);

/// Exponential distribution.
GpuArray<Float64> exponential({
  double scale = 1.0,
  List<int> shape = const [],
  GpuDevice? device,
}) =>
    defaultRandomState.exponential(scale: scale, shape: shape, device: device);

/// Random choice from [a].
GpuArray<T> choice<T extends DTypeTag>(
  GpuArray<T> a, {
  int? size,
  bool replace = true,
  List<double>? p,
  GpuDevice? device,
}) => defaultRandomState.choice(
  a,
  size: size,
  replace: replace,
  p: p,
  device: device,
);

/// Random permutation.
dynamic permutation(dynamic x, [GpuDevice? device]) =>
    defaultRandomState.permutation(x, device);

/// In-place shuffle.
void shuffle(GpuArray x) => defaultRandomState.shuffle(x);
