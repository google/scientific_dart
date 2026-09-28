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

import 'dart:ffi' as ffi;
import 'dart:math' as math;

import 'package:ndarray/ndarray.dart'
    show DType, DTypeTag, Float64, Int64, NDArray;

import '../device.dart';
import '../gpu_array.dart';

/// Counter-based Philox 4x32-10 pseudo-random number generator engine.
///
/// Generates blocks of four 32-bit unsigned integers per counter step using
/// 10 rounds of the Salmon et al. (2011) Philox bijection.
final class Philox4x32Engine {
  static const int _philoxM0 = 0xD2511F53;
  static const int _philoxM1 = 0xCD9E8D57;
  static const int _philoxW0 = 0x9E3779B9;
  static const int _philoxW1 = 0xBB67AE85;

  List<int> _key;
  int _counter;

  /// Creates a [Philox4x32Engine] initialized with [seed] and starting
  /// [counter].
  Philox4x32Engine({int seed = 0, int counter = 0})
    : _key = <int>[seed & 0xFFFFFFFF, (seed >>> 32) & 0xFFFFFFFF],
      _counter = counter;

  /// The current 64-bit counter offset of this engine.
  int get counter => _counter;

  /// Resets the engine state to [seed] and [counter].
  void reset({required int seed, int counter = 0}) {
    _key = <int>[seed & 0xFFFFFFFF, (seed >>> 32) & 0xFFFFFFFF];
    _counter = counter;
  }

  /// Advances the internal counter by 4 and returns the next block of four
  /// 32-bit unsigned random integers.
  List<int> nextBlock() {
    final c0 = _counter & 0xFFFFFFFF;
    final c1 = (_counter >>> 32) & 0xFFFFFFFF;
    _counter += 4;
    return philox4x32TenRounds(<int>[
      c0,
      c1,
      (c0 + 1) & 0xFFFFFFFF,
      (c1 + 1) & 0xFFFFFFFF,
    ], _key);
  }

  /// Multiplies two 32-bit unsigned integers [a] and [b], returning the low
  /// and high 32-bit words `(lo, hi)`.
  static (int, int) _mulhilo32(int a, int b) {
    final aBig = BigInt.from(a & 0xFFFFFFFF);
    final bBig = BigInt.from(b & 0xFFFFFFFF);
    final product = aBig * bBig;
    final lo = (product & BigInt.from(0xFFFFFFFF)).toInt();
    final hi = ((product >> 32) & BigInt.from(0xFFFFFFFF)).toInt();
    return (lo, hi);
  }

  /// Executes 10 rounds of Philox-4x32 on a 4-word [counter] and 2-word [key].
  ///
  /// The [counter] list must contain at least 4 elements and [key] must contain
  /// at least 2 elements.
  static List<int> philox4x32TenRounds(List<int> counter, List<int> key) {
    if (counter.length < 4) {
      throw ArgumentError.value(
        counter,
        'counter',
        'Must be a list of at least 4 32-bit words.',
      );
    }
    if (key.length < 2) {
      throw ArgumentError.value(
        key,
        'key',
        'Must be a list of at least 2 32-bit words.',
      );
    }
    var c0 = counter[0] & 0xFFFFFFFF;
    var c1 = counter[1] & 0xFFFFFFFF;
    var c2 = counter[2] & 0xFFFFFFFF;
    var c3 = counter[3] & 0xFFFFFFFF;
    var k0 = key[0] & 0xFFFFFFFF;
    var k1 = key[1] & 0xFFFFFFFF;

    for (var round = 0; round < 10; round++) {
      final (lo0, hi0) = _mulhilo32(_philoxM0, c0);
      final (lo1, hi1) = _mulhilo32(_philoxM1, c2);

      c0 = hi1 ^ c1 ^ k0;
      c1 = lo1;
      c2 = hi0 ^ c3 ^ k1;
      c3 = lo0;

      k0 = (k0 + _philoxW0) & 0xFFFFFFFF;
      k1 = (k1 + _philoxW1) & 0xFFFFFFFF;
    }

    return <int>[c0, c1, c2, c3];
  }
}

bool _shapesEqual(List<int> first, List<int> second) {
  if (first.length != second.length) return false;
  for (var i = 0; i < first.length; i++) {
    if (first[i] != second[i]) return false;
  }
  return true;
}

int _validateShapeAndCountElements(List<int> shape) {
  var count = 1;
  for (var i = 0; i < shape.length; i++) {
    if (shape[i] < 0) {
      throw ArgumentError.value(
        shape,
        'shape',
        'Must not contain negative dimensions (got ${shape[i]} at axis $i).',
      );
    }
    count *= shape[i];
  }
  return count;
}

GpuArray<R> _writeOrWrapResult<R extends DTypeTag>(
  NDArray<R> hostResult,
  GpuDevice device,
  GpuArray<R>? out,
) {
  if (out != null) {
    if (out.size > 1 && out.strides.contains(0)) {
      throw ArgumentError.value(
        out,
        'out',
        'Must be writeable and not a broadcasted view.',
      );
    }
    if (!_shapesEqual(out.shape, hostResult.shape) ||
        out.dtype != hostResult.dtype) {
      throw ArgumentError.value(
        out,
        'out',
        'Must be an array with shape ${hostResult.shape} and dtype '
            '${hostResult.dtype}, got shape ${out.shape} and dtype ${out.dtype}.',
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
    } else if (out.byteSize > 0) {
      out.buffer.copyToHost(
        out.buffer.address.cast<ffi.Void>(),
        out.buffer.sizeInBytes,
      );
      final totalBufferElements = out.buffer.sizeInBytes ~/ out.dtype.byteWidth;
      final rootBufferView = NDArray<R>.fromPointer(
        out.buffer.address.cast<ffi.Void>(),
        <int>[totalBufferElements],
        out.dtype,
      );
      final outView = NDArray<R>.view(
        rootBufferView,
        shape: out.shape,
        strides: out.strides,
        offsetElements: out.offsetElements,
      );
      hostResult.copy(out: outView);
      out.buffer.copyFromHost(
        out.buffer.address.cast<ffi.Void>(),
        out.buffer.sizeInBytes,
      );
    }
    return out;
  }
  final gpuResult = GpuArray<R>.fromNDArray(hostResult, device: device);
  gpuResult.detachToParentScope();
  return gpuResult;
}

/// Stateful pseudo-random number generator backed by [Philox4x32Engine].
final class RandomState {
  int _seed;
  final Philox4x32Engine _engine;
  List<int> _wordBuffer = const <int>[];
  int _wordBufferIndex = 4;
  double? _spareNormal;

  /// Creates a [RandomState] seeded with [seed] (or an entropy-derived seed if
  /// omitted).
  RandomState([int? seed])
    : _seed = seed ?? DateTime.now().microsecondsSinceEpoch,
      _engine = Philox4x32Engine(
        seed: seed ?? DateTime.now().microsecondsSinceEpoch,
      );

  /// The current seed of this generator.
  int get currentSeed => _seed;

  /// Resets the generator with [newSeed] and zeroes the internal counter.
  void seed(int newSeed) {
    _seed = newSeed;
    _engine.reset(seed: newSeed);
    _wordBuffer = const <int>[];
    _wordBufferIndex = 4;
    _spareNormal = null;
  }

  int _nextUint32() {
    if (_wordBufferIndex >= 4) {
      _wordBuffer = _engine.nextBlock();
      _wordBufferIndex = 0;
    }
    return _wordBuffer[_wordBufferIndex++];
  }

  /// Generates a uniform random `double` in `[0.0, 1.0)` with 53-bit precision.
  double _nextDouble() {
    final highBits = _nextUint32() >>> 5; // 27 bits
    final lowBits = _nextUint32() >>> 6; // 26 bits
    return (highBits * 67108864.0 + lowBits) * (1.0 / 9007199254740992.0);
  }

  /// Generates a standard normal variate $\mathcal{N}(0, 1)$ via Box-Muller.
  double _nextGaussian() {
    if (_spareNormal case final cached?) {
      _spareNormal = null;
      return cached;
    }
    double u1;
    do {
      u1 = _nextDouble();
    } while (u1 <= 1e-300);
    final u2 = _nextDouble();
    final radius = math.sqrt(-2.0 * math.log(u1));
    final theta = 2.0 * math.pi * u2;
    _spareNormal = radius * math.sin(theta);
    return radius * math.cos(theta);
  }

  /// Generates a uniform integer in `[0, maxExclusive)` using unbiased
  /// rejection sampling.
  int _nextInt(int maxExclusive) {
    if (maxExclusive <= 0) {
      throw ArgumentError.value(
        maxExclusive,
        'maxExclusive',
        'Must be positive.',
      );
    }
    if (maxExclusive == 1) return 0;
    final limit = 0x100000000 - (0x100000000 % maxExclusive);
    int sample;
    do {
      sample = _nextUint32();
    } while (sample >= limit);
    return sample % maxExclusive;
  }

  /// Samples uniform values in `[0.0, 1.0)` with the given [shape].
  ///
  /// All dimensions in [shape] must be non-negative. If [out] is provided, it
  /// must not be disposed and must have shape [shape] and dtype
  /// [DType.float64].
  GpuArray<Float64> rand([
    List<int> shape = const <int>[],
    GpuDevice? device,
    GpuArray<Float64>? out,
  ]) {
    if (out != null && out.isDisposed) {
      throw StateError(
        'Cannot write rand result to a disposed output GpuArray.',
      );
    }
    final count = _validateShapeAndCountElements(shape);
    final targetDevice = out?.device ?? device ?? GpuDevice.defaultDevice;
    return NDArray.scope(() {
      final hostResult = NDArray<Float64>.zeros(shape, DType.float64);
      final pointer = hostResult.pointer.cast<ffi.Double>();
      for (var i = 0; i < count; i++) {
        pointer[i] = _nextDouble();
      }
      return _writeOrWrapResult(hostResult, targetDevice, out);
    });
  }

  /// Samples standard normal $\mathcal{N}(0, 1)$ values with the given [shape].
  ///
  /// All dimensions in [shape] must be non-negative. If [out] is provided, it
  /// must not be disposed and must have shape [shape] and dtype
  /// [DType.float64].
  GpuArray<Float64> randn([
    List<int> shape = const <int>[],
    GpuDevice? device,
    GpuArray<Float64>? out,
  ]) {
    if (out != null && out.isDisposed) {
      throw StateError(
        'Cannot write randn result to a disposed output GpuArray.',
      );
    }
    final count = _validateShapeAndCountElements(shape);
    final targetDevice = out?.device ?? device ?? GpuDevice.defaultDevice;
    return NDArray.scope(() {
      final hostResult = NDArray<Float64>.zeros(shape, DType.float64);
      final pointer = hostResult.pointer.cast<ffi.Double>();
      for (var i = 0; i < count; i++) {
        pointer[i] = _nextGaussian();
      }
      return _writeOrWrapResult(hostResult, targetDevice, out);
    });
  }

  /// Samples random integers in `[low, high)` (or `[0, low)` when [high] is
  /// omitted) with the given [shape].
  ///
  /// The resolved lower bound must be strictly less than the upper bound, and
  /// all dimensions in [shape] must be non-negative. If [out] is provided, it
  /// must not be disposed and must have shape [shape] and dtype [DType.int64].
  GpuArray<Int64> randint(
    int low, [
    int? high,
    List<int> shape = const <int>[],
    GpuDevice? device,
    GpuArray<Int64>? out,
  ]) {
    if (out != null && out.isDisposed) {
      throw StateError(
        'Cannot write randint result to a disposed output GpuArray.',
      );
    }
    final int minBound;
    final int maxBound;
    if (high == null) {
      minBound = 0;
      maxBound = low;
    } else {
      minBound = low;
      maxBound = high;
    }
    final span = maxBound - minBound;
    if (span <= 0) {
      throw ArgumentError.value(
        high ?? low,
        high == null ? 'low' : 'high',
        'Must be greater than lower bound ($minBound).',
      );
    }
    final count = _validateShapeAndCountElements(shape);
    final targetDevice = out?.device ?? device ?? GpuDevice.defaultDevice;
    return NDArray.scope(() {
      final hostResult = NDArray<Int64>.zeros(shape, DType.int64);
      final pointer = hostResult.pointer.cast<ffi.Int64>();
      for (var i = 0; i < count; i++) {
        pointer[i] = minBound + _nextInt(span);
      }
      return _writeOrWrapResult(hostResult, targetDevice, out);
    });
  }

  /// Samples from a continuous uniform distribution over `[low, high)`.
  ///
  /// The lower bound [low] must not exceed [high], and all dimensions in
  /// [shape] must be non-negative. If [out] is provided, it must not be
  /// disposed and must have shape [shape] and dtype [DType.float64].
  GpuArray<Float64> uniform({
    double low = 0.0,
    double high = 1.0,
    List<int> shape = const <int>[],
    GpuDevice? device,
    GpuArray<Float64>? out,
  }) {
    if (out != null && out.isDisposed) {
      throw StateError(
        'Cannot write uniform result to a disposed output GpuArray.',
      );
    }
    if (low > high) {
      throw ArgumentError.value(
        low,
        'low',
        'Must be less than or equal to high ($high).',
      );
    }
    final count = _validateShapeAndCountElements(shape);
    final range = high - low;
    final targetDevice = out?.device ?? device ?? GpuDevice.defaultDevice;
    return NDArray.scope(() {
      final hostResult = NDArray<Float64>.zeros(shape, DType.float64);
      final pointer = hostResult.pointer.cast<ffi.Double>();
      for (var i = 0; i < count; i++) {
        pointer[i] = low + _nextDouble() * range;
      }
      return _writeOrWrapResult(hostResult, targetDevice, out);
    });
  }

  /// Samples from a Gaussian (normal) distribution $\mathcal{N}(\text{loc}, \text{scale}^2)$.
  ///
  /// The standard deviation [scale] must be positive, and all dimensions in
  /// [shape] must be non-negative. If [out] is provided, it must not be
  /// disposed and must have shape [shape] and dtype [DType.float64].
  GpuArray<Float64> normal({
    double loc = 0.0,
    double scale = 1.0,
    List<int> shape = const <int>[],
    GpuDevice? device,
    GpuArray<Float64>? out,
  }) {
    if (out != null && out.isDisposed) {
      throw StateError(
        'Cannot write normal result to a disposed output GpuArray.',
      );
    }
    if (scale <= 0.0) {
      throw ArgumentError.value(scale, 'scale', 'Must be positive.');
    }
    final count = _validateShapeAndCountElements(shape);
    final targetDevice = out?.device ?? device ?? GpuDevice.defaultDevice;
    return NDArray.scope(() {
      final hostResult = NDArray<Float64>.zeros(shape, DType.float64);
      final pointer = hostResult.pointer.cast<ffi.Double>();
      for (var i = 0; i < count; i++) {
        pointer[i] = loc + _nextGaussian() * scale;
      }
      return _writeOrWrapResult(hostResult, targetDevice, out);
    });
  }

  /// Samples from the standard normal distribution $\mathcal{N}(0, 1)$.
  ///
  /// All dimensions in [shape] must be non-negative. If [out] is provided, it
  /// must not be disposed and must have shape [shape] and dtype
  /// [DType.float64].
  GpuArray<Float64> standardNormal({
    List<int> shape = const <int>[],
    GpuDevice? device,
    GpuArray<Float64>? out,
  }) => randn(shape, device, out);

  /// Snake-case alias for [standardNormal].
  GpuArray<Float64> standard_normal({
    List<int> shape = const <int>[],
    GpuDevice? device,
    GpuArray<Float64>? out,
  }) => randn(shape, device, out);

  /// Samples from an exponential distribution with rate $\lambda = 1 / \text{scale}$.
  ///
  /// The scale parameter [scale] must be positive, and all dimensions in
  /// [shape] must be non-negative. If [out] is provided, it must not be
  /// disposed and must have shape [shape] and dtype [DType.float64].
  GpuArray<Float64> exponential({
    double scale = 1.0,
    List<int> shape = const <int>[],
    GpuDevice? device,
    GpuArray<Float64>? out,
  }) {
    if (out != null && out.isDisposed) {
      throw StateError(
        'Cannot write exponential result to a disposed output GpuArray.',
      );
    }
    if (scale <= 0.0) {
      throw ArgumentError.value(scale, 'scale', 'Must be positive.');
    }
    final count = _validateShapeAndCountElements(shape);
    final targetDevice = out?.device ?? device ?? GpuDevice.defaultDevice;
    return NDArray.scope(() {
      final hostResult = NDArray<Float64>.zeros(shape, DType.float64);
      final pointer = hostResult.pointer.cast<ffi.Double>();
      for (var i = 0; i < count; i++) {
        double u;
        do {
          u = _nextDouble();
        } while (u <= 1e-300);
        pointer[i] = -scale * math.log(u);
      }
      return _writeOrWrapResult(hostResult, targetDevice, out);
    });
  }

  /// Generates a random sample from a non-empty 1D [GpuArray] [a].
  ///
  /// Supports sampling with or without replacement ([replace]) and optional
  /// non-negative probability weights [p].
  /// Both [a] and [out] (if provided) must not be disposed.
  GpuArray<T> choice<T extends DTypeTag>(
    GpuArray<T> a, {
    List<int> shape = const <int>[1],
    bool replace = true,
    List<double>? p,
    GpuDevice? device,
    GpuArray<T>? out,
  }) {
    if (a.isDisposed) {
      throw StateError('Cannot sample choice from a disposed GpuArray.');
    }
    if (out != null && out.isDisposed) {
      throw StateError(
        'Cannot write choice result to a disposed output GpuArray.',
      );
    }
    if (a.rank != 1 || a.size == 0) {
      throw ArgumentError.value(
        a.shape,
        'a',
        'Must be a non-empty 1-dimensional GpuArray.',
      );
    }
    final populationSize = a.size;
    final sampleCount = _validateShapeAndCountElements(shape);
    if (!replace && sampleCount > populationSize) {
      throw ArgumentError.value(
        shape,
        'shape',
        'Must not request more samples ($sampleCount) than population size '
            '($populationSize) when replace is false.',
      );
    }

    List<double>? cumulative;
    if (p != null) {
      if (p.length != populationSize) {
        throw ArgumentError.value(
          p,
          'p',
          'Must be of length equal to a.size ($populationSize), '
              'got ${p.length}.',
        );
      }
      var sum = 0.0;
      for (final weight in p) {
        if (weight < 0.0 || weight.isNaN) {
          throw ArgumentError.value(
            p,
            'p',
            'Must not contain negative or NaN probabilities.',
          );
        }
        sum += weight;
      }
      if ((sum - 1.0).abs() > 1e-5) {
        throw ArgumentError.value(
          p,
          'p',
          'Must be a probability distribution summing to 1.0 (got $sum).',
        );
      }
      cumulative = List<double>.filled(populationSize, 0.0);
      var running = 0.0;
      for (var i = 0; i < populationSize; i++) {
        running += p[i] / sum;
        cumulative[i] = running;
      }
      cumulative[populationSize - 1] = 1.0;
    }

    final targetDevice = out?.device ?? device ?? a.device;
    return NDArray.scope(() {
      final hostInput = a.toNDArray();
      final contiguousInput = hostInput.isContiguous
          ? hostInput
          : hostInput.copy();
      final hostResult = NDArray<T>.zeros(shape, a.dtype);
      final byteWidth = a.dtype.byteWidth;
      final sourceBytes = contiguousInput.pointer.cast<ffi.Uint8>();
      final destinationBytes = hostResult.pointer.cast<ffi.Uint8>();

      void copyElement(int fromIndex, int toIndex) {
        final fromOffset = fromIndex * byteWidth;
        final toOffset = toIndex * byteWidth;
        for (var b = 0; b < byteWidth; b++) {
          destinationBytes[toOffset + b] = sourceBytes[fromOffset + b];
        }
      }

      if (replace) {
        for (var i = 0; i < sampleCount; i++) {
          final int selectedIndex;
          if (cumulative == null) {
            selectedIndex = _nextInt(populationSize);
          } else {
            final u = _nextDouble();
            var lowIndex = 0;
            var highIndex = populationSize - 1;
            while (lowIndex < highIndex) {
              final mid = (lowIndex + highIndex) >>> 1;
              if (u < cumulative[mid]) {
                highIndex = mid;
              } else {
                lowIndex = mid + 1;
              }
            }
            selectedIndex = lowIndex;
          }
          copyElement(selectedIndex, i);
        }
      } else if (p == null) {
        final pool = List<int>.generate(populationSize, (index) => index);
        for (var i = 0; i < sampleCount; i++) {
          final j = i + _nextInt(populationSize - i);
          final temp = pool[i];
          pool[i] = pool[j];
          pool[j] = temp;
          copyElement(pool[i], i);
        }
      } else {
        final weights = List<double>.of(p);
        for (var i = 0; i < sampleCount; i++) {
          var totalWeight = 0.0;
          for (final w in weights) {
            totalWeight += w;
          }
          final u = _nextDouble() * totalWeight;
          var running = 0.0;
          var chosen = weights.length - 1;
          for (var j = 0; j < weights.length; j++) {
            running += weights[j];
            if (u <= running && weights[j] > 0.0) {
              chosen = j;
              break;
            }
          }
          copyElement(chosen, i);
          weights[chosen] = 0.0;
        }
      }

      return _writeOrWrapResult(hostResult, targetDevice, out);
    });
  }

  /// Randomly permutes a sequence or returns a permuted range.
  ///
  /// If [x] is an `int`, returns a 1D [GpuArray] of `Int64` (unless [out]
  /// specifies another integer/float dtype) containing a random permutation of
  /// `0..x-1`. The integer [x] must be non-negative.
  ///
  /// If [x] is a [GpuArray], returns a new [GpuArray] with its elements
  /// shuffled along the first axis. Neither [x] nor [out] may be disposed.
  GpuArray<T> permutation<T extends DTypeTag>(
    Object x, {
    GpuDevice? device,
    GpuArray<T>? out,
  }) {
    if (out != null && out.isDisposed) {
      throw StateError(
        'Cannot write permutation result to a disposed output GpuArray.',
      );
    }
    if (x is int) {
      if (x < 0) {
        throw ArgumentError.value(x, 'x', 'Must be non-negative.');
      }
      final targetDevice = out?.device ?? device ?? GpuDevice.defaultDevice;
      return NDArray.scope(() {
        final hostInt64 = NDArray<Int64>.zeros(<int>[x], DType.int64);
        final pointer = hostInt64.pointer.cast<ffi.Int64>();
        for (var i = 0; i < x; i++) {
          pointer[i] = i;
        }
        for (var i = x - 1; i > 0; i--) {
          final j = _nextInt(i + 1);
          final temp = pointer[i];
          pointer[i] = pointer[j];
          pointer[j] = temp;
        }
        return _writeOrWrapResult(hostInt64 as NDArray<T>, targetDevice, out);
      });
    } else if (x is GpuArray) {
      if (x.isDisposed) {
        throw StateError('Cannot permute a disposed GpuArray.');
      }
      if (x.rank == 0) {
        throw ArgumentError.value(
          x.shape,
          'x',
          'Must be at least 1-dimensional for permutation.',
        );
      }
      final targetDevice = out?.device ?? device ?? x.device;
      return NDArray.scope(() {
        final hostInput = x.toNDArray();
        final contiguous = hostInput.copy();
        final length = contiguous.shape[0];
        if (length > 1 && contiguous.size > 0) {
          final sliceBytes =
              (contiguous.size ~/ length) * contiguous.dtype.byteWidth;
          final rawBytes = contiguous.pointer.cast<ffi.Uint8>();
          for (var i = length - 1; i > 0; i--) {
            final j = _nextInt(i + 1);
            if (i != j) {
              final offsetI = i * sliceBytes;
              final offsetJ = j * sliceBytes;
              for (var b = 0; b < sliceBytes; b++) {
                final temp = rawBytes[offsetI + b];
                rawBytes[offsetI + b] = rawBytes[offsetJ + b];
                rawBytes[offsetJ + b] = temp;
              }
            }
          }
        }
        return _writeOrWrapResult(contiguous as NDArray<T>, targetDevice, out);
      });
    } else {
      throw ArgumentError.value(x, 'x', 'Must be an int or a GpuArray.');
    }
  }

  /// Shuffles the elements of a 1D [GpuArray] [a] in-place.
  ///
  /// The array [a] must not be disposed and must be 1-dimensional.
  void shuffle(GpuArray a) {
    if (a.isDisposed) {
      throw StateError('Cannot shuffle a disposed GpuArray.');
    }
    if (a.rank != 1) {
      throw ArgumentError.value(
        a.shape,
        'a',
        'Must be 1-dimensional for in-place shuffle.',
      );
    }
    final length = a.size;
    if (length <= 1) return;

    NDArray.scope(() {
      final hostArray = a.toNDArray();
      final contiguous = hostArray.isContiguous ? hostArray : hostArray.copy();
      final byteWidth = a.dtype.byteWidth;
      final rawBytes = contiguous.pointer.cast<ffi.Uint8>();
      for (var i = length - 1; i > 0; i--) {
        final j = _nextInt(i + 1);
        if (i != j) {
          final offsetI = i * byteWidth;
          final offsetJ = j * byteWidth;
          for (var b = 0; b < byteWidth; b++) {
            final temp = rawBytes[offsetI + b];
            rawBytes[offsetI + b] = rawBytes[offsetJ + b];
            rawBytes[offsetJ + b] = temp;
          }
        }
      }
      _writeOrWrapResult(contiguous, a.device, a);
    });
  }
}

/// The default global [RandomState] instance.
final RandomState defaultRng = RandomState();

/// Seeds the [defaultRng] global random number generator with [newSeed].
void seed(int newSeed) => defaultRng.seed(newSeed);

/// Samples uniform values in `[0.0, 1.0)` with the given [shape].
///
/// If [out] is provided, the result is written directly into [out].
GpuArray<Float64> rand([
  List<int> shape = const <int>[],
  GpuDevice? device,
  GpuArray<Float64>? out,
]) => defaultRng.rand(shape, device, out);

/// Samples standard normal $\mathcal{N}(0, 1)$ values with the given [shape].
///
/// If [out] is provided, the result is written directly into [out].
GpuArray<Float64> randn([
  List<int> shape = const <int>[],
  GpuDevice? device,
  GpuArray<Float64>? out,
]) => defaultRng.randn(shape, device, out);

/// Samples random integers in `[low, high)` (or `[0, low)` if [high] is null).
///
/// If [out] is provided, the result is written directly into [out].
GpuArray<Int64> randint(
  int low, [
  int? high,
  List<int> shape = const <int>[],
  GpuDevice? device,
  GpuArray<Int64>? out,
]) => defaultRng.randint(low, high, shape, device, out);

/// Samples from a uniform distribution over `[low, high)`.
///
/// If [out] is provided, the result is written directly into [out].
GpuArray<Float64> uniform({
  double low = 0.0,
  double high = 1.0,
  List<int> shape = const <int>[],
  GpuDevice? device,
  GpuArray<Float64>? out,
}) => defaultRng.uniform(
  low: low,
  high: high,
  shape: shape,
  device: device,
  out: out,
);

/// Samples from a normal distribution $\mathcal{N}(\text{loc}, \text{scale}^2)$.
///
/// If [out] is provided, the result is written directly into [out].
GpuArray<Float64> normal({
  double loc = 0.0,
  double scale = 1.0,
  List<int> shape = const <int>[],
  GpuDevice? device,
  GpuArray<Float64>? out,
}) => defaultRng.normal(
  loc: loc,
  scale: scale,
  shape: shape,
  device: device,
  out: out,
);

/// Samples from the standard normal distribution $\mathcal{N}(0, 1)$.
///
/// If [out] is provided, the result is written directly into [out].
GpuArray<Float64> standardNormal({
  List<int> shape = const <int>[],
  GpuDevice? device,
  GpuArray<Float64>? out,
}) => defaultRng.standardNormal(shape: shape, device: device, out: out);

/// Snake-case alias for [standardNormal].
GpuArray<Float64> standard_normal({
  List<int> shape = const <int>[],
  GpuDevice? device,
  GpuArray<Float64>? out,
}) => defaultRng.standardNormal(shape: shape, device: device, out: out);

/// Samples from an exponential distribution with scale parameter [scale].
///
/// If [out] is provided, the result is written directly into [out].
GpuArray<Float64> exponential({
  double scale = 1.0,
  List<int> shape = const <int>[],
  GpuDevice? device,
  GpuArray<Float64>? out,
}) => defaultRng.exponential(
  scale: scale,
  shape: shape,
  device: device,
  out: out,
);

/// Generates a random sample from a 1D [GpuArray] [a].
///
/// If [out] is provided, the result is written directly into [out].
GpuArray<T> choice<T extends DTypeTag>(
  GpuArray<T> a, {
  List<int> shape = const <int>[1],
  bool replace = true,
  List<double>? p,
  GpuDevice? device,
  GpuArray<T>? out,
}) => defaultRng.choice(
  a,
  shape: shape,
  replace: replace,
  p: p,
  device: device,
  out: out,
);

/// Randomly permutes a sequence or returns a permuted range `0..x-1`.
///
/// If [out] is provided, the result is written directly into [out].
GpuArray<T> permutation<T extends DTypeTag>(
  Object x, {
  GpuDevice? device,
  GpuArray<T>? out,
}) => defaultRng.permutation<T>(x, device: device, out: out);

/// Shuffles a 1D [GpuArray] [a] in-place.
void shuffle(GpuArray a) => defaultRng.shuffle(a);
