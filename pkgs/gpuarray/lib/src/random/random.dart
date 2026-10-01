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

import '../device.dart';
import '../dtype.dart';
import '../exceptions.dart';
import '../fft/fft_wgsl.dart';
import '../gpu_array.dart';
import 'random_sample_wgsl.dart';
import 'random_wgsl.dart';

const int _workgroupSize = 64;

int _workgroupsFor(int totalItems) =>
    totalItems <= 0 ? 1 : (totalItems + _workgroupSize - 1) ~/ _workgroupSize;

int _float32Bits(double value) {
  final byteData = ByteData(4)..setFloat32(0, value, Endian.little);
  return byteData.getUint32(0, Endian.little);
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

/// Counter-based `Philox4x32-10` pseudo-random number generator engine.
///
/// Implements Salmon et al. (SC'11) 10-round Philox bijection over 4x32-bit
/// counters and a 2x32-bit key.
final class Philox4x32Engine {
  static const int _philoxM0 = 0xD2511F53;
  static const int _philoxM1 = 0xCD9E8D57;
  static const int _philoxW0 = 0x9E3779B9;
  static const int _philoxW1 = 0xBB67AE85;

  int _key0;
  int _key1;
  int _counter;
  int _counter0;
  int _counter1;
  int _counter2;
  int _counter3;

  final Uint32List _wordBuffer = Uint32List(4);
  int _wordPosition = 4;

  /// Creates a [Philox4x32Engine] initialized with [seed] and starting [counter].
  Philox4x32Engine({
    int seed = 0,
    int counter = 0,
    int? counter0,
    int counter1 = 0,
    int counter2 = 0,
    int counter3 = 0,
  }) : _key0 = seed & 0xFFFFFFFF,
       _key1 = (seed >>> 32) & 0xFFFFFFFF,
       _counter = counter,
       _counter0 = (counter0 ?? counter) & 0xFFFFFFFF,
       _counter1 = counter0 != null
           ? (counter1 & 0xFFFFFFFF)
           : ((counter >>> 32) & 0xFFFFFFFF),
       _counter2 = counter2 & 0xFFFFFFFF,
       _counter3 = counter3 & 0xFFFFFFFF;

  /// The current 64-bit word counter offset of this engine.
  int get counter => _counter;

  /// Lower 32-bit key word.
  int get key0 => _key0;

  /// Upper 32-bit key word.
  int get key1 => _key1;

  /// Counter word 0 (least significant 32 bits).
  int get counter0 => _counter0;

  /// Counter word 1.
  int get counter1 => _counter1;

  /// Counter word 2.
  int get counter2 => _counter2;

  /// Counter word 3 (most significant 32 bits).
  int get counter3 => _counter3;

  /// Resets the engine state to [seed] and [counter].
  void reset({required int seed, int counter = 0}) {
    _key0 = seed & 0xFFFFFFFF;
    _key1 = (seed >>> 32) & 0xFFFFFFFF;
    _counter = counter;
    _counter0 = counter & 0xFFFFFFFF;
    _counter1 = (counter >>> 32) & 0xFFFFFFFF;
    _counter2 = 0;
    _counter3 = 0;
    _wordPosition = 4;
  }

  /// Creates an independent stream with an incremented counter domain.
  Philox4x32Engine fork([int streamId = 1]) {
    return Philox4x32Engine(
      seed: (_key0 | (_key1 << 32)),
      counter: _counter,
      counter0: _counter0,
      counter1: _counter1,
      counter2: (_counter2 + streamId) & 0xFFFFFFFF,
      counter3: _counter3,
    ).._key1 = _key1;
  }

  static (int, int) _mulHiLo32(int a, int b) {
    final product = (a & 0xFFFFFFFF) * (b & 0xFFFFFFFF);
    final lo = product & 0xFFFFFFFF;
    final hi = (product >>> 32) & 0xFFFFFFFF;
    return (hi, lo);
  }

  /// Executes 10 rounds of `Philox4x32` on a 4-word [counter] and 2-word [key].
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
      final (hi0, lo0) = _mulHiLo32(_philoxM0, c0);
      final (hi1, lo1) = _mulHiLo32(_philoxM1, c2);
      c0 = hi1 ^ c1 ^ k0;
      c1 = lo1;
      c2 = hi0 ^ c3 ^ k1;
      c3 = lo0;
      k0 = (k0 + _philoxW0) & 0xFFFFFFFF;
      k1 = (k1 + _philoxW1) & 0xFFFFFFFF;
    }

    return <int>[c0, c1, c2, c3];
  }

  /// Advances the internal 128-bit counter by [blocks] 4x32-bit blocks.
  ///
  /// The [blocks] count must be non-negative.
  void skipBlocks(int blocks) {
    RangeError.checkNotNegative(blocks, 'blocks');
    if (blocks == 0) return;
    _counter += blocks * 4;
    final lowAdd = _counter0 + (blocks & 0xFFFFFFFF);
    _counter0 = lowAdd & 0xFFFFFFFF;
    var carry = (lowAdd >>> 32) + ((blocks >>> 32) & 0xFFFFFFFF);
    if (carry > 0) {
      final c1 = _counter1 + (carry & 0xFFFFFFFF);
      _counter1 = c1 & 0xFFFFFFFF;
      carry = c1 >>> 32;
      if (carry > 0) {
        final c2 = _counter2 + carry;
        _counter2 = c2 & 0xFFFFFFFF;
        carry = c2 >>> 32;
        if (carry > 0) {
          _counter3 = (_counter3 + carry) & 0xFFFFFFFF;
        }
      }
    }
    _wordPosition = 4;
  }

  /// Evaluates 10 rounds of `Philox4x32` on the current counter and increments it by 4 words.
  List<int> nextBlock() {
    final c0 = _counter & 0xFFFFFFFF;
    final c1 = (_counter >>> 32) & 0xFFFFFFFF;
    final result = philox4x32TenRounds(
      <int>[c0, c1, (c0 + 1) & 0xFFFFFFFF, (c1 + 1) & 0xFFFFFFFF],
      <int>[_key0, _key1],
    );
    skipBlocks(1);
    _wordBuffer[0] = result[0];
    _wordBuffer[1] = result[1];
    _wordBuffer[2] = result[2];
    _wordBuffer[3] = result[3];
    _wordPosition = 0;
    return result;
  }

  /// Generates the next pseudorandom 32-bit unsigned integer in `[0, 2^32)`.
  int nextUint32() {
    if (_wordPosition >= 4) {
      nextBlock();
    }
    final word = _wordBuffer[_wordPosition];
    _wordPosition++;
    return word;
  }

  /// Generates a 53-bit uniform `double` in `[0.0, 1.0)`.
  double nextFloat64() {
    final high26 = nextUint32() >>> 5;
    final low27 = nextUint32() >>> 6;
    return (high26 * 67108864.0 + low27) * (1.0 / 9007199254740992.0);
  }

  /// Generates a uniform `double` in `(0.0, 1.0)` suitable for logarithmic transforms.
  double nextOpenFloat64() {
    double sample;
    do {
      sample = nextFloat64();
    } while (sample <= 0.0 || sample >= 1.0);
    return sample;
  }
}

/// Stateful GPU random number generator backed by [Philox4x32Engine] WGSL compute shaders.
final class RandomState {
  int _seed;
  final Philox4x32Engine _engine;

  /// Creates a [RandomState] seeded with [seedValue].
  RandomState([int? seedValue])
    : _seed = seedValue ?? DateTime.now().microsecondsSinceEpoch,
      _engine = Philox4x32Engine(
        seed: seedValue ?? DateTime.now().microsecondsSinceEpoch,
      );

  /// The current seed of this generator.
  int get currentSeed => _seed;

  /// The underlying counter-based `Philox4x32-10` engine.
  Philox4x32Engine get engine => _engine;

  /// Reseeds this generator with [newSeed].
  void seed(int newSeed) {
    _seed = newSeed;
    _engine.reset(seed: newSeed);
  }

  static int _validateShape(List<int> shape) {
    if (shape.length > 8) {
      throw ArgumentError.value(
        shape,
        'shape',
        'Must have rank at most 8 for GPU random shaders',
      );
    }
    var count = 1;
    for (var i = 0; i < shape.length; i++) {
      final dim = shape[i];
      if (dim < 0) {
        throw ArgumentError.value(
          shape,
          'shape',
          'Must not contain negative dimensions (got $dim at index $i)',
        );
      }
      count *= dim;
    }
    return count;
  }

  static void _validateOut<T extends DTypeTag>(
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
      throw UnsupportedError(
        'Cannot write random output into a broadcasted view.',
      );
    }
    if (!identical(out.device, expectedDevice)) {
      throw ArgumentError.value(
        out,
        'out',
        'Must reside on the same GpuDevice ($expectedDevice)',
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
      throw GpuShapeMismatchException('random', expectedShape, out.shape);
    }
    for (var i = 0; i < expectedShape.length; i++) {
      if (out.shape[i] != expectedShape[i]) {
        throw GpuShapeMismatchException('random', expectedShape, out.shape);
      }
    }
  }

  GpuArray<Float64> _dispatchFloat64Distribution({
    required List<int> shape,
    required int mode,
    required int param0Lo,
    required int param0Hi,
    required int param1Lo,
    required int param1Hi,
    int param2Lo = 0,
    int param2Hi = 0,
    GpuDevice? device,
    GpuArray<Float64>? out,
  }) {
    if (out != null && out.isDisposed) {
      throw StateError('Output GpuArray has already been disposed.');
    }
    final totalElements = _validateShape(shape);
    final targetDevice = device ?? out?.device ?? GpuDevice.defaultDevice;
    _validateOut(out, shape, DType.float64, targetDevice);
    final destination =
        out ?? GpuArray.empty(shape, DType.float64, device: targetDevice);
    if (totalElements == 0) {
      return destination;
    }
    final uniforms = <int>[
      totalElements,
      destination.ndim,
      destination.offsetElements,
      mode,
      _engine.key0,
      _engine.key1,
      _engine.counter0,
      _engine.counter1,
      _engine.counter2,
      _engine.counter3,
      param0Lo,
      param0Hi,
      param1Lo,
      param1Hi,
      param2Lo,
      param2Hi,
      ..._packVec8U32(destination.shape),
      ..._packVec8I32(destination.strides),
    ];
    _engine.skipBlocks(math.max(1, totalElements));
    targetDevice.backend.dispatchComputePipeline(
      shaderModule: buildRandomFloat64Shader(),
      buffers: [destination.buffer],
      uniforms: uniforms,
      workgroupsX: _workgroupsFor(totalElements),
    );
    return destination;
  }

  GpuArray<Int64> _dispatchInt64Distribution({
    required List<int> shape,
    required int mode,
    required int param0Lo,
    required int param0Hi,
    required int param1Lo,
    required int param1Hi,
    GpuDevice? device,
    GpuArray<Int64>? out,
  }) {
    if (out != null && out.isDisposed) {
      throw StateError('Output GpuArray has already been disposed.');
    }
    final totalElements = _validateShape(shape);
    final targetDevice = device ?? out?.device ?? GpuDevice.defaultDevice;
    _validateOut(out, shape, DType.int64, targetDevice);
    final destination =
        out ?? GpuArray.empty(shape, DType.int64, device: targetDevice);
    if (totalElements == 0) {
      return destination;
    }
    final uniforms = <int>[
      totalElements,
      destination.ndim,
      destination.offsetElements,
      mode,
      _engine.key0,
      _engine.key1,
      _engine.counter0,
      _engine.counter1,
      _engine.counter2,
      _engine.counter3,
      param0Lo,
      param0Hi,
      param1Lo,
      param1Hi,
      0,
      0,
      ..._packVec8U32(destination.shape),
      ..._packVec8I32(destination.strides),
    ];
    _engine.skipBlocks(math.max(1, totalElements));
    targetDevice.backend.dispatchComputePipeline(
      shaderModule: buildRandomInt64Shader(),
      buffers: [destination.buffer],
      uniforms: uniforms,
      workgroupsX: _workgroupsFor(totalElements),
    );
    return destination;
  }

  /// Generates uniform random values in `[0.0, 1.0)` with the given [shape] on the GPU.
  GpuArray<Float64> rand([
    List<int> shape = const <int>[],
    GpuDevice? device,
    GpuArray<Float64>? out,
  ]) => uniform(low: 0.0, high: 1.0, shape: shape, device: device, out: out);

  /// Generates standard normal `N(0, 1)` random values with the given [shape] on the GPU.
  GpuArray<Float64> randn([
    List<int> shape = const <int>[],
    GpuDevice? device,
    GpuArray<Float64>? out,
  ]) => normal(loc: 0.0, scale: 1.0, shape: shape, device: device, out: out);

  /// Generates uniform random values in `[low, high)` with the given [shape] on the GPU.
  ///
  /// The [low] bound must be less than or equal to [high].
  GpuArray<Float64> uniform({
    double low = 0.0,
    double high = 1.0,
    List<int> shape = const <int>[],
    GpuDevice? device,
    GpuArray<Float64>? out,
  }) {
    if (out != null && out.isDisposed) {
      throw StateError('Output GpuArray has already been disposed.');
    }
    if (low > high || low.isNaN || high.isNaN) {
      throw ArgumentError.value(low, 'low', 'Must be <= high ($high)');
    }
    final (lowLo, lowHi) = encodeDoubleFloatUniform(low);
    final (spanLo, spanHi) = encodeDoubleFloatUniform(high - low);
    return _dispatchFloat64Distribution(
      shape: shape,
      mode: 0,
      param0Lo: lowLo,
      param0Hi: lowHi,
      param1Lo: spanLo,
      param1Hi: spanHi,
      device: device,
      out: out,
    );
  }

  /// Generates normal (Gaussian) random values with mean [loc] and standard deviation [scale] on the GPU.
  ///
  /// The [scale] parameter must be positive.
  GpuArray<Float64> normal({
    double loc = 0.0,
    double scale = 1.0,
    List<int> shape = const <int>[],
    GpuDevice? device,
    GpuArray<Float64>? out,
  }) {
    if (out != null && out.isDisposed) {
      throw StateError('Output GpuArray has already been disposed.');
    }
    if (scale <= 0.0 || scale.isNaN) {
      throw ArgumentError.value(scale, 'scale', 'Must be positive');
    }
    final (locLo, locHi) = encodeDoubleFloatUniform(loc);
    final (scaleLo, scaleHi) = encodeDoubleFloatUniform(scale);
    return _dispatchFloat64Distribution(
      shape: shape,
      mode: 1,
      param0Lo: locLo,
      param0Hi: locHi,
      param1Lo: scaleLo,
      param1Hi: scaleHi,
      device: device,
      out: out,
    );
  }

  /// Generates standard normal `N(0, 1)` random values with the given [shape] on the GPU.
  GpuArray<Float64> standardNormal({
    List<int> shape = const <int>[],
    GpuDevice? device,
    GpuArray<Float64>? out,
  }) => normal(loc: 0.0, scale: 1.0, shape: shape, device: device, out: out);

  /// Alias for [standardNormal].
  // ignore: non_constant_identifier_names
  GpuArray<Float64> standard_normal({
    List<int> shape = const <int>[],
    GpuDevice? device,
    GpuArray<Float64>? out,
  }) => normal(loc: 0.0, scale: 1.0, shape: shape, device: device, out: out);

  /// Generates truncated normal random values with standardized bounds `[low, high]`,
  /// mean [loc], and standard deviation [scale] on the GPU.
  ///
  /// The [low] bound must be strictly less than [high], and [scale] must be positive.
  GpuArray<Float64> truncatedNormal({
    double low = -2.0,
    double high = 2.0,
    double loc = 0.0,
    double scale = 1.0,
    List<int> shape = const <int>[],
    GpuDevice? device,
    GpuArray<Float64>? out,
  }) {
    if (out != null && out.isDisposed) {
      throw StateError('Output GpuArray has already been disposed.');
    }
    if (low >= high || low.isNaN || high.isNaN) {
      throw ArgumentError.value(
        low,
        'low',
        'Must be strictly less than high ($high)',
      );
    }
    if (scale <= 0.0 || scale.isNaN) {
      throw ArgumentError.value(scale, 'scale', 'Must be positive');
    }
    final (locLo, locHi) = encodeDoubleFloatUniform(loc);
    final (scaleLo, scaleHi) = encodeDoubleFloatUniform(scale);
    return _dispatchFloat64Distribution(
      shape: shape,
      mode: 4,
      param0Lo: _float32Bits(low),
      param0Hi: _float32Bits(high),
      param1Lo: locLo,
      param1Hi: locHi,
      param2Lo: scaleLo,
      param2Hi: scaleHi,
      device: device,
      out: out,
    );
  }

  /// Generates random 64-bit integers in `[low, high)` (or `[0, low)` when [high] is omitted) on the GPU.
  ///
  /// The lower bound must be strictly less than the upper bound.
  GpuArray<Int64> randint(
    int low, [
    int? high,
    List<int> shape = const <int>[],
    GpuDevice? device,
    GpuArray<Int64>? out,
  ]) {
    if (out != null && out.isDisposed) {
      throw StateError('Output GpuArray has already been disposed.');
    }
    final actualLow = high == null ? 0 : low;
    final actualHigh = high ?? low;
    if (actualLow >= actualHigh) {
      throw ArgumentError.value(
        actualLow,
        'low',
        'Must be strictly less than high ($actualHigh)',
      );
    }
    final span = actualHigh - actualLow;
    final lowLo = actualLow & 0xFFFFFFFF;
    final lowHi = (actualLow >> 32) & 0xFFFFFFFF;
    final spanLo = span & 0xFFFFFFFF;
    final spanHi = (span >>> 32) & 0xFFFFFFFF;
    return _dispatchInt64Distribution(
      shape: shape,
      mode: 0,
      param0Lo: lowLo,
      param0Hi: lowHi,
      param1Lo: spanLo,
      param1Hi: spanHi,
      device: device,
      out: out,
    );
  }

  /// Generates Bernoulli random values (`0.0` or `1.0`) with success probability [p] on the GPU.
  ///
  /// The probability [p] must lie in `[0.0, 1.0]`.
  GpuArray<Float64> bernoulli({
    double p = 0.5,
    List<int> shape = const <int>[],
    GpuDevice? device,
    GpuArray<Float64>? out,
  }) {
    if (out != null && out.isDisposed) {
      throw StateError('Output GpuArray has already been disposed.');
    }
    if (p < 0.0 || p > 1.0 || p.isNaN) {
      throw ArgumentError.value(p, 'p', 'Must be in [0.0, 1.0]');
    }
    return _dispatchFloat64Distribution(
      shape: shape,
      mode: 3,
      param0Lo: _float32Bits(p),
      param0Hi: 0,
      param1Lo: 0,
      param1Hi: 0,
      device: device,
      out: out,
    );
  }

  /// Generates exponential random values with scale parameter [scale] (`1 / lambda`) on the GPU.
  ///
  /// The [scale] parameter must be positive.
  GpuArray<Float64> exponential({
    double scale = 1.0,
    List<int> shape = const <int>[],
    GpuDevice? device,
    GpuArray<Float64>? out,
  }) {
    if (out != null && out.isDisposed) {
      throw StateError('Output GpuArray has already been disposed.');
    }
    if (scale <= 0.0 || scale.isNaN) {
      throw ArgumentError.value(scale, 'scale', 'Must be positive');
    }
    final (scaleLo, scaleHi) = encodeDoubleFloatUniform(scale);
    return _dispatchFloat64Distribution(
      shape: shape,
      mode: 2,
      param0Lo: scaleLo,
      param0Hi: scaleHi,
      param1Lo: 0,
      param1Hi: 0,
      device: device,
      out: out,
    );
  }

  /// Generates Gamma-distributed random values with shape [alpha] and scale [scale] on the GPU.
  ///
  /// Both [alpha] and [scale] must be positive.
  GpuArray<Float64> gamma({
    double alpha = 1.0,
    double scale = 1.0,
    List<int> shape = const <int>[],
    GpuDevice? device,
    GpuArray<Float64>? out,
  }) {
    if (out != null && out.isDisposed) {
      throw StateError('Output GpuArray has already been disposed.');
    }
    if (alpha <= 0.0 || alpha.isNaN) {
      throw ArgumentError.value(alpha, 'alpha', 'Must be positive');
    }
    if (scale <= 0.0 || scale.isNaN) {
      throw ArgumentError.value(scale, 'scale', 'Must be positive');
    }
    final (scaleLo, scaleHi) = encodeDoubleFloatUniform(scale);
    return _dispatchFloat64Distribution(
      shape: shape,
      mode: 5,
      param0Lo: _float32Bits(alpha),
      param0Hi: 0,
      param1Lo: scaleLo,
      param1Hi: scaleHi,
      device: device,
      out: out,
    );
  }

  /// Generates Beta-distributed random values in `[0.0, 1.0]` with concentration parameters [a] and [b] on the GPU.
  ///
  /// Both [a] and [b] must be positive.
  GpuArray<Float64> beta({
    double a = 1.0,
    double b = 1.0,
    List<int> shape = const <int>[],
    GpuDevice? device,
    GpuArray<Float64>? out,
  }) {
    if (out != null && out.isDisposed) {
      throw StateError('Output GpuArray has already been disposed.');
    }
    if (a <= 0.0 || a.isNaN) {
      throw ArgumentError.value(a, 'a', 'Must be positive');
    }
    if (b <= 0.0 || b.isNaN) {
      throw ArgumentError.value(b, 'b', 'Must be positive');
    }
    return _dispatchFloat64Distribution(
      shape: shape,
      mode: 6,
      param0Lo: _float32Bits(a),
      param0Hi: _float32Bits(b),
      param1Lo: 0,
      param1Hi: 0,
      device: device,
      out: out,
    );
  }

  /// Generates chi-square distributed random values with [df] degrees of freedom on the GPU.
  ///
  /// The [df] parameter must be positive.
  GpuArray<Float64> chisquare({
    double df = 1.0,
    List<int> shape = const <int>[],
    GpuDevice? device,
    GpuArray<Float64>? out,
  }) {
    if (out != null && out.isDisposed) {
      throw StateError('Output GpuArray has already been disposed.');
    }
    if (df <= 0.0 || df.isNaN) {
      throw ArgumentError.value(df, 'df', 'Must be positive');
    }
    return gamma(
      alpha: df * 0.5,
      scale: 2.0,
      shape: shape,
      device: device,
      out: out,
    );
  }

  /// Generates Poisson-distributed 64-bit integers with rate parameter [lam] on the GPU.
  ///
  /// The rate [lam] must be non-negative.
  GpuArray<Int64> poisson({
    double lam = 1.0,
    List<int> shape = const <int>[],
    GpuDevice? device,
    GpuArray<Int64>? out,
  }) {
    if (out != null && out.isDisposed) {
      throw StateError('Output GpuArray has already been disposed.');
    }
    if (lam < 0.0 || lam.isNaN) {
      throw ArgumentError.value(lam, 'lam', 'Must be non-negative');
    }
    return _dispatchInt64Distribution(
      shape: shape,
      mode: 1,
      param0Lo: _float32Bits(lam),
      param0Hi: 0,
      param1Lo: 0,
      param1Hi: 0,
      device: device,
      out: out,
    );
  }

  /// Generates binomial-distributed 64-bit integers for [n] trials and success probability [p] on the GPU.
  ///
  /// The trial count [n] must be non-negative, and [p] must lie in `[0.0, 1.0]`.
  GpuArray<Int64> binomial({
    int n = 1,
    double p = 0.5,
    List<int> shape = const <int>[],
    GpuDevice? device,
    GpuArray<Int64>? out,
  }) {
    if (out != null && out.isDisposed) {
      throw StateError('Output GpuArray has already been disposed.');
    }
    if (n < 0) {
      throw ArgumentError.value(n, 'n', 'Must be non-negative');
    }
    if (p < 0.0 || p > 1.0 || p.isNaN) {
      throw ArgumentError.value(p, 'p', 'Must be in [0.0, 1.0]');
    }
    return _dispatchInt64Distribution(
      shape: shape,
      mode: 2,
      param0Lo: n & 0xFFFFFFFF,
      param0Hi: _float32Bits(p),
      param1Lo: 0,
      param1Hi: 0,
      device: device,
      out: out,
    );
  }

  /// Samples class indices (`Int64`) from [logitsOrProbs] along its last axis on the GPU.
  ///
  /// When [fromLogits] is `true` (default), [logitsOrProbs] is interpreted as unnormalized
  /// log-probabilities; when `false`, as non-negative probabilities.
  GpuArray<Int64> categorical(
    GpuArray<DTypeTag> logitsOrProbs, {
    List<int>? shape,
    bool fromLogits = true,
    GpuDevice? device,
    GpuArray<Int64>? out,
  }) {
    if (logitsOrProbs.isDisposed) {
      throw StateError(
        'Cannot sample categorical from a disposed logitsOrProbs array.',
      );
    }
    if (out != null && out.isDisposed) {
      throw StateError('Output GpuArray has already been disposed.');
    }
    if (logitsOrProbs.ndim == 0 || logitsOrProbs.size == 0) {
      throw ArgumentError.value(
        logitsOrProbs.shape,
        'logitsOrProbs',
        'Must be a non-empty array with at least 1 dimension',
      );
    }
    final numClasses = logitsOrProbs.shape.last;
    if (numClasses <= 0) {
      throw ArgumentError.value(
        logitsOrProbs.shape,
        'logitsOrProbs',
        'Must have at least 1 class along the last axis',
      );
    }
    final List<int> targetShape;
    if (shape != null) {
      targetShape = shape;
    } else if (logitsOrProbs.ndim == 1) {
      targetShape = const [1];
    } else {
      targetShape = logitsOrProbs.shape.sublist(0, logitsOrProbs.ndim - 1);
    }
    final sampleCount = _validateShape(targetShape);
    final targetDevice = device ?? out?.device ?? logitsOrProbs.device;
    _validateOut(out, targetShape, DType.int64, targetDevice);

    final destination =
        out ?? GpuArray.empty(targetShape, DType.int64, device: targetDevice);
    if (sampleCount == 0) {
      return destination;
    }

    return ResourceScope.scope(() {
      GpuArray workingLogits = logitsOrProbs;
      if (workingLogits.dtype != DType.float32 &&
          workingLogits.dtype != DType.float64) {
        workingLogits = workingLogits.astype<Float32>(DType.float32);
      }
      if (!workingLogits.isContiguous && workingLogits.ndim > 2) {
        workingLogits = workingLogits.copy();
      }
      final batchRows = workingLogits.size ~/ numClasses;
      final rowStride = workingLogits.ndim >= 2
          ? workingLogits.strides[workingLogits.ndim - 2]
          : 0;
      final classStride = workingLogits.strides.last;

      final uniforms = <int>[
        sampleCount,
        math.max(1, batchRows),
        numClasses,
        fromLogits ? 1 : 0,
        workingLogits.offsetElements,
        rowStride & 0xFFFFFFFF,
        classStride & 0xFFFFFFFF,
        destination.ndim,
        destination.offsetElements,
        _engine.key0,
        _engine.key1,
        _engine.counter0,
        _engine.counter1,
        _engine.counter2,
        _engine.counter3,
        0,
        ..._packVec8U32(destination.shape),
        ..._packVec8I32(destination.strides),
      ];
      _engine.skipBlocks(math.max(1, sampleCount * numClasses));
      targetDevice.backend.dispatchComputePipeline(
        shaderModule: buildRandomCategoricalShader(workingLogits.dtype),
        buffers: [workingLogits.buffer, destination.buffer],
        uniforms: uniforms,
        workgroupsX: _workgroupsFor(sampleCount),
      );
      if (out == null) {
        destination.detachToParentScope();
      }
      return destination;
    });
  }

  /// Generates a random sample from a 1-D [GpuArray] [a] on the GPU.
  ///
  /// The input [a] must be 1-dimensional and non-empty. When [replace] is `false`,
  /// the requested sample size must not exceed `a.size`.
  GpuArray<T> choice<T extends DTypeTag>(
    GpuArray<T> a, {
    List<int> shape = const <int>[1],
    bool replace = true,
    List<double>? p,
    GpuDevice? device,
    GpuArray<T>? out,
  }) {
    if (a.isDisposed) {
      throw StateError('Cannot sample from a disposed GpuArray.');
    }
    if (out != null && out.isDisposed) {
      throw StateError('Output GpuArray has already been disposed.');
    }
    if (a.ndim != 1 || a.size == 0) {
      throw ArgumentError.value(
        a.shape,
        'a',
        'Must be a non-empty 1-D GpuArray',
      );
    }
    final sampleCount = _validateShape(shape);
    final populationSize = a.size;
    if (!replace && sampleCount > populationSize) {
      throw ArgumentError.value(
        shape,
        'shape',
        'Must not request a larger sample ($sampleCount) than population ($populationSize) when replace is false',
      );
    }

    List<double>? normalizedProbs;
    if (p != null) {
      if (p.length != populationSize) {
        throw ArgumentError.value(
          p.length,
          'p',
          'Must match population length ($populationSize)',
        );
      }
      var probSum = 0.0;
      var positiveCount = 0;
      for (var i = 0; i < p.length; i++) {
        if (p[i] < 0.0 || p[i].isNaN) {
          throw ArgumentError.value(
            p,
            'p',
            'Must contain non-negative probabilities',
          );
        }
        if (p[i] > 0.0) positiveCount++;
        probSum += p[i];
      }
      if ((probSum - 1.0).abs() > 1e-5) {
        throw ArgumentError.value(p, 'p', 'Must sum to 1.0 (got $probSum)');
      }
      if (!replace && sampleCount > positiveCount) {
        throw ArgumentError.value(
          p,
          'p',
          'Must have at least $sampleCount non-zero probabilities (got $positiveCount) when replace is false',
        );
      }
      normalizedProbs = [for (final weight in p) weight / probSum];
    }

    final targetDevice = device ?? out?.device ?? a.device;
    _validateOut(out, shape, a.dtype, targetDevice);
    final destination =
        out ?? GpuArray.zeros(shape, a.dtype, device: targetDevice);
    if (sampleCount == 0) {
      return destination;
    }

    if (out != null && a.dtype.byteWidth < 4) {
      final zeroFill = GpuArray.zeros(shape, a.dtype, device: targetDevice);
      try {
        zeroFill.copy(out: destination);
      } finally {
        zeroFill.dispose();
      }
    }

    return ResourceScope.scope(() {
      final GpuArray<Int64> indexArray;
      if (normalizedProbs == null) {
        if (replace) {
          indexArray = randint(0, populationSize, shape, targetDevice);
        } else {
          indexArray = permutation<Int64>(populationSize, device: targetDevice);
        }
      } else {
        final probGpu = GpuArray.fromList(
          normalizedProbs,
          [populationSize],
          DType.float32,
          device: targetDevice,
        );
        indexArray = GpuArray.empty(
          [sampleCount],
          DType.int64,
          device: targetDevice,
        );
        final dispatchCount = replace ? sampleCount : populationSize;
        targetDevice.backend.dispatchComputePipeline(
          shaderModule: buildRandomWeightedIndexShader(),
          buffers: [probGpu.buffer, indexArray.buffer],
          uniforms: [
            sampleCount,
            populationSize,
            replace ? 1 : 0,
            0,
            _engine.key0,
            _engine.key1,
            _engine.counter0,
            _engine.counter1,
            _engine.counter2,
            _engine.counter3,
            0,
            0,
          ],
          workgroupsX: _workgroupsFor(dispatchCount),
        );
        _engine.skipBlocks(math.max(1, dispatchCount));
      }

      final gatherUniforms = <int>[
        sampleCount,
        destination.ndim,
        a.offsetElements,
        a.strides[0] & 0xFFFFFFFF,
        destination.offsetElements,
        0,
        0,
        0,
        ..._packVec8U32(destination.shape),
        ..._packVec8I32(destination.strides),
      ];
      targetDevice.backend.dispatchComputePipeline(
        shaderModule: buildRandomGather1dShader(a.dtype),
        buffers: [a.buffer, indexArray.buffer, destination.buffer],
        uniforms: gatherUniforms,
        workgroupsX: _workgroupsFor(sampleCount),
      );
      if (out == null) {
        destination.detachToParentScope();
      }
      return destination;
    });
  }

  /// Randomly permutes a sequence or returns a permuted range on the GPU.
  ///
  /// If [x] is an `int`, generates a random permutation of `[0, 1, ..., x - 1]` as `GpuArray<Int64>`.
  /// If [x] is a [GpuArray], returns a [GpuArray] with slices along axis 0 randomly permuted.
  GpuArray<T> permutation<T extends DTypeTag>(
    Object x, {
    GpuDevice? device,
    GpuArray<T>? out,
  }) {
    if (out != null && out.isDisposed) {
      throw StateError('Output GpuArray has already been disposed.');
    }
    if (x is int) {
      if (x < 0) {
        throw ArgumentError.value(x, 'x', 'Must be non-negative');
      }
      final targetDevice = device ?? out?.device ?? GpuDevice.defaultDevice;
      if (out != null) {
        _validateOut(out, [x], out.dtype, targetDevice);
      }
      if (x == 0) {
        if (out != null) {
          return out;
        }
        return GpuArray.empty([0], DType.int64, device: targetDevice)
            as GpuArray<T>;
      }
      var bits = 0;
      var bound = 1;
      while (bound < x) {
        bound <<= 1;
        bits++;
      }
      final canWriteDirectly =
          out != null &&
          out.dtype == DType.int64 &&
          out.isContiguous &&
          out.offsetElements == 0;
      final outPerm = canWriteDirectly
          ? (out as GpuArray<Int64>)
          : GpuArray.empty([x], DType.int64, device: targetDevice);
      try {
        targetDevice.backend.dispatchComputePipeline(
          shaderModule: buildRandomPermutationInt64Shader(),
          buffers: [outPerm.buffer],
          uniforms: [
            x,
            bits,
            _engine.key0,
            _engine.key1,
            _engine.counter0,
            _engine.counter1,
            _engine.counter2,
            _engine.counter3,
          ],
          workgroupsX: _workgroupsFor(x),
        );
        _engine.skipBlocks(math.max(1, x));
        if (out == null) {
          return outPerm as GpuArray<T>;
        }
        if (!canWriteDirectly) {
          if (out.dtype == DType.int64) {
            outPerm.copy(out: out as GpuArray<Int64>);
          } else {
            final converted = outPerm.astype<T>(out.dtype);
            try {
              converted.copy(out: out);
            } finally {
              converted.dispose();
            }
          }
        }
        return out;
      } finally {
        if (out != null && !canWriteDirectly) {
          outPerm.dispose();
        }
      }
    }

    if (x is GpuArray) {
      if (x.isDisposed) {
        throw StateError('Cannot permute a disposed GpuArray.');
      }
      if (x.ndim == 0) {
        throw ArgumentError.value(
          x.shape,
          'x',
          'Must have at least 1 dimension',
        );
      }
      if (x.ndim > 8) {
        throw ArgumentError.value(
          x.shape,
          'x',
          'Must have rank at most 8 for GPU permutation shaders',
        );
      }
      final targetDevice = device ?? out?.device ?? x.device;
      if (out != null) {
        _validateOut(out, x.shape, x.dtype as DType<T>, targetDevice);
      }
      final rowCount = x.shape[0];
      if (rowCount <= 1 || x.size == 0) {
        if (out != null) {
          (x as GpuArray<T>).copy(out: out);
          return out;
        }
        return x.copy() as GpuArray<T>;
      }
      final aliasesInput = out != null && identical(out.buffer, x.buffer);
      final destination = (out != null && !aliasesInput)
          ? out
          : GpuArray.zeros(x.shape, x.dtype, device: targetDevice)
                as GpuArray<T>;
      if (out != null && !aliasesInput && x.dtype.byteWidth < 4) {
        final zeroFill =
            GpuArray.zeros(x.shape, x.dtype, device: targetDevice)
                as GpuArray<T>;
        try {
          zeroFill.copy(out: destination);
        } finally {
          zeroFill.dispose();
        }
      }
      final permIndices = permutation<Int64>(rowCount, device: targetDevice);
      try {
        final sliceSize = x.size ~/ rowCount;
        final gatherUniforms = <int>[
          x.size,
          sliceSize,
          x.ndim,
          x.offsetElements,
          destination.offsetElements,
          0,
          0,
          0,
          ..._packVec8U32(x.shape),
          ..._packVec8I32(x.strides),
          ..._packVec8I32(destination.strides),
        ];
        targetDevice.backend.dispatchComputePipeline(
          shaderModule: buildRandomAxis0SliceGatherShader(x.dtype),
          buffers: [x.buffer, permIndices.buffer, destination.buffer],
          uniforms: gatherUniforms,
          workgroupsX: _workgroupsFor(x.size),
        );
        if (aliasesInput) {
          try {
            destination.copy(out: out);
          } finally {
            destination.dispose();
          }
          return out;
        }
        return destination;
      } finally {
        permIndices.dispose();
      }
    }

    throw ArgumentError.value(
      x,
      'x',
      'Must be a non-negative int or a GpuArray',
    );
  }

  /// Shuffles the contents of [a] in-place along its first axis on the GPU.
  ///
  /// The array [a] must not be disposed and must have at least 1 dimension.
  void shuffle<T extends DTypeTag>(GpuArray<T> a) {
    if (a.isDisposed) {
      throw StateError('Cannot shuffle a disposed GpuArray.');
    }
    if (a.ndim == 0) {
      throw ArgumentError.value(a.shape, 'a', 'Must have at least 1 dimension');
    }
    if (a.shape[0] <= 1 || a.size == 0) {
      return;
    }
    permutation<T>(a, device: a.device, out: a);
  }
}

/// Default global [RandomState] instance.
final RandomState defaultRng = RandomState();

/// Seeds the global [defaultRng] generator with [seedValue].
void seed(int seedValue) => defaultRng.seed(seedValue);

/// Generates uniform random values in `[0.0, 1.0)` with the given [shape] on the GPU.
GpuArray<Float64> rand([
  List<int> shape = const <int>[],
  GpuDevice? device,
  GpuArray<Float64>? out,
]) => defaultRng.rand(shape, device, out);

/// Generates standard normal `N(0, 1)` random values with the given [shape] on the GPU.
GpuArray<Float64> randn([
  List<int> shape = const <int>[],
  GpuDevice? device,
  GpuArray<Float64>? out,
]) => defaultRng.randn(shape, device, out);

/// Generates uniform random values in `[low, high)` with the given [shape] on the GPU.
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

/// Generates normal (Gaussian) random values with mean [loc] and standard deviation [scale] on the GPU.
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

/// Generates standard normal `N(0, 1)` random values with the given [shape] on the GPU.
GpuArray<Float64> standardNormal({
  List<int> shape = const <int>[],
  GpuDevice? device,
  GpuArray<Float64>? out,
}) => defaultRng.standardNormal(shape: shape, device: device, out: out);

/// Alias for [standardNormal].
// ignore: non_constant_identifier_names
GpuArray<Float64> standard_normal({
  List<int> shape = const <int>[],
  GpuDevice? device,
  GpuArray<Float64>? out,
}) => defaultRng.standardNormal(shape: shape, device: device, out: out);

/// Generates truncated normal random values with standardized bounds `[low, high]`,
/// mean [loc], and standard deviation [scale] on the GPU.
GpuArray<Float64> truncatedNormal({
  double low = -2.0,
  double high = 2.0,
  double loc = 0.0,
  double scale = 1.0,
  List<int> shape = const <int>[],
  GpuDevice? device,
  GpuArray<Float64>? out,
}) => defaultRng.truncatedNormal(
  low: low,
  high: high,
  loc: loc,
  scale: scale,
  shape: shape,
  device: device,
  out: out,
);

/// Generates random 64-bit integers in `[low, high)` (or `[0, low)` when [high] is omitted) on the GPU.
GpuArray<Int64> randint(
  int low, [
  int? high,
  List<int> shape = const <int>[],
  GpuDevice? device,
  GpuArray<Int64>? out,
]) => defaultRng.randint(low, high, shape, device, out);

/// Generates Bernoulli random values (`0.0` or `1.0`) with success probability [p] on the GPU.
GpuArray<Float64> bernoulli({
  double p = 0.5,
  List<int> shape = const <int>[],
  GpuDevice? device,
  GpuArray<Float64>? out,
}) => defaultRng.bernoulli(p: p, shape: shape, device: device, out: out);

/// Generates exponential random values with scale parameter [scale] on the GPU.
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

/// Generates Gamma-distributed random values with shape [alpha] and scale [scale] on the GPU.
GpuArray<Float64> gamma({
  double alpha = 1.0,
  double scale = 1.0,
  List<int> shape = const <int>[],
  GpuDevice? device,
  GpuArray<Float64>? out,
}) => defaultRng.gamma(
  alpha: alpha,
  scale: scale,
  shape: shape,
  device: device,
  out: out,
);

/// Generates Beta-distributed random values in `[0.0, 1.0]` with parameters [a] and [b] on the GPU.
GpuArray<Float64> beta({
  double a = 1.0,
  double b = 1.0,
  List<int> shape = const <int>[],
  GpuDevice? device,
  GpuArray<Float64>? out,
}) => defaultRng.beta(a: a, b: b, shape: shape, device: device, out: out);

/// Generates chi-square distributed random values with [df] degrees of freedom on the GPU.
GpuArray<Float64> chisquare({
  double df = 1.0,
  List<int> shape = const <int>[],
  GpuDevice? device,
  GpuArray<Float64>? out,
}) => defaultRng.chisquare(df: df, shape: shape, device: device, out: out);

/// Generates Poisson-distributed 64-bit integers with rate parameter [lam] on the GPU.
GpuArray<Int64> poisson({
  double lam = 1.0,
  List<int> shape = const <int>[],
  GpuDevice? device,
  GpuArray<Int64>? out,
}) => defaultRng.poisson(lam: lam, shape: shape, device: device, out: out);

/// Generates binomial-distributed 64-bit integers for [n] trials and success probability [p] on the GPU.
GpuArray<Int64> binomial({
  int n = 1,
  double p = 0.5,
  List<int> shape = const <int>[],
  GpuDevice? device,
  GpuArray<Int64>? out,
}) => defaultRng.binomial(n: n, p: p, shape: shape, device: device, out: out);

/// Samples class indices (`Int64`) from [logitsOrProbs] along its last axis on the GPU.
GpuArray<Int64> categorical(
  GpuArray<DTypeTag> logitsOrProbs, {
  List<int>? shape,
  bool fromLogits = true,
  GpuDevice? device,
  GpuArray<Int64>? out,
}) => defaultRng.categorical(
  logitsOrProbs,
  shape: shape,
  fromLogits: fromLogits,
  device: device,
  out: out,
);

/// Generates a random sample from a 1-D [GpuArray] [a] on the GPU.
GpuArray<T> choice<T extends DTypeTag>(
  GpuArray<T> a, {
  List<int> shape = const <int>[1],
  bool replace = true,
  List<double>? p,
  GpuDevice? device,
  GpuArray<T>? out,
}) => defaultRng.choice<T>(
  a,
  shape: shape,
  replace: replace,
  p: p,
  device: device,
  out: out,
);

/// Randomly permutes a sequence or returns a permuted range on the GPU.
GpuArray<T> permutation<T extends DTypeTag>(
  Object x, {
  GpuDevice? device,
  GpuArray<T>? out,
}) => defaultRng.permutation<T>(x, device: device, out: out);

/// Shuffles the contents of [a] in-place along its first axis on the GPU.
void shuffle<T extends DTypeTag>(GpuArray<T> a) => defaultRng.shuffle<T>(a);
