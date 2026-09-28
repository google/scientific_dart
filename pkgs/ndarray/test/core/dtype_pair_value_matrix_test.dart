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

import 'package:ndarray/ndarray.dart';
import 'package:ndarray/src/operations/helpers.dart'
    show castValue, resolveDType;
import 'package:test/test.dart';

enum _LayoutMode { contiguous, strided, masked }

NDArray<AnySpec> _makeArray(
  DType dt,
  List<int> shape, {
  required bool strided,
  required List<int> baseInts,
}) {
  final size = shape.fold<int>(1, (a, b) => a * b);
  final raw = <Object>[];
  for (var i = 0; i < size; i++) {
    final v = baseInts[i % baseInts.length];
    Object elem;
    if (dt == DType.boolean) {
      elem = v != 0;
    } else if (dt.isComplex) {
      elem = Complex(v.toDouble(), (v + 1).toDouble());
    } else if (dt.isFloating) {
      elem = v.toDouble();
    } else {
      elem = v;
    }
    raw.add(elem);
    if (strided) {
      // Interleave a dummy element that must never be read by strided kernels.
      raw.add(elem);
    }
  }
  final d = dt as DType<AnySpec>;
  if (strided) {
    final full = NDArray.fromList(raw, [size * 2], d);
    return full[const Slice(step: 2)].reshape(shape);
  }
  return NDArray.fromList(raw, shape, d);
}

NDArray<AnySpec> _makeSentinelArray(
  DType dt,
  List<int> shape, {
  Object? sentinelOverride,
}) {
  final size = shape.fold<int>(1, (a, b) => a * b);
  final Object sentinel =
      sentinelOverride ??
      switch (dt) {
        DType.boolean => true,
        DType.complex128 || DType.complex64 => Complex(77.0, -55.0),
        DType.float64 ||
        DType.float32 ||
        DType.float16 ||
        DType.bfloat16 => 77.0,
        _ => 77,
      };
  return NDArray.fromList(
    List<Object>.filled(size, sentinel),
    shape,
    dt as DType<AnySpec>,
  );
}

Complex _toComplex(Object? v) {
  if (v is Complex) return v;
  if (v is bool) return Complex(v ? 1.0 : 0.0, 0.0);
  return Complex((v as num).toDouble(), 0.0);
}

double _toDouble(Object? v) {
  if (v is bool) return v ? 1.0 : 0.0;
  return (v as num).toDouble();
}

int _toInt(Object? v) {
  if (v is bool) return v ? 1 : 0;
  return (v as num).toInt();
}

Object _wrapCast(Object val, DType dt) {
  final c = castValue(val, dt) as Object;
  if (c is int) {
    return switch (dt) {
      DType.int8 => c.toSigned(8),
      DType.uint8 => c.toUnsigned(8),
      DType.int16 => c.toSigned(16),
      DType.uint16 => c.toUnsigned(16),
      DType.int32 => c.toSigned(32),
      DType.uint32 => c.toUnsigned(32),
      _ => c,
    };
  }
  return c;
}

double _toleranceFor(DType dt) => switch (dt) {
  DType.float16 => 5e-2,
  DType.bfloat16 => 2e-1,
  DType.float32 || DType.complex64 => 1e-4,
  _ => 1e-9,
};

void _expectCellMatches(
  Object? actual,
  Object? expected,
  DType dt, {
  required String context,
}) {
  final tol = _toleranceFor(dt);
  if (dt == DType.boolean) {
    expect(actual, equals(expected), reason: context);
  } else if (dt.isComplex) {
    final got = actual as Complex;
    final exp = expected as Complex;
    expect(
      got.real.isNaN,
      equals(exp.real.isNaN),
      reason: '$context (real NaN)',
    );
    expect(
      got.imag.isNaN,
      equals(exp.imag.isNaN),
      reason: '$context (imag NaN)',
    );
    if (!exp.real.isNaN) {
      expect(
        got.real,
        closeTo(exp.real, tol * math.max(1.0, exp.real.abs())),
        reason: '$context (real)',
      );
    }
    if (!exp.imag.isNaN) {
      expect(
        got.imag,
        closeTo(exp.imag, tol * math.max(1.0, exp.imag.abs())),
        reason: '$context (imag)',
      );
    }
  } else if (dt.isFloating) {
    final got = (actual as num).toDouble();
    final exp = (expected as num).toDouble();
    expect(got.isNaN, equals(exp.isNaN), reason: '$context (NaN)');
    if (!exp.isNaN && !exp.isInfinite) {
      expect(
        got,
        closeTo(exp, tol * math.max(1.0, exp.abs())),
        reason: context,
      );
    } else if (exp.isInfinite) {
      expect(got, equals(exp), reason: context);
    }
  } else {
    expect(actual, equals(expected), reason: context);
  }
}

int _scalarGcd(int a, int b) {
  var u = a.abs();
  var v = b.abs();
  while (v != 0) {
    final t = v;
    v = u % v;
    u = t;
  }
  return u;
}

int _scalarLcm(int a, int b) {
  if (a == 0 || b == 0) return 0;
  return (a.abs() ~/ _scalarGcd(a, b)) * b.abs();
}

int _scalarFloorDivInt(int x, int y) {
  final res = x ~/ y;
  final rem = x % y;
  if (rem != 0 && ((x < 0) ^ (y < 0))) {
    return res - 1;
  }
  return res;
}

int _scalarRemainderInt(int x, int y) {
  final rem = x % y;
  if (rem != 0 && ((rem < 0) != (y < 0))) {
    return rem + y;
  }
  return rem;
}

double _scalarRemainderDouble(double x, double y) {
  final rem = x % y;
  if (rem != 0.0 && ((rem < 0.0) != (y < 0.0))) {
    return rem + y;
  }
  return rem;
}

double _scalarLogAddExp(double x, double y) {
  final hi = math.max(x, y);
  final lo = math.min(x, y);
  return hi + math.log(1.0 + math.exp(lo - hi));
}

double _scalarLogAddExp2(double x, double y) {
  const ln2 = math.ln2;
  final hi = math.max(x, y);
  final lo = math.min(x, y);
  return hi + math.log(1.0 + math.exp((lo - hi) * ln2)) / ln2;
}

Complex _complexPow(Complex base, Complex exp) {
  if (base.real == 0.0 && base.imag == 0.0) {
    if (exp.real == 0.0 && exp.imag == 0.0) return Complex(1.0, 0.0);
    return Complex(0.0, 0.0);
  }
  final r = math.sqrt(base.real * base.real + base.imag * base.imag);
  final theta = math.atan2(base.imag, base.real);
  final lnR = math.log(r);
  final mag = math.exp(exp.real * lnR - exp.imag * theta);
  final angle = exp.real * theta + exp.imag * lnR;
  return Complex(mag * math.cos(angle), mag * math.sin(angle));
}

void main() {
  const shape = [2, 2];
  const maskPattern = [true, false, true, true];

  group('DType Pair Value Matrix & Sentinel Buffer Invariants', () {
    test(
      'All 15x15 ordered DType pairs x {contiguous, strided, masked} produce exact reference values for binary ops',
      () {
        for (final dtA in DType.values) {
          for (final dtB in DType.values) {
            for (final mode in _LayoutMode.values) {
              NDArray.scope(() {
                final isStrided = mode == _LayoutMode.strided;
                final useMask = mode == _LayoutMode.masked;
                final a = _makeArray(
                  dtA,
                  shape,
                  strided: isStrided,
                  baseInts: dtA == DType.boolean
                      ? const [1, 0, 1, 0]
                      : const [6, 4, 3, 2],
                );
                final b = _makeArray(
                  dtB,
                  shape,
                  strided: isStrided,
                  baseInts: const [2, 3, 2, 1],
                );
                final where = useMask
                    ? NDArray.fromList(maskPattern, shape, DType.boolean)
                    : null;

                final resolved = resolveDType(dtA, dtB);

                void verifyGrid(
                  NDArray res,
                  DType expectedDType,
                  Object Function(Object? va, Object? vb) refFn,
                  String opName,
                ) {
                  expect(
                    res.dtype,
                    equals(expectedDType),
                    reason: '$opName($dtA, $dtB, $mode) dtype',
                  );
                  expect(
                    res.shape,
                    equals(shape),
                    reason: '$opName($dtA, $dtB, $mode) shape',
                  );
                  var flat = 0;
                  for (var r = 0; r < shape[0]; r++) {
                    for (var c = 0; c < shape[1]; c++) {
                      final m = !useMask || maskPattern[flat];
                      final actual = res.getCell([r, c]);
                      final Object expected;
                      if (!m) {
                        expected = castValue(
                          expectedDType == DType.boolean
                              ? false
                              : (expectedDType.isComplex ? Complex(0, 0) : 0),
                          expectedDType,
                        );
                      } else {
                        expected = _wrapCast(
                          refFn(a.getCell([r, c]), b.getCell([r, c])),
                          expectedDType,
                        );
                      }
                      _expectCellMatches(
                        actual,
                        expected,
                        expectedDType,
                        context: '$opName($dtA, $dtB, $mode) at [$r, $c]',
                      );
                      flat++;
                    }
                  }
                }

                if (dtA != dtB) {
                  expect(() => add(a, b, where: where), throwsArgumentError);
                  expect(
                    () => subtract(a, b, where: where),
                    throwsArgumentError,
                  );
                  expect(
                    () => multiply(a, b, where: where),
                    throwsArgumentError,
                  );
                  expect(() => divide(a, b, where: where), throwsArgumentError);
                  expect(() => power(a, b, where: where), throwsArgumentError);
                  expect(() => hypot(a, b, where: where), throwsArgumentError);
                  expect(
                    () => floorDivide(a, b, where: where),
                    throwsArgumentError,
                  );
                  expect(
                    () => remainder(a, b, where: where),
                    throwsArgumentError,
                  );
                  expect(() => fmod(a, b, where: where), throwsArgumentError);
                  expect(
                    () => heaviside(a, b, where: where),
                    throwsArgumentError,
                  );
                  expect(
                    () => logaddexp(a, b, where: where),
                    throwsA(
                      anyOf(isA<ArgumentError>(), isA<UnsupportedError>()),
                    ),
                  );
                  expect(
                    () => logaddexp2(a, b, where: where),
                    throwsA(
                      anyOf(isA<ArgumentError>(), isA<UnsupportedError>()),
                    ),
                  );
                  expect(
                    () => copysign(a, b, where: where),
                    throwsA(
                      anyOf(isA<ArgumentError>(), isA<UnsupportedError>()),
                    ),
                  );
                  expect(() => atan2(a, b, where: where), throwsArgumentError);
                  expect(() => gcd(a, b, where: where), throwsArgumentError);
                  expect(() => lcm(a, b, where: where), throwsArgumentError);
                  if (!useMask) {
                    expect(() => kron(a, b), throwsArgumentError);
                  }
                  return;
                }

                // 1. add, subtract, multiply, divide (same-dtype pairs)
                verifyGrid(
                  add(a, b, where: where),
                  dtA,
                  (va, vb) => dtA == DType.boolean
                      ? (va as bool) || (vb as bool)
                      : (dtA.isComplex
                            ? _toComplex(va) + _toComplex(vb)
                            : (dtA.isInteger
                                  ? _toInt(va) + _toInt(vb)
                                  : _toDouble(va) + _toDouble(vb))),
                  'add',
                );

                verifyGrid(
                  subtract(a, b, where: where),
                  resolved,
                  (va, vb) => resolved.isComplex
                      ? _toComplex(va) - _toComplex(vb)
                      : (resolved.isInteger
                            ? _toInt(va) - _toInt(vb)
                            : _toDouble(va) - _toDouble(vb)),
                  'subtract',
                );

                verifyGrid(
                  multiply(a, b, where: where),
                  dtA,
                  (va, vb) => dtA == DType.boolean
                      ? (va as bool) && (vb as bool)
                      : (dtA.isComplex
                            ? _toComplex(va) * _toComplex(vb)
                            : (dtA.isInteger
                                  ? _toInt(va) * _toInt(vb)
                                  : _toDouble(va) * _toDouble(vb))),
                  'multiply',
                );

                final divDType =
                    (resolved.isInteger || resolved == DType.boolean)
                    ? DType.float64
                    : resolved;
                verifyGrid(
                  divide(a, b, where: where),
                  divDType,
                  (va, vb) => divDType.isComplex
                      ? _toComplex(va) / _toComplex(vb)
                      : _toDouble(va) / _toDouble(vb),
                  'divide',
                );

                // 2. power (same dtype succeeds)
                verifyGrid(power(a, b, where: where), dtA, (va, vb) {
                  if (dtA == DType.boolean) {
                    final ba = va as bool;
                    final bb = vb as bool;
                    return !bb || ba;
                  }
                  if (dtA.isComplex) {
                    return _complexPow(_toComplex(va), _toComplex(vb));
                  }
                  if (dtA.isFloating) {
                    return math.pow(_toDouble(va), _toDouble(vb)).toDouble();
                  }
                  var res = 1;
                  final base = _toInt(va);
                  final exp = _toInt(vb);
                  for (var k = 0; k < exp; k++) {
                    res *= base;
                  }
                  return res;
                }, 'power');

                // 3. hypot (disallows complex)
                if (dtA.isComplex) {
                  expect(
                    () => hypot(a, b, where: where),
                    throwsUnsupportedError,
                  );
                } else {
                  final hypotDType = resolved == DType.float32
                      ? DType.float32
                      : DType.float64;
                  verifyGrid(hypot(a, b, where: where), hypotDType, (va, vb) {
                    final da = _toDouble(va);
                    final db = _toDouble(vb);
                    return math.sqrt(da * da + db * db);
                  }, 'hypot');
                }

                // 4. floorDivide, remainder, fmod, heaviside, logaddexp, logaddexp2
                if (dtA.isComplex || dtB.isComplex) {
                  expect(
                    () => floorDivide(a, b, where: where),
                    throwsUnsupportedError,
                  );
                  expect(
                    () => remainder(a, b, where: where),
                    throwsUnsupportedError,
                  );
                  expect(
                    () => fmod(a, b, where: where),
                    throwsUnsupportedError,
                  );
                  expect(
                    () => heaviside(a, b, where: where),
                    throwsUnsupportedError,
                  );
                  expect(
                    () => logaddexp(a, b, where: where),
                    throwsUnsupportedError,
                  );
                  expect(
                    () => logaddexp2(a, b, where: where),
                    throwsUnsupportedError,
                  );
                } else {
                  verifyGrid(
                    floorDivide(a, b, where: where),
                    resolved,
                    (va, vb) => resolved.isFloating
                        ? (_toDouble(va) / _toDouble(vb)).floorToDouble()
                        : _scalarFloorDivInt(_toInt(va), _toInt(vb)),
                    'floorDivide',
                  );

                  verifyGrid(
                    remainder(a, b, where: where),
                    resolved,
                    (va, vb) => resolved.isFloating
                        ? _scalarRemainderDouble(_toDouble(va), _toDouble(vb))
                        : _scalarRemainderInt(_toInt(va), _toInt(vb)),
                    'remainder',
                  );

                  verifyGrid(
                    fmod(a, b, where: where),
                    resolved,
                    (va, vb) => resolved.isFloating
                        ? _toDouble(va).remainder(_toDouble(vb))
                        : _toInt(va).remainder(_toInt(vb)),
                    'fmod',
                  );

                  verifyGrid(heaviside(a, b, where: where), resolved, (va, vb) {
                    final dx = _toDouble(va);
                    if (dx < 0) return resolved.isFloating ? 0.0 : 0;
                    if (dx > 0) return resolved.isFloating ? 1.0 : 1;
                    return resolved.isFloating ? _toDouble(vb) : _toInt(vb);
                  }, 'heaviside');

                  final laeDType =
                      (dtA == DType.float32 && dtB == DType.float32)
                      ? DType.float32
                      : DType.float64;
                  verifyGrid(
                    logaddexp(a, b, where: where),
                    laeDType,
                    (va, vb) => _scalarLogAddExp(_toDouble(va), _toDouble(vb)),
                    'logaddexp',
                  );
                  verifyGrid(
                    logaddexp2(a, b, where: where),
                    laeDType,
                    (va, vb) => _scalarLogAddExp2(_toDouble(va), _toDouble(vb)),
                    'logaddexp2',
                  );
                }

                // 5. copysign and atan2 (disallow complex)
                if (dtA.isComplex || dtB.isComplex) {
                  expect(
                    () => copysign(a, b, where: where),
                    throwsUnsupportedError,
                  );
                  expect(
                    () => atan2(a, b, where: where),
                    throwsUnsupportedError,
                  );
                } else {
                  verifyGrid(
                    copysign(a, b, where: where),
                    dtA,
                    (va, vb) => _toDouble(vb) < 0
                        ? -_toDouble(va).abs()
                        : _toDouble(va).abs(),
                    'copysign',
                  );
                  final atan2DType =
                      (dtA == DType.float32 && dtB == DType.float32)
                      ? DType.float32
                      : DType.float64;
                  verifyGrid(
                    atan2(a, b, where: where),
                    atan2DType,
                    (va, vb) => math.atan2(_toDouble(va), _toDouble(vb)),
                    'atan2',
                  );
                }

                // 6. gcd and lcm (only integer dtypes)
                if (dtA.isInteger && dtB.isInteger) {
                  verifyGrid(
                    gcd(a, b, where: where),
                    resolved,
                    (va, vb) => _scalarGcd(_toInt(va), _toInt(vb)),
                    'gcd',
                  );
                  verifyGrid(
                    lcm(a, b, where: where),
                    resolved,
                    (va, vb) => _scalarLcm(_toInt(va), _toInt(vb)),
                    'lcm',
                  );
                } else {
                  expect(() => gcd(a, b, where: where), throwsUnsupportedError);
                  expect(() => lcm(a, b, where: where), throwsUnsupportedError);
                }

                // 7. kron (when not masked)
                if (!useMask) {
                  final kRes = kron(a, b);
                  expect(kRes.dtype, equals(dtA));
                  expect(kRes.shape, equals([4, 4]));
                  for (var r = 0; r < 4; r++) {
                    for (var c = 0; c < 4; c++) {
                      final va = a.getCell([r ~/ 2, c ~/ 2]);
                      final vb = b.getCell([r % 2, c % 2]);
                      final Object expected = _wrapCast(
                        dtA == DType.boolean
                            ? (va as bool) && (vb as bool)
                            : (dtA.isComplex
                                  ? _toComplex(va) * _toComplex(vb)
                                  : (dtA.isInteger
                                        ? _toInt(va) * _toInt(vb)
                                        : _toDouble(va) * _toDouble(vb))),
                        dtA,
                      );
                      _expectCellMatches(
                        kRes.getCell([r, c]),
                        expected,
                        dtA,
                        context: 'kron($dtA, $dtB, $mode) at [$r, $c]',
                      );
                    }
                  }
                  if (dtB.isFloating || dtB.isComplex) {
                    final fracB = dtB.isComplex
                        ? NDArray.fromList(
                            [
                              Complex(0.5, -1.25),
                              Complex(1.5, 0.75),
                              Complex(-0.25, 0.5),
                              Complex(2.25, -0.5),
                            ],
                            shape,
                            dtB as DType<AnySpec>,
                          )
                        : NDArray.fromList(
                            [0.5, 1.5, -0.25, 2.25],
                            shape,
                            dtB as DType<AnySpec>,
                          );
                    final fracRes = kron(a, fracB);
                    for (var r = 0; r < 4; r++) {
                      for (var c = 0; c < 4; c++) {
                        final va = a.getCell([r ~/ 2, c ~/ 2]);
                        final vb = fracB.getCell([r % 2, c % 2]);
                        final Object expected = _wrapCast(
                          resolved.isComplex
                              ? _toComplex(va) * _toComplex(vb)
                              : _toDouble(va) * _toDouble(vb),
                          resolved,
                        );
                        _expectCellMatches(
                          fracRes.getCell([r, c]),
                          expected,
                          resolved,
                          context: 'kronFrac($dtA, $dtB, $mode) at [$r, $c]',
                        );
                      }
                    }
                  }
                }
              });
            }
          }
        }
      },
    );

    test(
      'All 15 DTypes overwrite pre-filled sentinel output buffers (never returning unwritten memory)',
      () {
        for (final dt in DType.values) {
          for (final isStrided in [false, true]) {
            NDArray.scope(() {
              final a = _makeArray(
                dt,
                shape,
                strided: isStrided,
                baseInts: const [4, 2, 3, 1],
              );
              final b = _makeArray(
                dt,
                shape,
                strided: isStrided,
                baseInts: const [2, 3, 2, 2],
              );

              void checkSentinelWritten(
                DType outDType,
                void Function(NDArray<AnySpec> out) runWithOut,
                NDArray Function() runWithoutOut,
                String opName, {
                Object? sentinelOverride,
              }) {
                NDArray? expectedArr;
                try {
                  expectedArr = runWithoutOut();
                } on Object catch (e) {
                  if (e is! UnsupportedError) rethrow;
                  final out = _makeSentinelArray(
                    outDType,
                    shape,
                    sentinelOverride: sentinelOverride,
                  );
                  expect(
                    () => runWithOut(out),
                    throwsUnsupportedError,
                    reason:
                        '$opName($dt, strided=$isStrided) must throw UnsupportedError with out:',
                  );
                  return;
                }

                // Choose a sentinel guaranteed to differ from every expected element.
                final expFirst = expectedArr.getCell([0, 0]);
                final Object sentinel =
                    sentinelOverride ??
                    switch (outDType) {
                      DType.boolean => !(expFirst as bool),
                      DType.complex128 ||
                      DType.complex64 => Complex(77.0, -55.0),
                      DType.float64 ||
                      DType.float32 ||
                      DType.float16 ||
                      DType.bfloat16 => 77.0,
                      _ => 77,
                    };
                final out = _makeSentinelArray(
                  outDType,
                  expectedArr.shape,
                  sentinelOverride: sentinel,
                );
                runWithOut(out);

                for (var i = 0; i < out.size; i++) {
                  final got = out.getCellFlat(i);
                  final exp = expectedArr.getCellFlat(i);
                  if (outDType == DType.boolean) {
                    if (i == 0) {
                      expect(
                        got,
                        isNot(equals(sentinel)),
                        reason:
                            '$opName($dt, strided=$isStrided) left sentinel at 0',
                      );
                    }
                  } else {
                    expect(
                      got,
                      isNot(equals(sentinel)),
                      reason:
                          '$opName($dt, strided=$isStrided) left sentinel at flat index $i',
                    );
                  }
                  _expectCellMatches(
                    got,
                    exp,
                    outDType,
                    context: '$opName($dt, strided=$isStrided) flat index $i',
                  );
                }
              }

              // Same-dtype unary ops
              checkSentinelWritten(
                dt,
                (out) => square(a, out: out),
                () => square(a),
                'square',
              );
              checkSentinelWritten(
                dt,
                (out) => sign(a, out: out),
                () => sign(a),
                'sign',
              );
              checkSentinelWritten(
                dt,
                (out) => negative(a, out: out),
                () => negative(a),
                'negative',
              );
              checkSentinelWritten(
                dt,
                (out) => positive(a, out: out),
                () => positive(a),
                'positive',
              );
              checkSentinelWritten(
                dt,
                (out) => reciprocal(a, out: out),
                () => reciprocal(a),
                'reciprocal',
              );
              checkSentinelWritten(
                dt,
                (out) => ceil(a, out: out),
                () => ceil(a),
                'ceil',
              );
              checkSentinelWritten(
                dt,
                (out) => floor(a, out: out),
                () => floor(a),
                'floor',
              );
              checkSentinelWritten(
                dt,
                (out) => round(a, out: out),
                () => round(a),
                'round',
              );

              // Float-promoting unary ops
              final floatOrCpx = dt.isComplex
                  ? dt
                  : (dt == DType.float32 ? DType.float32 : DType.float64);
              final absDType = switch (dt) {
                DType.complex64 => DType.float32,
                DType.complex128 => DType.float64,
                _ => dt,
              };
              checkSentinelWritten(
                absDType,
                (out) => abs(a, out: out),
                () => abs(a),
                'abs',
              );
              checkSentinelWritten(
                floatOrCpx,
                (out) => sqrt(a, out: out),
                () => sqrt(a),
                'sqrt',
              );
              checkSentinelWritten(
                floatOrCpx,
                (out) => expm1(a, out: out),
                () => expm1(a),
                'expm1',
              );
              checkSentinelWritten(
                floatOrCpx,
                (out) => log1p(a, out: out),
                () => log1p(a),
                'log1p',
              );
              checkSentinelWritten(
                floatOrCpx,
                (out) => exp(a, out: out),
                () => exp(a),
                'exp',
              );
              checkSentinelWritten(
                floatOrCpx,
                (out) => log(a, out: out),
                () => log(a),
                'log',
              );
              checkSentinelWritten(
                floatOrCpx,
                (out) => sin(a, out: out),
                () => sin(a),
                'sin',
              );
              checkSentinelWritten(
                floatOrCpx,
                (out) => cos(a, out: out),
                () => cos(a),
                'cos',
              );

              final rintDType = dt == DType.float32
                  ? DType.float32
                  : DType.float64;
              checkSentinelWritten(
                rintDType,
                (out) => rint(a, out: out),
                () => rint(a),
                'rint',
              );
              checkSentinelWritten(
                rintDType,
                (out) => trunc(a, out: out),
                () => trunc(a),
                'trunc',
              );

              // Binary ops with out:
              final resolved = resolveDType(dt, dt);
              checkSentinelWritten(
                dt,
                (out) => power(a, b, out: out),
                () => power(a, b),
                'power',
              );
              checkSentinelWritten(
                dt,
                (out) => add(a, b, out: out),
                () => add(a, b),
                'add',
              );
              checkSentinelWritten(
                resolved,
                (out) => subtract(a, b, out: out),
                () => subtract(a, b),
                'subtract',
              );
              checkSentinelWritten(
                dt,
                (out) => multiply(a, b, out: out),
                () => multiply(a, b),
                'multiply',
              );
              final divDt = (resolved.isInteger || resolved == DType.boolean)
                  ? DType.float64
                  : resolved;
              checkSentinelWritten(
                divDt,
                (out) => divide(a, b, out: out),
                () => divide(a, b),
                'divide',
              );
              checkSentinelWritten(
                resolved,
                (out) => floorDivide(a, b, out: out),
                () => floorDivide(a, b),
                'floorDivide',
              );
              checkSentinelWritten(
                resolved,
                (out) => remainder(a, b, out: out),
                () => remainder(a, b),
                'remainder',
              );
              checkSentinelWritten(
                resolved,
                (out) => fmod(a, b, out: out),
                () => fmod(a, b),
                'fmod',
              );
              checkSentinelWritten(
                resolved,
                (out) => gcd(a, b, out: out),
                () => gcd(a, b),
                'gcd',
              );
              checkSentinelWritten(
                resolved,
                (out) => lcm(a, b, out: out),
                () => lcm(a, b),
                'lcm',
              );
              checkSentinelWritten(
                resolved,
                (out) => heaviside(a, b, out: out),
                () => heaviside(a, b),
                'heaviside',
              );
              checkSentinelWritten(
                dt,
                (out) => kron(a, b, out: out),
                () => kron(a, b),
                'kron',
              );
              if (dt == DType.boolean) {
                // Also verify kron(bool, bool, out: boolOut) overwrites sentinel.
                final boolOut = _makeSentinelArray(DType.boolean, [
                  4,
                  4,
                ], sentinelOverride: false);
                kron(a, b, out: boolOut);
                expect(boolOut.getCell([0, 0]), isTrue);
              }
            });
          }
        }
      },
    );
  });
}
