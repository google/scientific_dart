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

import 'dart:ffi' as ffi;
import 'dart:io';

import 'package:ndarray/ndarray.dart';
import 'package:ndarray/src/ndarray_bindings.dart';
import 'package:test/test.dart';

enum _NonFiniteKind { singleNaN, posInf, negInf, allNaN, singular }

final class _CanaryBuffer<T extends DTypeTag> {
  _CanaryBuffer(this.backing, this.view, this.guardElements, this.dtype);

  final NDArray<T> backing;
  final NDArray<T> view;
  final int guardElements;
  final DType<T> dtype;

  static const double _sentinelF64 = -987654.321;
  static const double _sentinelF32 = -12345.5;

  static _CanaryBuffer<T> create<T extends DTypeTag>(
    List<int> shape,
    DType<T> dtype, {
    int guardElements = 16,
  }) {
    var innerSize = 1;
    for (final dim in shape) {
      innerSize *= dim;
    }
    final totalElements = innerSize + 2 * guardElements;
    final backing = NDArray<T>.create([totalElements], dtype);
    _fillRangeWithSentinel(backing, 0, totalElements);

    final sliced = backing.slice([
      Slice(start: guardElements, stop: guardElements + innerSize),
    ]);
    final view = sliced.reshape(shape);
    sliced.dispose();
    return _CanaryBuffer<T>(backing, view, guardElements, dtype);
  }

  static void _fillRangeWithSentinel(NDArray arr, int start, int end) {
    switch (arr.dtype) {
      case DType.float64:
        final ptr = arr.pointer.cast<ffi.Double>();
        for (var i = start; i < end; i++) {
          ptr[i] = _sentinelF64;
        }
      case DType.float32:
        final ptr = arr.pointer.cast<ffi.Float>();
        for (var i = start; i < end; i++) {
          ptr[i] = _sentinelF32;
        }
      case DType.complex128:
        final ptr = arr.pointer.cast<ffi.Double>();
        for (var i = start; i < end; i++) {
          ptr[2 * i] = _sentinelF64;
          ptr[2 * i + 1] = -_sentinelF64;
        }
      case DType.complex64:
        final ptr = arr.pointer.cast<ffi.Float>();
        for (var i = start; i < end; i++) {
          ptr[2 * i] = _sentinelF32;
          ptr[2 * i + 1] = -_sentinelF32;
        }
      default:
        throw UnsupportedError('Unsupported canary dtype: ${arr.dtype}');
    }
  }

  void verifySentinelsIntact(String context) {
    final total = backing.size;
    final innerSize = view.size;
    final tailStart = guardElements + innerSize;

    void checkRange(int start, int end, String region) {
      switch (dtype) {
        case DType.float64:
          final ptr = backing.pointer.cast<ffi.Double>();
          for (var i = start; i < end; i++) {
            expect(
              ptr[i],
              equals(_sentinelF64),
              reason: '$context corrupted $region canary at index $i',
            );
          }
        case DType.float32:
          final ptr = backing.pointer.cast<ffi.Float>();
          for (var i = start; i < end; i++) {
            expect(
              ptr[i],
              equals(_sentinelF32),
              reason: '$context corrupted $region canary at index $i',
            );
          }
        case DType.complex128:
          final ptr = backing.pointer.cast<ffi.Double>();
          for (var i = start; i < end; i++) {
            expect(
              ptr[2 * i],
              equals(_sentinelF64),
              reason: '$context corrupted $region canary real at index $i',
            );
            expect(
              ptr[2 * i + 1],
              equals(-_sentinelF64),
              reason: '$context corrupted $region canary imag at index $i',
            );
          }
        case DType.complex64:
          final ptr = backing.pointer.cast<ffi.Float>();
          for (var i = start; i < end; i++) {
            expect(
              ptr[2 * i],
              equals(_sentinelF32),
              reason: '$context corrupted $region canary real at index $i',
            );
            expect(
              ptr[2 * i + 1],
              equals(-_sentinelF32),
              reason: '$context corrupted $region canary imag at index $i',
            );
          }
        default:
          throw UnsupportedError('Unsupported canary dtype: $dtype');
      }
    }

    checkRange(0, guardElements, 'leading');
    checkRange(tailStart, total, 'trailing');
  }

  void dispose() {
    view.dispose();
    backing.dispose();
  }
}

NDArray<AnySpec> _buildMatrix(
  int rows,
  int cols,
  DType<AnySpec> dtype,
  _NonFiniteKind kind,
) {
  final size = rows * cols;
  final values = <double>[];
  for (var r = 0; r < rows; r++) {
    for (var c = 0; c < cols; c++) {
      if (kind == _NonFiniteKind.allNaN) {
        values.add(double.nan);
      } else if (kind == _NonFiniteKind.singular) {
        // Row 0 and Col 0 are zero so matrix is exactly singular in LU, Cholesky, and SVD.
        values.add((r == 0 || c == 0) ? 0.0 : (r == c ? (r + 2.0) : 0.25));
      } else {
        // Strictly diagonally dominant symmetric positive-definite base matrix.
        values.add(r == c ? (rows + r + 2.0) : 0.25 / (1 + (r - c).abs()));
      }
    }
  }

  // Place the non-finite entry at (1, 0) (lower triangle) when rows > 1 so
  // lower-triangular routines (cholesky, eigh, eigvalsh) also inspect it.
  final targetIndex = rows > 1 ? cols : 0;
  switch (kind) {
    case _NonFiniteKind.singleNaN:
      values[targetIndex] = double.nan;
    case _NonFiniteKind.posInf:
      values[targetIndex] = double.infinity;
    case _NonFiniteKind.negInf:
      values[targetIndex] = double.negativeInfinity;
    case _NonFiniteKind.allNaN:
    case _NonFiniteKind.singular:
      break;
  }

  switch (dtype) {
    case DType.float64:
      return NDArray<Float64>.fromList(values, [rows, cols], DType.float64);
    case DType.float32:
      return NDArray<Float32>.fromList(values, [rows, cols], DType.float32);
    case DType.complex128:
      final cpx = <Complex>[
        for (var i = 0; i < size; i++) Complex(values[i], 0.0),
      ];
      return NDArray<Complex128>.fromList(cpx, [rows, cols], DType.complex128);
    case DType.complex64:
      final cpx = <Complex>[
        for (var i = 0; i < size; i++) Complex(values[i], 0.0),
      ];
      return NDArray<Complex64>.fromList(cpx, [rows, cols], DType.complex64);
    default:
      throw UnsupportedError('Unsupported dtype: $dtype');
  }
}

NDArray<AnySpec> _buildVector(
  int length,
  DType<AnySpec> dtype, {
  _NonFiniteKind? kind,
}) {
  final values = List<double>.generate(length, (i) => i + 1.0);
  if (kind == _NonFiniteKind.singleNaN) {
    values[0] = double.nan;
  } else if (kind == _NonFiniteKind.posInf) {
    values[0] = double.infinity;
  } else if (kind == _NonFiniteKind.negInf) {
    values[0] = double.negativeInfinity;
  } else if (kind == _NonFiniteKind.allNaN) {
    for (var i = 0; i < length; i++) {
      values[i] = double.nan;
    }
  }
  switch (dtype) {
    case DType.float64:
      return NDArray<Float64>.fromList(values, [length], DType.float64);
    case DType.float32:
      return NDArray<Float32>.fromList(values, [length], DType.float32);
    case DType.complex128:
      return NDArray<Complex128>.fromList(
        [for (final v in values) Complex(v, 0.0)],
        [length],
        DType.complex128,
      );
    case DType.complex64:
      return NDArray<Complex64>.fromList(
        [for (final v in values) Complex(v, 0.0)],
        [length],
        DType.complex64,
      );
    default:
      throw UnsupportedError('Unsupported dtype: $dtype');
  }
}

bool _allElementsAreNaN(NDArray arr) {
  final contig = arr.isContiguous ? arr : arr.copy();
  try {
    switch (contig.dtype) {
      case DType.float64:
        final ptr = contig.pointer.cast<ffi.Double>();
        for (var i = 0; i < contig.size; i++) {
          if (!ptr[i].isNaN) return false;
        }
        return true;
      case DType.float32:
        final ptr = contig.pointer.cast<ffi.Float>();
        for (var i = 0; i < contig.size; i++) {
          if (!ptr[i].isNaN) return false;
        }
        return true;
      case DType.complex128:
        final ptr = contig.pointer.cast<ffi.Double>();
        for (var i = 0; i < contig.size * 2; i++) {
          if (!ptr[i].isNaN) return false;
        }
        return true;
      case DType.complex64:
        final ptr = contig.pointer.cast<ffi.Float>();
        for (var i = 0; i < contig.size * 2; i++) {
          if (!ptr[i].isNaN) return false;
        }
        return true;
      default:
        throw UnsupportedError('Unsupported dtype: ${contig.dtype}');
    }
  } finally {
    if (!identical(contig, arr)) contig.dispose();
  }
}

bool _anyElementIsNaN(NDArray arr) {
  final contig = arr.isContiguous ? arr : arr.copy();
  try {
    switch (contig.dtype) {
      case DType.float64:
        final ptr = contig.pointer.cast<ffi.Double>();
        for (var i = 0; i < contig.size; i++) {
          if (ptr[i].isNaN) return true;
        }
        return false;
      case DType.float32:
        final ptr = contig.pointer.cast<ffi.Float>();
        for (var i = 0; i < contig.size; i++) {
          if (ptr[i].isNaN) return true;
        }
        return false;
      case DType.complex128:
        final ptr = contig.pointer.cast<ffi.Double>();
        for (var i = 0; i < contig.size * 2; i++) {
          if (ptr[i].isNaN) return true;
        }
        return false;
      case DType.complex64:
        final ptr = contig.pointer.cast<ffi.Float>();
        for (var i = 0; i < contig.size * 2; i++) {
          if (ptr[i].isNaN) return true;
        }
        return false;
      default:
        throw UnsupportedError('Unsupported dtype: ${contig.dtype}');
    }
  } finally {
    if (!identical(contig, arr)) contig.dispose();
  }
}

DType<DTypeTag> _realDTypeFor(DType dtype) =>
    (dtype == DType.float32 || dtype == DType.complex64)
    ? DType.float32
    : DType.float64;

DType<DTypeTag> _complexDTypeFor(DType dtype) =>
    (dtype == DType.float32 || dtype == DType.complex64)
    ? DType.complex64
    : DType.complex128;

void _withNoLeaks(void Function() body) {
  final arenaBefore = ScratchArena.marker;
  NDArray.clearTrackedAllocations();
  try {
    body();
  } finally {
    expect(ScratchArena.marker, equals(arenaBefore));
    expect(NDArray.trackedAllocations, isEmpty);
  }
}

void main() {
  group('H1 Native assemble_eigenvectors_* bounds safety', () {
    for (final n in [1, 2, 3, 4, 5]) {
      test(
        'assemble_eigenvectors_double does not OOB when wi is NaN (n=$n)',
        () {
          final canaryW = _CanaryBuffer.create([n], DType.complex128);
          final canaryVR = _CanaryBuffer.create([n, n], DType.complex128);
          final wr = NDArray<Float64>.create([n], DType.float64);
          final wi = NDArray<Float64>.create([n], DType.float64);
          final vrReal = NDArray<Float64>.create([n, n], DType.float64);
          try {
            for (var i = 0; i < n; i++) {
              wr.pointer.cast<ffi.Double>()[i] = double.nan;
              wi.pointer.cast<ffi.Double>()[i] = double.nan;
            }
            for (var i = 0; i < n * n; i++) {
              vrReal.pointer.cast<ffi.Double>()[i] = 1.0;
            }
            assemble_eigenvectors_double(
              canaryW.view.pointer.cast<cpx_t>(),
              1,
              canaryVR.view.pointer.cast<cpx_t>(),
              n,
              1,
              wr.pointer.cast<ffi.Double>(),
              wi.pointer.cast<ffi.Double>(),
              vrReal.pointer.cast<ffi.Double>(),
              n,
            );
            canaryW.verifySentinelsIntact(
              'assemble_eigenvectors_double w (n=$n)',
            );
            canaryVR.verifySentinelsIntact(
              'assemble_eigenvectors_double vr (n=$n)',
            );
            expect(_anyElementIsNaN(canaryW.view), isTrue);
            expect(_anyElementIsNaN(canaryVR.view), isTrue);
          } finally {
            wr.dispose();
            wi.dispose();
            vrReal.dispose();
            canaryW.dispose();
            canaryVR.dispose();
          }
        },
      );

      test(
        'assemble_eigenvectors_float does not OOB when wi is NaN (n=$n)',
        () {
          final canaryW = _CanaryBuffer.create([n], DType.complex64);
          final canaryVR = _CanaryBuffer.create([n, n], DType.complex64);
          final wr = NDArray<Float32>.create([n], DType.float32);
          final wi = NDArray<Float32>.create([n], DType.float32);
          final vrReal = NDArray<Float32>.create([n, n], DType.float32);
          try {
            for (var i = 0; i < n; i++) {
              wr.pointer.cast<ffi.Float>()[i] = double.nan;
              wi.pointer.cast<ffi.Float>()[i] = double.nan;
            }
            for (var i = 0; i < n * n; i++) {
              vrReal.pointer.cast<ffi.Float>()[i] = 1.0;
            }
            assemble_eigenvectors_float(
              canaryW.view.pointer.cast<cpx_f_t>(),
              1,
              canaryVR.view.pointer.cast<cpx_f_t>(),
              n,
              1,
              wr.pointer.cast<ffi.Float>(),
              wi.pointer.cast<ffi.Float>(),
              vrReal.pointer.cast<ffi.Float>(),
              n,
            );
            canaryW.verifySentinelsIntact(
              'assemble_eigenvectors_float w (n=$n)',
            );
            canaryVR.verifySentinelsIntact(
              'assemble_eigenvectors_float vr (n=$n)',
            );
            expect(_anyElementIsNaN(canaryW.view), isTrue);
            expect(_anyElementIsNaN(canaryVR.view), isTrue);
          } finally {
            wr.dispose();
            wi.dispose();
            vrReal.dispose();
            canaryW.dispose();
            canaryVR.dispose();
          }
        },
      );
    }
  });

  group('Linalg non-finite & singular cross-cutting contracts', () {
    const dtypes = <DType<AnySpec>>[
      DType.float64,
      DType.float32,
      DType.complex128,
      DType.complex64,
    ];
    const sizes = <int>[3, 4];
    const kinds = <_NonFiniteKind>[
      _NonFiniteKind.singleNaN,
      _NonFiniteKind.posInf,
      _NonFiniteKind.negInf,
      _NonFiniteKind.allNaN,
      _NonFiniteKind.singular,
    ];

    for (final dtype in dtypes) {
      for (final n in sizes) {
        for (final kind in kinds) {
          final label = 'dtype=$dtype n=$n kind=${kind.name}';

          test('solve ($label)', () {
            _withNoLeaks(() {
              final a = _buildMatrix(n, n, dtype, kind);
              final b = _buildVector(n, dtype);
              final canaryOut = _CanaryBuffer.create([n], dtype);
              try {
                if (kind == _NonFiniteKind.singleNaN ||
                    kind == _NonFiniteKind.allNaN) {
                  final res = solve(a, b, out: canaryOut.view);
                  expect(_allElementsAreNaN(res), isTrue);
                } else if (kind == _NonFiniteKind.singular) {
                  expect(
                    () => solve(a, b, out: canaryOut.view),
                    throwsA(isA<SingularMatrixException>()),
                  );
                } else {
                  // ±Inf: either throws SingularMatrixException/LinAlgException or returns NaN/finite, never Error.
                  try {
                    final res = solve(a, b, out: canaryOut.view);
                    expect(res.shape, equals([n]));
                  } on LinAlgException {
                    // Allowed by contract.
                  }
                }
                canaryOut.verifySentinelsIntact('solve ($label)');
              } finally {
                a.dispose();
                b.dispose();
                canaryOut.dispose();
              }
            });
          });

          test('inv ($label)', () {
            _withNoLeaks(() {
              final a = _buildMatrix(n, n, dtype, kind);
              final canaryOut = _CanaryBuffer.create([n, n], dtype);
              try {
                if (kind == _NonFiniteKind.singleNaN ||
                    kind == _NonFiniteKind.allNaN) {
                  final res = inv(a, out: canaryOut.view);
                  expect(_allElementsAreNaN(res), isTrue);
                } else if (kind == _NonFiniteKind.singular) {
                  expect(
                    () => inv(a, out: canaryOut.view),
                    throwsA(isA<SingularMatrixException>()),
                  );
                } else {
                  try {
                    final res = inv(a, out: canaryOut.view);
                    expect(res.shape, equals([n, n]));
                  } on LinAlgException {
                    // Allowed if LU factor becomes singular.
                  }
                }
                canaryOut.verifySentinelsIntact('inv ($label)');
              } finally {
                a.dispose();
                canaryOut.dispose();
              }
            });
          });

          test('det ($label)', () {
            _withNoLeaks(() {
              final a = _buildMatrix(n, n, dtype, kind);
              final canaryOut = _CanaryBuffer.create(<int>[], dtype);
              try {
                final res = det(a, out: canaryOut.view);
                canaryOut.verifySentinelsIntact('det ($label)');
                if (kind == _NonFiniteKind.singleNaN ||
                    kind == _NonFiniteKind.allNaN) {
                  expect(_anyElementIsNaN(res), isTrue);
                } else if (kind == _NonFiniteKind.singular) {
                  if (dtype.isComplex) {
                    final c = res.scalar as Complex;
                    expect(c.real, equals(0.0));
                    expect(c.imag, equals(0.0));
                  } else {
                    expect(res.scalar, equals(0.0));
                  }
                }
              } finally {
                a.dispose();
                canaryOut.dispose();
              }
            });
          });

          test('slogdet ($label)', () {
            _withNoLeaks(() {
              final a = _buildMatrix(n, n, dtype, kind);
              final realDType = _realDTypeFor(dtype);
              final canarySign = _CanaryBuffer.create(<int>[], dtype);
              final canaryLogdet = _CanaryBuffer.create(<int>[], realDType);
              try {
                final res = slogdet(
                  a,
                  outSign: canarySign.view,
                  outLogdet: canaryLogdet.view,
                );
                canarySign.verifySentinelsIntact('slogdet sign ($label)');
                canaryLogdet.verifySentinelsIntact('slogdet logdet ($label)');
                if (kind == _NonFiniteKind.singleNaN ||
                    kind == _NonFiniteKind.allNaN) {
                  expect(_anyElementIsNaN(res.sign), isTrue);
                  expect(_anyElementIsNaN(res.logabsdet), isTrue);
                } else if (kind == _NonFiniteKind.singular) {
                  expect(res.logabsdet.scalar, equals(double.negativeInfinity));
                }
              } finally {
                a.dispose();
                canarySign.dispose();
                canaryLogdet.dispose();
              }
            });
          });

          test('eig ($label)', () {
            _withNoLeaks(() {
              final a = _buildMatrix(n, n, dtype, kind);
              final compDType = _complexDTypeFor(dtype);
              final canaryW = _CanaryBuffer.create([n], compDType);
              final canaryVR = _CanaryBuffer.create([n, n], compDType);
              try {
                if (kind == _NonFiniteKind.singular) {
                  final res = eig(
                    a,
                    out: (
                      eigenvalues: canaryW.view,
                      eigenvectors: canaryVR.view,
                    ),
                  );
                  expect(res.eigenvalues.shape, equals([n]));
                  expect(res.eigenvectors.shape, equals([n, n]));
                } else {
                  expect(
                    () => eig(
                      a,
                      out: (
                        eigenvalues: canaryW.view,
                        eigenvectors: canaryVR.view,
                      ),
                    ),
                    throwsA(isA<LinAlgException>()),
                  );
                }
                canaryW.verifySentinelsIntact('eig w ($label)');
                canaryVR.verifySentinelsIntact('eig vr ($label)');
              } finally {
                a.dispose();
                canaryW.dispose();
                canaryVR.dispose();
              }
            });
          });

          test('eigvals ($label)', () {
            _withNoLeaks(() {
              final a = _buildMatrix(n, n, dtype, kind);
              final compDType = _complexDTypeFor(dtype);
              final canaryW = _CanaryBuffer.create([n], compDType);
              try {
                if (kind == _NonFiniteKind.singular) {
                  final res = eigvals(a, out: canaryW.view);
                  expect(res.shape, equals([n]));
                } else {
                  expect(
                    () => eigvals(a, out: canaryW.view),
                    throwsA(isA<LinAlgException>()),
                  );
                }
                canaryW.verifySentinelsIntact('eigvals ($label)');
              } finally {
                a.dispose();
                canaryW.dispose();
              }
            });
          });

          test('eigh ($label)', () {
            _withNoLeaks(() {
              final a = _buildMatrix(n, n, dtype, kind);
              final realDType = _realDTypeFor(dtype);
              final canaryW = _CanaryBuffer.create([n], realDType);
              final canaryV = _CanaryBuffer.create([n, n], dtype);
              try {
                if (kind == _NonFiniteKind.singular) {
                  final res = eigh(
                    a,
                    outEigenvalues: canaryW.view,
                    outEigenvectors: canaryV.view,
                  );
                  expect(res.eigenvalues.shape, equals([n]));
                  expect(res.eigenvectors.shape, equals([n, n]));
                } else {
                  expect(
                    () => eigh(
                      a,
                      outEigenvalues: canaryW.view,
                      outEigenvectors: canaryV.view,
                    ),
                    throwsA(isA<LinAlgException>()),
                  );
                }
                canaryW.verifySentinelsIntact('eigh w ($label)');
                canaryV.verifySentinelsIntact('eigh v ($label)');
              } finally {
                a.dispose();
                canaryW.dispose();
                canaryV.dispose();
              }
            });
          });

          test('eigvalsh ($label)', () {
            _withNoLeaks(() {
              final a = _buildMatrix(n, n, dtype, kind);
              final realDType = _realDTypeFor(dtype);
              final canaryW = _CanaryBuffer.create([n], realDType);
              try {
                if (kind == _NonFiniteKind.singular) {
                  final res = eigvalsh(a, out: canaryW.view);
                  expect(res.shape, equals([n]));
                } else {
                  expect(
                    () => eigvalsh(a, out: canaryW.view),
                    throwsA(isA<LinAlgException>()),
                  );
                }
                canaryW.verifySentinelsIntact('eigvalsh ($label)');
              } finally {
                a.dispose();
                canaryW.dispose();
              }
            });
          });

          test('qr ($label)', () {
            _withNoLeaks(() {
              final a = _buildMatrix(n, n, dtype, kind);
              final canaryQ = _CanaryBuffer.create([n, n], dtype);
              final canaryR = _CanaryBuffer.create([n, n], dtype);
              try {
                final res = qr(a, out: (q: canaryQ.view, r: canaryR.view));
                canaryQ.verifySentinelsIntact('qr q ($label)');
                canaryR.verifySentinelsIntact('qr r ($label)');
                if (kind != _NonFiniteKind.singular) {
                  expect(_allElementsAreNaN(res.q), isTrue);
                  expect(_anyElementIsNaN(res.r), isTrue);
                }
              } finally {
                a.dispose();
                canaryQ.dispose();
                canaryR.dispose();
              }
            });
          });

          test('svd ($label)', () {
            _withNoLeaks(() {
              final a = _buildMatrix(n, n, dtype, kind);
              final realDType = _realDTypeFor(dtype);
              final canaryU = _CanaryBuffer.create([n, n], dtype);
              final canaryS = _CanaryBuffer.create([n], realDType);
              final canaryVh = _CanaryBuffer.create([n, n], dtype);
              try {
                if (kind == _NonFiniteKind.singular) {
                  final res = svd(
                    a,
                    out: (u: canaryU.view, s: canaryS.view, vh: canaryVh.view),
                  );
                  expect(res.s.shape, equals([n]));
                } else {
                  expect(
                    () => svd(
                      a,
                      out: (
                        u: canaryU.view,
                        s: canaryS.view,
                        vh: canaryVh.view,
                      ),
                    ),
                    throwsA(isA<LinAlgException>()),
                  );
                }
                canaryU.verifySentinelsIntact('svd u ($label)');
                canaryS.verifySentinelsIntact('svd s ($label)');
                canaryVh.verifySentinelsIntact('svd vh ($label)');
              } finally {
                a.dispose();
                canaryU.dispose();
                canaryS.dispose();
                canaryVh.dispose();
              }
            });
          });

          test('cholesky ($label)', () {
            _withNoLeaks(() {
              final a = _buildMatrix(n, n, dtype, kind);
              final canaryOut = _CanaryBuffer.create([n, n], dtype);
              try {
                if (kind == _NonFiniteKind.singleNaN ||
                    kind == _NonFiniteKind.allNaN ||
                    kind == _NonFiniteKind.singular ||
                    kind == _NonFiniteKind.negInf) {
                  expect(
                    () => cholesky(a, out: canaryOut.view),
                    throwsA(isA<NonPositiveDefiniteException>()),
                  );
                } else {
                  // posInf in row 0 col 1 (upper triangle) is ignored by lower Cholesky,
                  // or if in lower triangle may either succeed or throw NonPositiveDefiniteException.
                  try {
                    final res = cholesky(a, out: canaryOut.view);
                    expect(res.shape, equals([n, n]));
                  } on LinAlgException {
                    // Allowed.
                  }
                }
                canaryOut.verifySentinelsIntact('cholesky ($label)');
              } finally {
                a.dispose();
                canaryOut.dispose();
              }
            });
          });

          test('lstsq ($label)', () {
            _withNoLeaks(() {
              final a = _buildMatrix(n, n, dtype, kind);
              final b = _buildVector(n, dtype);
              final canaryOut = _CanaryBuffer.create([n], dtype);
              try {
                if (kind == _NonFiniteKind.singular) {
                  final res = lstsq(a, b, out: canaryOut.view);
                  expect(res.x.shape, equals([n]));
                  res.residuals.dispose();
                  res.s.dispose();
                } else {
                  expect(
                    () => lstsq(a, b, out: canaryOut.view),
                    throwsA(isA<LinAlgException>()),
                  );
                }
                canaryOut.verifySentinelsIntact('lstsq ($label)');
              } finally {
                a.dispose();
                b.dispose();
                canaryOut.dispose();
              }
            });
          });

          test('pinv ($label)', () {
            _withNoLeaks(() {
              final a = _buildMatrix(n, n, dtype, kind);
              final canaryOut = _CanaryBuffer.create([n, n], dtype);
              try {
                if (kind == _NonFiniteKind.singular) {
                  final res = pinv(a, out: canaryOut.view);
                  expect(res.shape, equals([n, n]));
                } else {
                  expect(
                    () => pinv(a, out: canaryOut.view),
                    throwsA(isA<LinAlgException>()),
                  );
                }
                canaryOut.verifySentinelsIntact('pinv ($label)');
              } finally {
                a.dispose();
                canaryOut.dispose();
              }
            });
          });

          test('matrix_power n=-1 ($label)', () {
            _withNoLeaks(() {
              final a = _buildMatrix(n, n, dtype, kind);
              final canaryOut = _CanaryBuffer.create([n, n], dtype);
              try {
                if (kind == _NonFiniteKind.singleNaN ||
                    kind == _NonFiniteKind.allNaN) {
                  final res = matrix_power(a, -1, out: canaryOut.view);
                  expect(_allElementsAreNaN(res), isTrue);
                } else if (kind == _NonFiniteKind.singular) {
                  expect(
                    () => matrix_power(a, -1, out: canaryOut.view),
                    throwsA(isA<SingularMatrixException>()),
                  );
                } else {
                  try {
                    final res = matrix_power(a, -1, out: canaryOut.view);
                    expect(res.shape, equals([n, n]));
                  } on LinAlgException {
                    // Allowed.
                  }
                }
                canaryOut.verifySentinelsIntact('matrix_power ($label)');
              } finally {
                a.dispose();
                canaryOut.dispose();
              }
            });
          });

          test('cond ($label)', () {
            _withNoLeaks(() {
              final a = _buildMatrix(n, n, dtype, kind);
              final realDType = _realDTypeFor(dtype);
              final canaryOut = _CanaryBuffer.create(<int>[], realDType);
              try {
                for (final p in <Object?>[
                  null,
                  2,
                  -2,
                  1,
                  -1,
                  double.infinity,
                  double.negativeInfinity,
                  NormKind.frobenius,
                ]) {
                  final isSvdNorm = p == null || p == 2 || p == -2;
                  if (kind == _NonFiniteKind.singleNaN ||
                      kind == _NonFiniteKind.allNaN) {
                    if (isSvdNorm) {
                      expect(
                        () => cond(a, p: p, out: canaryOut.view),
                        throwsA(isA<LinAlgException>()),
                      );
                    } else {
                      final res = cond(a, p: p, out: canaryOut.view);
                      expect((res.scalar as double).isNaN, isTrue);
                    }
                  } else if (kind == _NonFiniteKind.posInf ||
                      kind == _NonFiniteKind.negInf) {
                    final res = cond(a, p: p, out: canaryOut.view);
                    final v = res.scalar as double;
                    if (p == -1 || p == double.negativeInfinity) {
                      // min row/col sum can be finite while inv has 0 -> 0.0 or inf.
                      expect(v.isNaN, isFalse);
                    } else {
                      expect(v, equals(double.infinity));
                    }
                  } else if (kind == _NonFiniteKind.singular) {
                    final res = cond(a, p: p, out: canaryOut.view);
                    final v = res.scalar as double;
                    if (p == -2) {
                      expect(v, equals(0.0));
                    } else {
                      expect(v, equals(double.infinity));
                    }
                  }
                  canaryOut.verifySentinelsIntact('cond p=$p ($label)');
                }
              } finally {
                a.dispose();
                canaryOut.dispose();
              }
            });
          });

          test('norm SVD-backed orders ($label)', () {
            _withNoLeaks(() {
              final a = _buildMatrix(n, n, dtype, kind);
              final realDType = _realDTypeFor(dtype);
              final canaryOut = _CanaryBuffer.create(<int>[], realDType);
              try {
                for (final ord in <Object>[2, -2, NormKind.nuclear]) {
                  if (kind == _NonFiniteKind.singleNaN ||
                      kind == _NonFiniteKind.allNaN) {
                    expect(
                      () => norm(a, ord: ord, out: canaryOut.view),
                      throwsA(isA<LinAlgException>()),
                    );
                  } else if (kind == _NonFiniteKind.posInf ||
                      kind == _NonFiniteKind.negInf) {
                    final res = norm(a, ord: ord, out: canaryOut.view);
                    expect((res.scalar as double).isNaN, isTrue);
                  } else {
                    final res = norm(a, ord: ord, out: canaryOut.view);
                    expect((res.scalar as double).isFinite, isTrue);
                  }
                  canaryOut.verifySentinelsIntact('norm ord=$ord ($label)');
                }
              } finally {
                a.dispose();
                canaryOut.dispose();
              }
            });
          });

          test('schur ($label)', () {
            _withNoLeaks(() {
              final a = _buildMatrix(n, n, dtype, kind);
              final canaryT = _CanaryBuffer.create([n, n], dtype);
              final canaryZ = _CanaryBuffer.create([n, n], dtype);
              try {
                if (kind == _NonFiniteKind.singular) {
                  final res = schur(a, outT: canaryT.view, outZ: canaryZ.view);
                  expect(res.t.shape, equals([n, n]));
                  expect(res.z.shape, equals([n, n]));
                } else {
                  expect(
                    () => schur(a, outT: canaryT.view, outZ: canaryZ.view),
                    throwsA(isA<LinAlgException>()),
                  );
                }
                canaryT.verifySentinelsIntact('schur t ($label)');
                canaryZ.verifySentinelsIntact('schur z ($label)');
              } finally {
                a.dispose();
                canaryT.dispose();
                canaryZ.dispose();
              }
            });
          });

          test('hessenberg ($label)', () {
            _withNoLeaks(() {
              final a = _buildMatrix(n, n, dtype, kind);
              final canaryH = _CanaryBuffer.create([n, n], dtype);
              final canaryQ = _CanaryBuffer.create([n, n], dtype);
              try {
                if (kind == _NonFiniteKind.singular) {
                  final res = hessenberg(
                    a,
                    outH: canaryH.view,
                    outQ: canaryQ.view,
                  );
                  expect(res.h.shape, equals([n, n]));
                  expect(res.q.shape, equals([n, n]));
                } else {
                  expect(
                    () => hessenberg(a, outH: canaryH.view, outQ: canaryQ.view),
                    throwsA(isA<LinAlgException>()),
                  );
                }
                canaryH.verifySentinelsIntact('hessenberg h ($label)');
                canaryQ.verifySentinelsIntact('hessenberg q ($label)');
              } finally {
                a.dispose();
                canaryH.dispose();
                canaryQ.dispose();
              }
            });
          });
        }
      }

      test('solve and lstsq with non-finite RHS b ($dtype)', () {
        NDArray.scope(() {
          final a = _buildMatrix(3, 3, dtype, _NonFiniteKind.singular);
          // Make `a` well-conditioned by rebuilding with finite entries:
          final aGood = NDArray.eye(3, dtype);
          final bNaN = _buildVector(3, dtype, kind: _NonFiniteKind.singleNaN);
          final bInf = _buildVector(3, dtype, kind: _NonFiniteKind.posInf);

          final solvedNaN = solve(aGood, bNaN);
          expect(_allElementsAreNaN(solvedNaN), isTrue);

          final solvedInf = solve(aGood, bInf);
          expect(solvedInf.shape, equals([3]));

          expect(() => lstsq(aGood, bNaN), throwsA(isA<LinAlgException>()));
          expect(() => lstsq(aGood, bInf), throwsA(isA<LinAlgException>()));
          expect(
            () => lstsq(aGood, _buildVector(3, dtype), rcond: double.nan),
            throwsA(isA<LinAlgException>()),
          );
          a.dispose();
        });
      });

      test('rectangular qr and svd with non-finite inputs ($dtype)', () {
        for (final shape in [
          [4, 3],
          [3, 4],
        ]) {
          for (final kind in [
            _NonFiniteKind.singleNaN,
            _NonFiniteKind.posInf,
            _NonFiniteKind.negInf,
          ]) {
            NDArray.scope(() {
              final a = _buildMatrix(shape[0], shape[1], dtype, kind);
              final qrRes = qr(a);
              expect(_allElementsAreNaN(qrRes.q), isTrue);
              expect(_anyElementIsNaN(qrRes.r), isTrue);

              expect(() => svd(a), throwsA(isA<LinAlgException>()));
            });
          }
        }
      });
    }
  });

  group(
    'Static source invariants for linalg LAPACK info and eigenvector bounds',
    () {
      test(
        'linalg.dart never throws ArgumentError or StateError on LAPACKE info',
        () {
          final file = File('lib/src/operations/linalg.dart');
          final content = file.readAsStringSync();

          expect(
            content.contains('Illegal value in call to LAPACKE'),
            isFalse,
            reason:
                'LAPACKE negative info codes must throw LinAlgException via _checkLapackInfo, never ArgumentError.',
          );
          final badThrowPattern = RegExp(
            r'if\s*\(\s*info(?:Tri|Org)?\s*[<!=]+\s*0\s*\)\s*\{?\s*throw\s+(?:ArgumentError|StateError)',
          );
          expect(
            badThrowPattern.hasMatch(content),
            isFalse,
            reason:
                'All LAPACKE info checks in linalg.dart must route through _checkLapackInfo and throw LinAlgException subclasses.',
          );
        },
      );

      test(
        'custom_ufuncs.cpp bounds-checks j + 1 < n in assemble_eigenvectors_*',
        () {
          final file = File('hook/custom_ufuncs.cpp');
          final content = file.readAsStringSync();

          expect(
            content.contains('if (j + 1 < n && wi[j] > 0.0)'),
            isTrue,
            reason:
                'assemble_eigenvectors_double must check j + 1 < n && wi[j] > 0.0 before reading/writing column j + 1.',
          );
          expect(
            content.contains('if (j + 1 < n && wi[j] > 0.0f)'),
            isTrue,
            reason:
                'assemble_eigenvectors_float must check j + 1 < n && wi[j] > 0.0f before reading/writing column j + 1.',
          );
        },
      );
    },
    skip: const bool.fromEnvironment('dart.tool.dart2wasm')
        ? 'Host source file inspection is not supported on Wasm'
        : false,
  );
}
