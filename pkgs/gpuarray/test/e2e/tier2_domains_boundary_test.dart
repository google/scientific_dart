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

import 'dart:typed_data';

import 'package:gpuarray/fft.dart' as gpu_fft;
import 'package:gpuarray/gpuarray.dart';
import 'package:gpuarray/linalg.dart' as gpu_linalg;
import 'package:gpuarray/nn.dart' as gpu_nn;
import 'package:gpuarray/random.dart' as gpu_random;
import 'package:resource_scope/resource_scope.dart';
import 'package:test/test.dart';

final Matcher _throwsArgOrRangeOrGpuError = throwsA(
  anyOf(isA<ArgumentError>(), isA<RangeError>(), isA<GpuException>()),
);

void main() {
  group('Tier 2 — Domain Modules Boundary/Negative (F11–F21)', () {
    group('F11: Advanced Indexing & Selection Boundaries', () {
      test(
        'F11.B1: Slice with step == 0 or out-of-bounds Index throws error',
        () {
          expect(
            () => Slice(0, 4, 0),
            throwsA(anyOf(isA<AssertionError>(), isA<ArgumentError>())),
          );
          ResourceScope.scope(() {
            final tensor = GpuArray.zeros([3, 3], DType.float32);
            expect(() => tensor[5], _throwsArgOrRangeOrGpuError);
            expect(() => tensor[-4], _throwsArgOrRangeOrGpuError);
            expect(
              () => tensor.slice([const Ellipsis(), const Ellipsis()]),
              _throwsArgOrRangeOrGpuError,
            );
          });
        },
      );

      test(
        'F11.B2: where and select with non-broadcastable or empty lists throw',
        () {
          ResourceScope.scope(() {
            final cond = GpuArray.fromList(
              <bool>[true, false],
              [2],
              DType.boolean,
            );
            final xVec = GpuArray.ones([3], DType.float32);
            final yVec = GpuArray.zeros([2], DType.float32);
            expect(() => where(cond, xVec, yVec), _throwsArgOrRangeOrGpuError);
            expect(
              () => select(<GpuArray<Boolean>>[], <GpuArray<Float32>>[]),
              _throwsArgOrRangeOrGpuError,
            );
            expect(
              () => select([cond], <GpuArray<Float32>>[]),
              _throwsArgOrRangeOrGpuError,
            );
          });
        },
      );

      test('F11.B3: take and put with out-of-bounds indices or axis throw', () {
        ResourceScope.scope(() {
          final tensor = GpuArray.ones([2, 3], DType.float32);
          final badIndices = GpuArray.fromList(<int>[10], [1], DType.int32);
          expect(() => take(tensor, badIndices), _throwsArgOrRangeOrGpuError);
          expect(
            () => take(
              tensor,
              GpuArray.fromList(<int>[0], [1], DType.int32),
              axis: 5,
            ),
            _throwsArgOrRangeOrGpuError,
          );
          expect(
            () => put(
              tensor,
              badIndices,
              GpuArray.fromList(<double>[1.0], [1], DType.float32),
            ),
            _throwsArgOrRangeOrGpuError,
          );
        });
      });

      test(
        'F11.B4: takeAlongAxis and putAlongAxis rank/shape mismatch throws',
        () {
          ResourceScope.scope(() {
            final matrix = GpuArray.ones([2, 3], DType.float32);
            final wrongRankIdx = GpuArray.fromList(
              <int>[0, 1],
              [2],
              DType.int32,
            );
            expect(
              () => takeAlongAxis(matrix, wrongRankIdx, 1),
              _throwsArgOrRangeOrGpuError,
            );
            expect(
              () => putAlongAxis(
                matrix,
                wrongRankIdx,
                GpuArray.ones([2], DType.float32),
                1,
              ),
              _throwsArgOrRangeOrGpuError,
            );
          });
        },
      );

      test(
        'F11.B5: indexing operations reject invalid or disposed out: buffers',
        () {
          ResourceScope.scope(() {
            final cond = GpuArray.fromList(
              <bool>[true, false],
              [2],
              DType.boolean,
            );
            final xVec = GpuArray.ones([2], DType.float32);
            final yVec = GpuArray.zeros([2], DType.float32);
            final badOut = GpuArray.zeros([5], DType.float32);
            expect(
              () => where(cond, xVec, yVec, out: badOut),
              _throwsArgOrRangeOrGpuError,
            );
            final disposedOut = GpuArray.zeros([2], DType.float32)..dispose();
            expect(
              () => where(cond, xVec, yVec, out: disposedOut),
              throwsStateError,
            );
          });
        },
      );
    });

    group('F12: Tensor Manipulation & Geometry Boundaries', () {
      test(
        'F12.B1: concatenate and stack with empty list or mismatched shapes throw',
        () {
          ResourceScope.scope(() {
            expect(
              () => concatenate(<GpuArray<Float32>>[]),
              _throwsArgOrRangeOrGpuError,
            );
            expect(
              () => stack(<GpuArray<Float32>>[]),
              _throwsArgOrRangeOrGpuError,
            );
            final aMat = GpuArray.zeros([2, 3], DType.float32);
            final bMat = GpuArray.zeros([2, 4], DType.float32);
            expect(
              () => concatenate([aMat, bMat], axis: 0),
              _throwsArgOrRangeOrGpuError,
            );
            expect(() => stack([aMat, bMat]), _throwsArgOrRangeOrGpuError);
          });
        },
      );

      test(
        'F12.B2: split with non-divisible sections throws ArgumentError',
        () {
          ResourceScope.scope(() {
            final vec = GpuArray.zeros([5], DType.float32);
            expect(() => split(vec, 2), _throwsArgOrRangeOrGpuError);
            expect(() => split(vec, 0), _throwsArgOrRangeOrGpuError);
          });
        },
      );

      test(
        'F12.B3: pad with negative pad widths or empty reflect input throws',
        () {
          ResourceScope.scope(() {
            final vec = GpuArray.ones([3], DType.float32);
            expect(
              () => pad(vec, [
                [-1, 1],
              ]),
              _throwsArgOrRangeOrGpuError,
            );
            expect(
              () => pad(vec, [
                [1, 1],
                [1, 1],
              ]),
              _throwsArgOrRangeOrGpuError,
            );
          });
        },
      );

      test(
        'F12.B4: diag on 3D tensor or rot90 on 1D tensor throws ArgumentError',
        () {
          ResourceScope.scope(() {
            final cube = GpuArray.zeros([2, 2, 2], DType.float32);
            expect(() => diag(cube), _throwsArgOrRangeOrGpuError);
            final vec = GpuArray.zeros([4], DType.float32);
            expect(() => rot90(vec), _throwsArgOrRangeOrGpuError);
            expect(() => triu(vec), _throwsArgOrRangeOrGpuError);
            expect(() => tril(vec), _throwsArgOrRangeOrGpuError);
          });
        },
      );

      test(
        'F12.B5: broadcastTo incompatible target shape throws ArgumentError',
        () {
          ResourceScope.scope(() {
            final vec = GpuArray.ones([3], DType.float32);
            expect(() => broadcastTo(vec, [2, 4]), _throwsArgOrRangeOrGpuError);
            expect(
              () => broadcastArrays([
                GpuArray.ones([3], DType.float32),
                GpuArray.ones([4], DType.float32),
              ]),
              _throwsArgOrRangeOrGpuError,
            );
          });
        },
      );
    });

    group('F13: WGSL JIT Compiler & Kernel Fusion Boundaries', () {
      test(
        'F13.B1: validateWgslShader catches unbalanced braces, parens, and brackets',
        () {
          final badBraces = validateWgslShader(
            '@compute @workgroup_size(64) fn main() { let x = 1.0;',
          );
          expect(badBraces.isValid, isFalse);
          expect(badBraces.errors, isNotEmpty);

          final badParens = validateWgslShader(
            '@compute @workgroup_size(64) fn main() { let x = (1.0 + 2.0)); }',
          );
          expect(badParens.isValid, isFalse);
        },
      );

      test(
        'F13.B2: validateWgslShader catches missing @compute or duplicate @binding',
        () {
          final missingStage = validateWgslShader('fn main() {}');
          expect(missingStage.isValid, isFalse);

          final duplicateBinding = validateWgslShader('''
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(0) var<storage, read_write> b: array<f32>;
@compute @workgroup_size(64) fn main() {}
''');
          expect(duplicateBinding.isValid, isFalse);
        },
      );

      test(
        'F13.B3: FusedKernelDescriptor without expression throws ArgumentError',
        () {
          expect(
            () => FusedKernelDescriptor(name: 'empty_desc'),
            throwsArgumentError,
          );
          expect(() => Expr.from('invalid_object'), throwsArgumentError);
        },
      );

      test(
        'F13.B4: WgslTemplates unsupported operator throws ArgumentError',
        () {
          expect(
            () => WgslTemplates.elementwiseBinary(op: 'unsupported_binary_op'),
            throwsArgumentError,
          );
          expect(
            () => WgslTemplates.elementwiseUnary(op: 'unsupported_unary_op'),
            throwsArgumentError,
          );
          expect(
            () => WgslTemplates.treeReduction(op: 'unsupported_reduction'),
            throwsArgumentError,
          );
        },
      );

      test(
        'F13.B5: GpuComputePipelinePackage.fromJson and WebGpuSlider.fromJson reject malformed JSON',
        () {
          expect(
            () => WebGpuSlider.fromJson(const {'name': 123}),
            throwsFormatException,
          );
          expect(
            () => GpuComputePipelinePackage.fromJson(const {'invalid': true}),
            throwsFormatException,
          );
        },
      );
    });

    group('F14: safetensors Serialization & File I/O Boundaries', () {
      test('F14.B1: loadSafetensors rejects truncated payload (< 8 bytes)', () {
        expect(
          () => loadSafetensors(Uint8List.fromList([1, 2, 3])),
          throwsFormatException,
        );
      });

      test(
        'F14.B2: loadSafetensors rejects header length exceeding buffer length',
        () {
          final badHeader = Uint8List(16);
          final view = ByteData.sublistView(badHeader);
          view.setUint64(0, 999999, Endian.little);
          expect(() => loadSafetensors(badHeader), throwsFormatException);
        },
      );

      test(
        'F14.B3: loadSafetensors rejects malformed UTF-8 or non-JSON header',
        () {
          final badJson = Uint8List(16);
          final view = ByteData.sublistView(badJson);
          view.setUint64(0, 4, Endian.little);
          badJson.setRange(8, 12, [0x7B, 0x21, 0x21, 0x7D]);
          expect(() => loadSafetensors(badJson), throwsFormatException);
        },
      );

      test(
        'F14.B4: saveSafetensors on disposed GpuArray throws StateError',
        () {
          final disposed = GpuArray.ones([2, 2], DType.float32)..dispose();
          expect(
            () => saveSafetensors({'disposed': disposed}),
            throwsStateError,
          );
        },
      );

      test(
        'F14.B5: loadSafetensors on disposed device throws GpuDeviceDisposedException',
        () async {
          final device = await createWebGpuDevice(name: 'Disposed-Safetensors');
          late final Uint8List validBytes;
          ResourceScope.scope(() {
            final tensor = GpuArray.ones([2], DType.float32);
            validBytes = saveSafetensors({'t': tensor});
          });
          device.dispose();
          expect(
            () => loadSafetensors(validBytes, device: device),
            throwsA(isA<GpuDeviceDisposedException>()),
          );
        },
      );
    });

    group('F15: linalg Matrix Decompositions Boundaries', () {
      test('F15.B1: decompositions reject 1D vector inputs (< 2D)', () {
        ResourceScope.scope(() {
          final vec = GpuArray.ones([4], DType.float64);
          expect(() => gpu_linalg.svd(vec), _throwsArgOrRangeOrGpuError);
          expect(() => gpu_linalg.qr(vec), _throwsArgOrRangeOrGpuError);
          expect(() => gpu_linalg.cholesky(vec), _throwsArgOrRangeOrGpuError);
          expect(() => gpu_linalg.eigh(vec), _throwsArgOrRangeOrGpuError);
          expect(() => gpu_linalg.lu(vec), _throwsArgOrRangeOrGpuError);
        });
      });

      test(
        'F15.B2: cholesky, eigh, eig, and luFactor reject non-square matrices',
        () {
          ResourceScope.scope(() {
            final rect = GpuArray.ones([2, 3], DType.float64);
            expect(
              () => gpu_linalg.cholesky(rect),
              _throwsArgOrRangeOrGpuError,
            );
            expect(() => gpu_linalg.eigh(rect), _throwsArgOrRangeOrGpuError);
            expect(() => gpu_linalg.eig(rect), _throwsArgOrRangeOrGpuError);
            expect(
              () => gpu_linalg.luFactor(rect),
              _throwsArgOrRangeOrGpuError,
            );
          });
        },
      );

      test(
        'F15.B3: cholesky on non-positive-definite matrix throws or yields NaN',
        () {
          ResourceScope.scope(() {
            final nonSpd = GpuArray.fromList(
              <double>[-4.0, 1.0, 1.0, -2.0],
              [2, 2],
              DType.float64,
            );
            try {
              final factor = gpu_linalg.cholesky(nonSpd);
              final values = factor.toList().cast<num>();
              expect(values.any((v) => v.isNaN), isTrue);
            } on Object catch (error) {
              expect(
                error,
                anyOf(
                  isA<ArgumentError>(),
                  isA<StateError>(),
                  isA<GpuException>(),
                ),
              );
            }
          });
        },
      );

      test(
        'F15.B4: luSolve with mismatched pivots or RHS shape throws error',
        () {
          ResourceScope.scope(() {
            final mat = GpuArray.fromList(
              <double>[2.0, 1.0, 1.0, 2.0],
              [2, 2],
              DType.float64,
            );
            final factored = gpu_linalg.luFactor(mat);
            final badRhs = GpuArray.ones([4], DType.float64);
            expect(
              () => gpu_linalg.luSolve(factored.lu, factored.pivots, badRhs),
              _throwsArgOrRangeOrGpuError,
            );
            factored.dispose();
          });
        },
      );

      test(
        'F15.B5: decompositions reject wrong-shaped or disposed out: buffers',
        () {
          ResourceScope.scope(() {
            final mat = GpuArray.fromList(
              <double>[4.0, 1.0, 1.0, 3.0],
              [2, 2],
              DType.float64,
            );
            final wrongOut = GpuArray.zeros([3, 3], DType.float64);
            expect(
              () => gpu_linalg.cholesky(mat, out: wrongOut),
              _throwsArgOrRangeOrGpuError,
            );
            final disposedOut = GpuArray.zeros([2, 2], DType.float64)
              ..dispose();
            expect(
              () => gpu_linalg.cholesky(mat, out: disposedOut),
              throwsStateError,
            );
          });
        },
      );
    });

    group('F16: linalg Solvers, Inverses & Norms Boundaries', () {
      test(
        'F16.B1: solve, inv, det, slogdet, matrixPower reject non-square matrices',
        () {
          ResourceScope.scope(() {
            final rect = GpuArray.ones([2, 3], DType.float64);
            final rhs = GpuArray.ones([2], DType.float64);
            expect(
              () => gpu_linalg.solve(rect, rhs),
              _throwsArgOrRangeOrGpuError,
            );
            expect(() => gpu_linalg.inv(rect), _throwsArgOrRangeOrGpuError);
            expect(() => gpu_linalg.det(rect), _throwsArgOrRangeOrGpuError);
            expect(() => gpu_linalg.slogdet(rect), _throwsArgOrRangeOrGpuError);
            expect(
              () => gpu_linalg.matrixPower(rect, 2),
              _throwsArgOrRangeOrGpuError,
            );
          });
        },
      );

      test(
        'F16.B2: solve with incompatible RHS dimension throws ArgumentError',
        () {
          ResourceScope.scope(() {
            final mat = GpuArray.fromList(
              <double>[2.0, 0.0, 0.0, 2.0],
              [2, 2],
              DType.float64,
            );
            final badRhs = GpuArray.ones([3], DType.float64);
            expect(
              () => gpu_linalg.solve(mat, badRhs),
              _throwsArgOrRangeOrGpuError,
            );
          });
        },
      );

      test(
        'F16.B3: matrixPower with n == 0 returns identity and n < 0 inverts',
        () {
          ResourceScope.scope(() {
            final diagMat = GpuArray.fromList(
              <double>[2.0, 0.0, 0.0, 4.0],
              [2, 2],
              DType.float64,
            );
            final zeroPow = gpu_linalg.matrixPower(diagMat, 0);
            expect(
              zeroPow.toList().map((e) => (e as num).toDouble()).toList(),
              equals(<double>[1.0, 0.0, 0.0, 1.0]),
            );
            final negPow = gpu_linalg.matrixPower(diagMat, -1);
            final negList = negPow.toList().cast<num>();
            expect(negList[0].toDouble(), closeTo(0.5, 1e-4));
            expect(negList[3].toDouble(), closeTo(0.25, 1e-4));
          });
        },
      );

      test(
        'F16.B4: multiDot with < 2 matrices or mismatched inner dims throws',
        () {
          ResourceScope.scope(() {
            final mat = GpuArray.ones([2, 2], DType.float64);
            expect(
              () => gpu_linalg.multiDot([mat]),
              _throwsArgOrRangeOrGpuError,
            );
            final incompatible = GpuArray.ones([3, 2], DType.float64);
            expect(
              () => gpu_linalg.multiDot([mat, incompatible]),
              _throwsArgOrRangeOrGpuError,
            );
          });
        },
      );

      test('F16.B5: norm and cond reject invalid axes or disposed inputs', () {
        ResourceScope.scope(() {
          final vec = GpuArray.ones([3], DType.float64);
          expect(
            () => gpu_linalg.norm(vec, ord: gpu_linalg.NormOrd.nuclear),
            _throwsArgOrRangeOrGpuError,
          );
          expect(() => gpu_linalg.cond(vec), _throwsArgOrRangeOrGpuError);
        });
      });
    });

    group('F17: linalg Tensor Contractions Boundaries', () {
      test(
        'F17.B1: einsum with malformed subscripts or wrong operand count throws',
        () {
          ResourceScope.scope(() {
            final mat = GpuArray.ones([2, 2], DType.float32);
            expect(
              () => gpu_linalg.einsum('ij,jk->ik', [mat]),
              _throwsArgOrRangeOrGpuError,
            );
            expect(
              () => gpu_linalg.einsum('invalid->subscripts->here', [mat]),
              _throwsArgOrRangeOrGpuError,
            );
          });
        },
      );

      test(
        'F17.B2: einsum with mismatched contracted dimension sizes throws',
        () {
          ResourceScope.scope(() {
            final left = GpuArray.ones([2, 3], DType.float32);
            final right = GpuArray.ones([4, 2], DType.float32);
            expect(
              () => gpu_linalg.einsum('ij,jk->ik', [left, right]),
              _throwsArgOrRangeOrGpuError,
            );
          });
        },
      );

      test(
        'F17.B3: tensordot with invalid axes count or mismatched dimensions throws',
        () {
          ResourceScope.scope(() {
            final left = GpuArray.ones([2, 3], DType.float32);
            final right = GpuArray.ones([4, 2], DType.float32);
            expect(
              () => gpu_linalg.tensordot(left, right, axes: 1),
              _throwsArgOrRangeOrGpuError,
            );
            expect(
              () => gpu_linalg.tensordot(left, right, axes: 5),
              _throwsArgOrRangeOrGpuError,
            );
          });
        },
      );

      test(
        'F17.B4: inner with mismatched last-axis lengths and cross with non-3D vectors throw',
        () {
          ResourceScope.scope(() {
            final v3 = GpuArray.ones([3], DType.float32);
            final v4 = GpuArray.ones([4], DType.float32);
            expect(() => gpu_linalg.inner(v3, v4), _throwsArgOrRangeOrGpuError);
            expect(() => gpu_linalg.cross(v4, v4), _throwsArgOrRangeOrGpuError);
          });
        },
      );

      test(
        'F17.B5: tensor contractions reject wrong-shaped or disposed out: buffers',
        () {
          ResourceScope.scope(() {
            final uVec = GpuArray.ones([2], DType.float32);
            final vVec = GpuArray.ones([3], DType.float32);
            final wrongOut = GpuArray.zeros([4, 4], DType.float32);
            expect(
              () => gpu_linalg.outer(uVec, vVec, out: wrongOut),
              _throwsArgOrRangeOrGpuError,
            );
            final disposedOut = GpuArray.zeros([2, 3], DType.float32)
              ..dispose();
            expect(
              () => gpu_linalg.outer(uVec, vVec, out: disposedOut),
              throwsStateError,
            );
          });
        },
      );
    });

    group('F18: fft Fast Fourier Transforms Boundaries', () {
      test('F18.B1: fft and ifft with n <= 0 or invalid axis throw error', () {
        ResourceScope.scope(() {
          final sig = GpuArray.ones([8], DType.float64);
          expect(() => gpu_fft.fft(sig, n: 0), _throwsArgOrRangeOrGpuError);
          expect(() => gpu_fft.fft(sig, n: -4), _throwsArgOrRangeOrGpuError);
          expect(() => gpu_fft.fft(sig, axis: 3), _throwsArgOrRangeOrGpuError);
        });
      });

      test('F18.B2: rfft on complex input or invalid n throws error', () {
        ResourceScope.scope(() {
          final sig = GpuArray.ones([8], DType.float64);
          expect(() => gpu_fft.rfft(sig, n: 0), _throwsArgOrRangeOrGpuError);
          expect(() => gpu_fft.irfft(sig, n: 0), _throwsArgOrRangeOrGpuError);
        });
      });

      test('F18.B3: fft2 and ifft2 on 1D tensor throw ArgumentError', () {
        ResourceScope.scope(() {
          final vec = GpuArray.ones([8], DType.float64);
          expect(() => gpu_fft.fft2(vec), _throwsArgOrRangeOrGpuError);
          expect(() => gpu_fft.ifft2(vec), _throwsArgOrRangeOrGpuError);
        });
      });

      test(
        'F18.B4: fftfreq and rfftfreq with n <= 0 or d <= 0 throw ArgumentError',
        () {
          expect(() => gpu_fft.fftfreq(0), _throwsArgOrRangeOrGpuError);
          expect(() => gpu_fft.fftfreq(-4), _throwsArgOrRangeOrGpuError);
          expect(() => gpu_fft.fftfreq(4, d: 0.0), _throwsArgOrRangeOrGpuError);
          expect(() => gpu_fft.rfftfreq(0), _throwsArgOrRangeOrGpuError);
        },
      );

      test(
        'F18.B5: fft operations reject wrong-shaped or disposed out: buffers',
        () {
          ResourceScope.scope(() {
            final sig = GpuArray.ones([4], DType.float64);
            final wrongOut = GpuArray.zeros([8], DType.complex128);
            expect(
              () => gpu_fft.fft(sig, out: wrongOut),
              _throwsArgOrRangeOrGpuError,
            );
            final disposedOut = GpuArray.zeros([4], DType.complex128)
              ..dispose();
            expect(() => gpu_fft.fft(sig, out: disposedOut), throwsStateError);
          });
        },
      );
    });

    group('F19: random Counter-Based Philox4x32 RNG Boundaries', () {
      test('F19.B1: uniform with low > high throws ArgumentError', () {
        final rng = gpu_random.RandomState(1);
        expect(
          () => rng.uniform(low: 5.01, high: 5.0, shape: [4]),
          _throwsArgOrRangeOrGpuError,
        );
        expect(
          () => rng.uniform(low: 10.0, high: 2.0, shape: [4]),
          _throwsArgOrRangeOrGpuError,
        );
      });

      test('F19.B2: randint with low >= high throws ArgumentError', () {
        final rng = gpu_random.RandomState(2);
        expect(() => rng.randint(5, 5, [4]), _throwsArgOrRangeOrGpuError);
        expect(() => rng.randint(8, 3, [4]), _throwsArgOrRangeOrGpuError);
      });

      test(
        'F19.B3: normal and exponential with scale <= 0 throw ArgumentError',
        () {
          final rng = gpu_random.RandomState(3);
          expect(
            () => rng.normal(scale: 0.0, shape: [4]),
            _throwsArgOrRangeOrGpuError,
          );
          expect(
            () => rng.normal(scale: -1.0, shape: [4]),
            _throwsArgOrRangeOrGpuError,
          );
          expect(
            () => rng.exponential(scale: 0.0, shape: [4]),
            _throwsArgOrRangeOrGpuError,
          );
          expect(
            () => rng.exponential(scale: -2.0, shape: [4]),
            _throwsArgOrRangeOrGpuError,
          );
        },
      );

      test(
        'F19.B4: choice without replacement exceeding population or invalid probabilities throws',
        () {
          ResourceScope.scope(() {
            final rng = gpu_random.RandomState(4);
            final pop3 = GpuArray.fromList(<int>[0, 1, 2], [3], DType.int64);
            expect(
              () => rng.choice(pop3, shape: [5], replace: false),
              _throwsArgOrRangeOrGpuError,
            );
            final pop2 = GpuArray.fromList(<int>[0, 1], [2], DType.int64);
            expect(
              () => rng.choice(pop2, shape: [1], p: [-0.5, 1.5]),
              _throwsArgOrRangeOrGpuError,
            );
          });
        },
      );

      test(
        'F19.B5: random generators reject negative shape dims or disposed out:',
        () {
          ResourceScope.scope(() {
            final rng = gpu_random.RandomState(5);
            expect(() => rng.rand([-2]), _throwsArgOrRangeOrGpuError);
            final disposedOut = GpuArray.zeros([4], DType.float64)..dispose();
            expect(() => rng.rand([4], null, disposedOut), throwsStateError);
          });
        },
      );
    });

    group('F20: autograd Reverse-Mode Automatic Differentiation Boundaries', () {
      test(
        'F20.B1: backward on non-scalar without explicit gradient throws error',
        () {
          ResourceScope.scope(() {
            final vec = GpuArray.fromList(
              <double>[1.0, 2.0],
              [2],
              DType.float64,
              requiresGrad: true,
            );
            final out = vec * 2.0;
            expect(
              () => out.backward(),
              throwsA(anyOf(isA<StateError>(), isA<ArgumentError>())),
            );
          });
        },
      );

      test(
        'F20.B2: backward with mismatched explicit gradient shape throws error',
        () {
          ResourceScope.scope(() {
            final vec = GpuArray.fromList(
              <double>[1.0, 2.0],
              [2],
              DType.float64,
              requiresGrad: true,
            );
            final out = vec * 2.0;
            final wrongGrad = GpuArray.ones([3], DType.float64);
            expect(
              () => out.backward(gradient: wrongGrad),
              _throwsArgOrRangeOrGpuError,
            );
          });
        },
      );

      test(
        'F20.B3: second backward without retainGraph: true throws StateError',
        () {
          ResourceScope.scope(() {
            final xLeaf = GpuArray.fromList(
              <double>[2.0, 3.0],
              [2],
              DType.float64,
              requiresGrad: true,
            );
            final loss = (xLeaf * xLeaf).sum();
            loss.backward();
            expect(() => loss.backward(), throwsStateError);
          });
        },
      );

      test(
        'F20.B4: requiresGrad on integer or boolean tensor is rejected or produces no gradFn',
        () {
          expect(
            () => GpuArray.fromList(
              <int>[1, 2],
              [2],
              DType.int32,
              requiresGrad: true,
            ),
            throwsA(anyOf(isA<ArgumentError>(), isA<UnsupportedError>())),
          );
        },
      );

      test(
        'F20.B5: exception inside noGrad still restores isGradEnabled == true',
        () {
          expect(isGradEnabled, isTrue);
          expect(
            () => noGrad(() {
              expect(isGradEnabled, isFalse);
              throw StateError('Simulated error in noGrad');
            }),
            throwsStateError,
          );
          expect(isGradEnabled, isTrue);
        },
      );
    });

    group(
      'F21: nn Modules, Activations, Losses, Optimizers & ResourceScope Boundaries',
      () {
        test(
          'F21.B1: Linear with non-positive features or mismatched input shape throws',
          () {
            ResourceScope.scope(() {
              expect(() => gpu_nn.Linear(0, 4), _throwsArgOrRangeOrGpuError);
              expect(() => gpu_nn.Linear(4, -1), _throwsArgOrRangeOrGpuError);
              final layer = gpu_nn.Linear(4, 2);
              final wrongInput = GpuArray.ones([2, 3], DType.float64);
              expect(
                () => layer.forward(wrongInput),
                _throwsArgOrRangeOrGpuError,
              );
            });
          },
        );

        test(
          'F21.B2: Conv2d with invalid channels, stride <= 0, or 2D input throws',
          () {
            ResourceScope.scope(() {
              expect(() => gpu_nn.Conv2d(0, 2, 3), _throwsArgOrRangeOrGpuError);
              expect(
                () => gpu_nn.Conv2d(1, 2, 3, stride: 0),
                _throwsArgOrRangeOrGpuError,
              );
              final conv = gpu_nn.Conv2d(2, 4, 3);
              final wrongChannels = GpuArray.ones([1, 1, 5, 5], DType.float64);
              expect(
                () => conv.forward(wrongChannels),
                _throwsArgOrRangeOrGpuError,
              );
            });
          },
        );

        test(
          'F21.B3: LayerNorm and RMSNorm with empty normalizedShape or eps <= 0 throw',
          () {
            expect(
              () => gpu_nn.LayerNorm(const []),
              _throwsArgOrRangeOrGpuError,
            );
            expect(
              () => gpu_nn.LayerNorm(const [4], eps: 0.0),
              _throwsArgOrRangeOrGpuError,
            );
            expect(() => gpu_nn.RMSNorm(const []), _throwsArgOrRangeOrGpuError);
            expect(
              () => gpu_nn.RMSNorm(const [4], eps: -1e-5),
              _throwsArgOrRangeOrGpuError,
            );
          },
        );

        test('F21.B4: Dropout with p < 0 or p >= 1 throws ArgumentError', () {
          expect(() => gpu_nn.Dropout(p: -0.1), _throwsArgOrRangeOrGpuError);
          expect(() => gpu_nn.Dropout(p: 1.0), _throwsArgOrRangeOrGpuError);
          expect(() => gpu_nn.Dropout(p: 1.5), _throwsArgOrRangeOrGpuError);
        });

        test(
          'F21.B5: Embedding with out-of-range token index throws RangeError or ArgumentError',
          () {
            ResourceScope.scope(() {
              final emb = gpu_nn.Embedding(4, 3);
              final badIds = GpuArray.fromList(<int>[0, 5], [2], DType.int32);
              expect(() => emb.forward(badIds), _throwsArgOrRangeOrGpuError);
            });
          },
        );

        test(
          'F21.B6: MultiheadAttention with embedDim not divisible by numHeads throws',
          () {
            expect(
              () => gpu_nn.MultiheadAttention(10, 3),
              _throwsArgOrRangeOrGpuError,
            );
            expect(
              () => gpu_nn.RotaryEmbedding(3),
              _throwsArgOrRangeOrGpuError,
            );
          },
        );

        test(
          'F21.B7: mseLoss with mismatched prediction and target shapes throws',
          () {
            ResourceScope.scope(() {
              final pred = GpuArray.ones([2, 3], DType.float64);
              final target = GpuArray.ones([2, 4], DType.float64);
              expect(
                () => gpu_nn.mseLoss(pred, target),
                _throwsArgOrRangeOrGpuError,
              );
            });
          },
        );

        test(
          'F21.B8: crossEntropy with out-of-range class label or mismatched batch size throws',
          () {
            ResourceScope.scope(() {
              final logits = GpuArray.ones([2, 3], DType.float64);
              final badTargets = GpuArray.fromList(
                <int>[0, 5],
                [2],
                DType.int32,
              );
              expect(
                () => gpu_nn.crossEntropy(logits, badTargets),
                _throwsArgOrRangeOrGpuError,
              );
              final wrongBatch = GpuArray.fromList(<int>[0], [1], DType.int32);
              expect(
                () => gpu_nn.crossEntropy(logits, wrongBatch),
                _throwsArgOrRangeOrGpuError,
              );
            });
          },
        );

        test(
          'F21.B9: SGD, Adam, and AdamW reject negative learning rate or invalid hyperparameters',
          () {
            ResourceScope.scope(() {
              final param = GpuArray.ones(
                [2],
                DType.float64,
                requiresGrad: true,
              );
              expect(
                () => gpu_nn.SGD([param], lr: -0.01),
                _throwsArgOrRangeOrGpuError,
              );
              expect(
                () => gpu_nn.Adam([param], lr: -0.001),
                _throwsArgOrRangeOrGpuError,
              );
              expect(
                () => gpu_nn.AdamW([param], lr: 0.01, weightDecay: -0.1),
                _throwsArgOrRangeOrGpuError,
              );
            });
          },
        );

        test(
          'F21.B10: detachToParentScope at root scope (no parent) throws StateError',
          () {
            final tensor = GpuArray.ones([2], DType.float32);
            try {
              expect(() => tensor.detachToParentScope(), throwsStateError);
            } finally {
              tensor.dispose();
            }
          },
        );
      },
    );
  });
}
