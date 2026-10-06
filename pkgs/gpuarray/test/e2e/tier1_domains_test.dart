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

import 'dart:io';

import 'package:gpuarray/fft.dart' as gpu_fft;
import 'package:gpuarray/gpuarray.dart';
import 'package:gpuarray/jit.dart';
import 'package:gpuarray/linalg.dart' as gpu_linalg;
import 'package:gpuarray/nn.dart' as gpu_nn;
import 'package:gpuarray/random.dart' as gpu_random;
import 'package:gpuarray/serialization.dart';
import 'package:gpuarray/wgsl.dart';
import 'package:ndarray/ndarray.dart' as nd;
import 'package:resource_scope/resource_scope.dart';
import 'package:test/test.dart';

void _expectCloseList(
  List<Object?> actual,
  List<Object?> expected, {
  double tolerance = 1e-4,
}) {
  expect(actual.length, equals(expected.length));
  for (var i = 0; i < actual.length; i++) {
    final actualItem = actual[i];
    final expectedItem = expected[i];
    if (actualItem is num && expectedItem is num) {
      expect(
        actualItem.toDouble(),
        closeTo(expectedItem.toDouble(), tolerance),
        reason: 'Mismatch at element $i: $actualItem vs $expectedItem',
      );
    } else if (actualItem is Complex && expectedItem is Complex) {
      expect(actualItem.real, closeTo(expectedItem.real, tolerance));
      expect(actualItem.imag, closeTo(expectedItem.imag, tolerance));
    } else {
      expect(actualItem, equals(expectedItem));
    }
  }
}

void main() {
  group('Tier 1 — Domain Modules Happy Path (F11–F21)', () {
    group('F11: Advanced Indexing & Selection', () {
      test('F11.1: slice specs (Slice, Index, All, NewAxis, Ellipsis)', () {
        ResourceScope.scope(() {
          final values = List<double>.generate(12, (i) => (i + 1).toDouble());
          final tensor = GpuArray.fromList(values, [3, 4], DType.float32);

          final subgrid = tensor.slice([Slice(0, 2), Slice(1, 4, 2)]);
          expect(subgrid.shape, equals([2, 2]));
          _expectCloseList(subgrid.toList(), <double>[2, 4, 6, 8]);

          final rowView = tensor[1];
          expect(rowView.shape, equals([4]));
          _expectCloseList(rowView.toList(), <double>[5, 6, 7, 8]);

          final expanded = tensor.slice([const Ellipsis(), const NewAxis()]);
          expect(expanded.shape, equals([3, 4, 1]));

          final negativeStep = rowView.slice([Slice(3, null, -1)]);
          _expectCloseList(negativeStep.toList(), <double>[8, 7, 6, 5]);
        });
      });

      test('F11.2: where and select conditional selection match NDArray', () {
        ResourceScope.scope(() {
          final cond = GpuArray.fromList(
            <bool>[true, false, true, false],
            [4],
            DType.boolean,
          );
          final posBranch = GpuArray.fromList(
            <double>[10.0, 20.0, 30.0, 40.0],
            [4],
            DType.float32,
          );
          final negBranch = GpuArray.fromList(
            <double>[-1.0, -2.0, -3.0, -4.0],
            [4],
            DType.float32,
          );
          final chosen = where(cond, posBranch, negBranch);
          _expectCloseList(chosen.toList(), <double>[10.0, -2.0, 30.0, -4.0]);

          final secondCond = GpuArray.fromList(
            <bool>[false, true, false, false],
            [4],
            DType.boolean,
          );
          final multiSelect = select(
            [cond, secondCond],
            [posBranch, negBranch],
            defaultValue: GpuArray.filled([4], 99.0, DType.float32),
          );
          _expectCloseList(multiSelect.toList(), <double>[
            10.0,
            -2.0,
            30.0,
            99.0,
          ]);
        });
      });

      test('F11.3: extract, nonzero, flatnonzero, and argwhere', () {
        ResourceScope.scope(() {
          final matrix = GpuArray.fromList(
            <double>[0.0, 3.0, 0.0, -5.0, 7.0, 0.0],
            [2, 3],
            DType.float32,
          );
          final mask = matrix.notEqual(0.0);
          final extracted = extract(mask, matrix);
          _expectCloseList(extracted.toList(), <double>[3.0, -5.0, 7.0]);

          final flatLocations = flatnonzero(matrix);
          expect(flatLocations.toList(), equals(<int>[1, 3, 4]));

          final tupleIndices = nonzero(matrix);
          expect(tupleIndices.length, equals(2));
          expect(tupleIndices[0].toList(), equals(<int>[0, 1, 1]));
          expect(tupleIndices[1].toList(), equals(<int>[1, 0, 1]));

          final coords = argwhere(matrix);
          expect(coords.shape, equals([3, 2]));
          expect(coords.toList(), equals(<int>[0, 1, 1, 0, 1, 1]));
        });
      });

      test('F11.4: take and put along flattened and specific axes', () {
        ResourceScope.scope(() {
          final source = GpuArray.fromList(
            <double>[10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            [2, 3],
            DType.float32,
          );
          final flatIndices = GpuArray.fromList(
            <int>[5, 0, 3],
            [3],
            DType.int32,
          );
          final gatheredFlat = take(source, flatIndices);
          _expectCloseList(gatheredFlat.toList(), <double>[60.0, 10.0, 40.0]);

          final colIndices = GpuArray.fromList(<int>[2, 0], [2], DType.int32);
          final gatheredAxis = take(source, colIndices, axis: 1);
          expect(gatheredAxis.shape, equals([2, 2]));
          _expectCloseList(gatheredAxis.toList(), <double>[
            30.0,
            10.0,
            60.0,
            40.0,
          ]);

          final putTarget = source.copy();
          final putValues = GpuArray.fromList(
            <double>[-99.0, -11.0, -44.0],
            [3],
            DType.float32,
          );
          put(putTarget, flatIndices, putValues);
          _expectCloseList(putTarget.toList(), <double>[
            -11.0,
            20.0,
            30.0,
            -44.0,
            50.0,
            -99.0,
          ]);
        });
      });

      test('F11.5: takeAlongAxis and putAlongAxis', () {
        ResourceScope.scope(() {
          final matrix = GpuArray.fromList(
            <double>[10.0, 30.0, 20.0, 60.0, 40.0, 50.0],
            [2, 3],
            DType.float32,
          );
          final axisIndices = GpuArray.fromList(
            <int>[1, 2, 0, 1],
            [2, 2],
            DType.int32,
          );
          final taken = takeAlongAxis(matrix, axisIndices, 1);
          expect(taken.shape, equals([2, 2]));
          _expectCloseList(taken.toList(), <double>[30.0, 20.0, 60.0, 40.0]);

          final mutated = matrix.copy();
          final replacements = GpuArray.fromList(
            <double>[1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float32,
          );
          putAlongAxis(mutated, axisIndices, replacements, 1);
          _expectCloseList(mutated.toList(), <double>[
            10.0,
            1.0,
            2.0,
            3.0,
            4.0,
            50.0,
          ]);
        });
      });
    });

    group('F12: Tensor Manipulation & Geometry', () {
      test(
        'F12.1: concatenate, stack, vstack, hstack, dstack, columnStack',
        () {
          ResourceScope.scope(() {
            final left = GpuArray.fromList(
              <double>[1, 2, 3, 4],
              [2, 2],
              DType.float32,
            );
            final right = GpuArray.fromList(
              <double>[5, 6, 7, 8],
              [2, 2],
              DType.float32,
            );

            final cat0 = concatenate([left, right], axis: 0);
            expect(cat0.shape, equals([4, 2]));
            _expectCloseList(cat0.toList(), <double>[1, 2, 3, 4, 5, 6, 7, 8]);

            final cat1 = hstack([left, right]);
            expect(cat1.shape, equals([2, 4]));
            _expectCloseList(cat1.toList(), <double>[1, 2, 5, 6, 3, 4, 7, 8]);

            final stacked = stack([left, right], axis: 0);
            expect(stacked.shape, equals([2, 2, 2]));

            final vstacked = vstack([left, right]);
            expect(vstacked.shape, equals([4, 2]));

            final depthStacked = dstack([left, right]);
            expect(depthStacked.shape, equals([2, 2, 2]));

            final colStacked = columnStack([
              GpuArray.fromList(<double>[1, 2], [2], DType.float32),
              GpuArray.fromList(<double>[3, 4], [2], DType.float32),
            ]);
            expect(colStacked.shape, equals([2, 2]));
            _expectCloseList(colStacked.toList(), <double>[1, 3, 2, 4]);
          });
        },
      );

      test('F12.2: split, arraySplit, hsplit, vsplit, dsplit', () {
        ResourceScope.scope(() {
          final matrix = GpuArray.fromList(
            List<double>.generate(12, (i) => (i + 1).toDouble()),
            [3, 4],
            DType.float32,
          );
          final parts = split(matrix, 2, axis: 1);
          expect(parts.length, equals(2));
          expect(parts[0].shape, equals([3, 2]));
          expect(parts[1].shape, equals([3, 2]));

          final uneven = arraySplit(matrix, 2, axis: 0);
          expect(uneven.length, equals(2));
          expect(uneven[0].shape, equals([2, 4]));
          expect(uneven[1].shape, equals([1, 4]));

          expect(hsplit(matrix, 2).length, equals(2));
          expect(vsplit(matrix, 3).length, equals(3));

          final cube = GpuArray.zeros([2, 2, 4], DType.float32);
          expect(dsplit(cube, 2).length, equals(2));
        });
      });

      test('F12.3: tile, repeat, and pad across all 5 PadModes', () {
        ResourceScope.scope(() {
          final vec = GpuArray.fromList(
            <double>[1.0, 2.0, 3.0],
            [3],
            DType.float32,
          );
          final tiled = tile(vec, [2]);
          _expectCloseList(tiled.toList(), <double>[1, 2, 3, 1, 2, 3]);

          final repeated = repeat(vec, 2);
          _expectCloseList(repeated.toList(), <double>[1, 1, 2, 2, 3, 3]);

          final paddedConst = pad(
            vec,
            [
              [1, 1],
            ],
            mode: PadMode.constant,
            constantValues: 9.0,
          );
          _expectCloseList(paddedConst.toList(), <double>[9, 1, 2, 3, 9]);

          final paddedEdge = pad(vec, [
            [1, 1],
          ], mode: PadMode.edge);
          _expectCloseList(paddedEdge.toList(), <double>[1, 1, 2, 3, 3]);

          final paddedReflect = pad(vec, [
            [1, 1],
          ], mode: PadMode.reflect);
          _expectCloseList(paddedReflect.toList(), <double>[2, 1, 2, 3, 2]);

          final paddedSymmetric = pad(vec, [
            [1, 1],
          ], mode: PadMode.symmetric);
          _expectCloseList(paddedSymmetric.toList(), <double>[1, 1, 2, 3, 3]);

          final paddedWrap = pad(vec, [
            [1, 1],
          ], mode: PadMode.wrap);
          _expectCloseList(paddedWrap.toList(), <double>[3, 1, 2, 3, 1]);
        });
      });

      test('F12.4: roll, flip, fliplr, flipud, ravel, and rot90', () {
        ResourceScope.scope(() {
          final grid = GpuArray.fromList(
            <double>[1, 2, 3, 4],
            [2, 2],
            DType.float32,
          );
          _expectCloseList(roll(grid, 1).toList(), <double>[4, 1, 2, 3]);
          _expectCloseList(fliplr(grid).toList(), <double>[2, 1, 4, 3]);
          _expectCloseList(flipud(grid).toList(), <double>[3, 4, 1, 2]);
          _expectCloseList(flip(grid).toList(), <double>[4, 3, 2, 1]);
          _expectCloseList(rot90(grid, k: 1).toList(), <double>[2, 4, 1, 3]);
          _expectCloseList(ravel(grid.transpose()).toList(), <double>[
            1,
            3,
            2,
            4,
          ]);
        });
      });

      test(
        'F12.5: diag, diagonal, trace, triu, tril, moveaxis, broadcastTo',
        () {
          ResourceScope.scope(() {
            final vec = GpuArray.fromList(
              <double>[2.0, 3.0, 4.0],
              [3],
              DType.float32,
            );
            final diagMatrix = diag(vec);
            expect(diagMatrix.shape, equals([3, 3]));
            _expectCloseList(diagonal(diagMatrix).toList(), <double>[2, 3, 4]);
            expect(
              (trace(diagMatrix).scalar as num).toDouble(),
              closeTo(9.0, 1e-5),
            );

            final dense = GpuArray.fromList(
              <double>[1, 2, 3, 4, 5, 6, 7, 8, 9],
              [3, 3],
              DType.float32,
            );
            _expectCloseList(triu(dense).toList(), <double>[
              1,
              2,
              3,
              0,
              5,
              6,
              0,
              0,
              9,
            ]);
            _expectCloseList(tril(dense).toList(), <double>[
              1,
              0,
              0,
              4,
              5,
              0,
              7,
              8,
              9,
            ]);

            final tensor3d = GpuArray.zeros([2, 3, 4], DType.float32);
            expect(moveaxis(tensor3d, 0, 2).shape, equals([3, 4, 2]));
            expect(swapaxes(tensor3d, 0, 1).shape, equals([3, 2, 4]));
            expect(expandDims(vec, 0).shape, equals([1, 3]));
            expect(broadcastTo(vec, [2, 3]).shape, equals([2, 3]));
            final broadcastedPair = broadcastArrays([
              GpuArray.ones([2, 1], DType.float32),
              GpuArray.ones([1, 3], DType.float32),
            ]);
            expect(broadcastedPair[0].shape, equals([2, 3]));
            expect(broadcastedPair[1].shape, equals([2, 3]));
          });
        },
      );
    });

    group('F13: WGSL JIT Compiler & Kernel Fusion', () {
      test('F13.1: Expr AST construction, symbolic grad, and CSE', () {
        final inputX = Expr.variable('x', bindingIndex: 0);
        final alpha = Expr.scalar('alpha', defaultValue: 2.0);
        final formula = (inputX * inputX) + (inputX * inputX) * alpha;
        final optimized = formula.cse();
        expect(optimized.toFingerprint(), isNotEmpty);

        final derivative = (inputX * inputX * Expr.constant(3.0)).grad(inputX);
        final wgslDerivative = derivative.toWgsl();
        expect(wgslDerivative, contains('x_val'));
      });

      test(
        'F13.2: Expr.let, Expr.loop, Expr.coord, Expr.index, and stencil offset',
        () {
          final inputGrid = Expr.variable('grid', bindingIndex: 0);
          final neighbor = inputGrid.offset(
            [0, 1],
            shape: [4, 4],
            boundary: BoundaryMode.clamp,
          );
          final coordX = Expr.coord(1, shape: [4, 4], normalized: true);
          final loopNode = Expr.loop(
            initialValues: [neighbor + coordX],
            maxIterations: 4,
            condition: (state, iter) => state[0].lessThan(100.0),
            step: (state, iter) => [state[0] * 1.5 + Expr.index()],
          );
          final bound = Expr.let(loopNode, (local) => local.relu());
          final descriptor = FusedKernelDescriptor(
            name: 'stencil_loop_kernel',
            expression: bound,
          );
          final source = descriptor.generateWgslSource();
          final validation = validateWgslShader(source);
          expect(
            validation.isValid,
            isTrue,
            reason: validation.errors.join('\n'),
          );
        },
      );

      test(
        'F13.3: WgslJitCompiler compiles and caches fused shader modules',
        () {
          final compiler = WgslJitCompiler(maxCacheSize: 8);
          final varA = Expr.variable('a', bindingIndex: 0);
          final varB = Expr.variable('b', bindingIndex: 1);
          final fusedExpr = (varA * 2.0 + varB).silu();

          final firstModule = compiler.compile(
            fusedExpr,
            kernelName: 'silu_fma',
          );
          expect(compiler.cacheMisses, equals(1));
          expect(compiler.cacheHits, equals(0));
          expect(
            WgslSyntaxValidator.validate(firstModule.code).isValid,
            isTrue,
          );

          final secondModule = compiler.compile(
            fusedExpr,
            kernelName: 'silu_fma',
          );
          expect(identical(firstModule, secondModule), isTrue);
          expect(compiler.cacheHits, equals(1));
          compiler.clearCache();
          expect(compiler.cachedCount, equals(0));
        },
      );

      test(
        'F13.4: WgslTemplates generates valid shaders across template families',
        () {
          final binaryShader = WgslTemplates.elementwiseBinary(
            op: 'add',
            strided: true,
          );
          final unaryShader = WgslTemplates.elementwiseUnary(op: 'gelu');
          final reductionShader = WgslTemplates.treeReduction(op: 'sum');
          final matmulShader = WgslTemplates.tiledMatmul(tileSize: 16);

          for (final module in [
            binaryShader,
            unaryShader,
            reductionShader,
            matmulShader,
          ]) {
            final check = validateWgslShader(module.code);
            expect(
              check.isValid,
              isTrue,
              reason: '${module.name}: ${check.errors}',
            );
          }
        },
      );

      test(
        'F13.5: GpuComputePipelinePackage and WebGpuWidget JSON/HTML round-trip',
        () {
          ResourceScope.scope(() {
            final inputTensor = GpuArray.fromList(
              <double>[1.0, 2.0, 3.0, 4.0],
              [2, 2],
              DType.float32,
            );
            final varX = Expr.variable('x', bindingIndex: 0);
            final gain = Expr.scalar('gain', defaultValue: 1.5);
            final descriptor = FusedKernelDescriptor(
              name: 'interactive_gain',
              expression: varX * gain,
            );
            final widget = descriptor.createBrowserWidget(
              inputArrays: [inputTensor],
              outputShape: [2, 2],
              sliders: const [
                WebGpuSlider(
                  name: 'gain',
                  label: 'Gain',
                  min: 0.0,
                  max: 5.0,
                  initialValue: 1.5,
                ),
              ],
              renderToCanvas: true,
              colorMap: ColorMap.viridis,
            );

            final jsonMap = widget.pipeline.toJson();
            final restored = GpuComputePipelinePackage.fromJson(jsonMap);
            expect(restored.name, equals('interactive_gain'));
            expect(restored.sliders.length, equals(1));
            expect(widget.toHtml(), contains('interactive_gain'));
            expect(inputTensor.toWebGpuWidget().mimeType, equals('text/html'));
          });
        },
      );
    });

    group('F14: safetensors Serialization & File I/O', () {
      test(
        'F14.1: in-memory saveSafetensors and loadSafetensors round-trip',
        () {
          ResourceScope.scope(() {
            final weights = GpuArray.fromList(
              <double>[0.25, -0.5, 0.75, 1.25],
              [2, 2],
              DType.float32,
            );
            final bias = GpuArray.fromList(
              <double>[10.0, -20.0],
              [2],
              DType.float64,
            );
            final encoded = saveSafetensors({'weights': weights, 'bias': bias});
            final decoded = loadSafetensors(encoded);
            expect(decoded.keys, containsAll(['weights', 'bias']));
            expect(decoded['weights']!.shape, equals([2, 2]));
            expect(decoded['weights']!.dtype, equals(DType.float32));
            _expectCloseList(decoded['weights']!.toList(), weights.toList());
            _expectCloseList(decoded['bias']!.toList(), bias.toList());
          });
        },
      );

      test(
        'F14.2: multi-DType safetensors serialization (integers, bool, floats)',
        () {
          ResourceScope.scope(() {
            final intTensor = GpuArray.fromList(
              <int>[-5, 0, 5, 10],
              [4],
              DType.int32,
            );
            final uintTensor = GpuArray.fromList(
              <int>[1, 2, 3, 255],
              [2, 2],
              DType.uint8,
            );
            final boolTensor = GpuArray.fromList(
              <bool>[true, false],
              [2],
              DType.boolean,
            );
            final bytes = saveSafetensors({
              'ints': intTensor,
              'uints': uintTensor,
              'flags': boolTensor,
            });
            final loaded = loadSafetensors(bytes);
            expect(loaded['ints']!.toList(), equals(<int>[-5, 0, 5, 10]));
            expect(loaded['uints']!.toList(), equals(<int>[1, 2, 3, 255]));
            expect(loaded['flags']!.toList(), equals(<bool>[true, false]));
          });
        },
      );

      test(
        'F14.3: non-contiguous view serialization materializes contiguous bytes',
        () {
          ResourceScope.scope(() {
            final matrix = GpuArray.fromList(
              <double>[1, 2, 3, 4, 5, 6],
              [2, 3],
              DType.float32,
            );
            final transposed = matrix.transpose();
            final bytes = saveSafetensors({'transposed': transposed});
            final loaded = loadSafetensors(bytes);
            expect(loaded['transposed']!.shape, equals([3, 2]));
            _expectCloseList(loaded['transposed']!.toList(), <double>[
              1,
              4,
              2,
              5,
              3,
              6,
            ]);
          });
        },
      );

      test('F14.4: custom metadata map is accepted in saveSafetensors', () {
        ResourceScope.scope(() {
          final tensor = GpuArray.fromList(
            <double>[1.0, 2.0],
            [2],
            DType.float32,
          );
          final bytes = saveSafetensors(
            {'vec': tensor},
            metadata: {'format': 'pt', 'epoch': '12'},
          );
          final loaded = loadSafetensors(bytes);
          _expectCloseList(loaded['vec']!.toList(), <double>[1.0, 2.0]);
        });
      });

      test(
        'F14.5: saveSafetensorsFile and loadSafetensorsFile disk round-trip',
        () async {
          final tempDir = await Directory.systemTemp.createTemp(
            'gpuarray_e2e_',
          );
          final filePath = '${tempDir.path}/checkpoint.safetensors';
          try {
            ResourceScope.scope(() {
              final param = GpuArray.fromList(
                <double>[3.14, 2.71, 1.41, 1.73],
                [2, 2],
                DType.float32,
              );
              saveSafetensorsFile(
                filePath,
                {'param': param},
                metadata: {'author': 'e2e'},
              );
              final restored = loadSafetensorsFile(filePath);
              expect(restored.containsKey('param'), isTrue);
              _expectCloseList(restored['param']!.toList(), <double>[
                3.14,
                2.71,
                1.41,
                1.73,
              ]);
            });
          } finally {
            await tempDir.delete(recursive: true);
          }
        },
      );
    });

    group('F15: linalg Matrix Decompositions', () {
      test(
        'F15.1: svd and svdValues reconstruct original matrix U * S * Vh = A',
        () {
          ResourceScope.scope(() {
            final matrix = GpuArray.fromList(
              <double>[3.0, 1.0, 1.0, 3.0],
              [2, 2],
              DType.float64,
            );
            final decomposition = gpu_linalg.svd(matrix);
            final singularValues = gpu_linalg.svdValues(matrix);
            _expectCloseList(
              decomposition.s.toList(),
              singularValues.toList(),
              tolerance: 1e-4,
            );
            _expectCloseList(singularValues.toList(), <double>[
              4.0,
              2.0,
            ], tolerance: 1e-4);

            final sDiag = diag(decomposition.s);
            final reconstructed = decomposition.u
                .matmul(sDiag)
                .matmul(decomposition.vt);
            _expectCloseList(
              reconstructed.toList(),
              matrix.toList(),
              tolerance: 1e-4,
            );
            decomposition.dispose();
          });
        },
      );

      test('F15.2: qr decomposition satisfies Q * R = A and Q^T * Q = I', () {
        ResourceScope.scope(() {
          final matrix = GpuArray.fromList(
            <double>[12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0],
            [3, 3],
            DType.float64,
          );
          final qrResult = gpu_linalg.qr(
            matrix,
            mode: gpu_linalg.QrMode.reduced,
          );
          final product = qrResult.q.matmul(qrResult.r);
          _expectCloseList(product.toList(), matrix.toList(), tolerance: 1e-4);

          final orthogonality = qrResult.q.transpose().matmul(qrResult.q);
          _expectCloseList(orthogonality.toList(), <double>[
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
          ], tolerance: 1e-4);
          qrResult.dispose();
        });
      });

      test(
        'F15.3: cholesky lower and upper factors reconstruct SPD matrix',
        () {
          ResourceScope.scope(() {
            final spd = GpuArray.fromList(
              <double>[4.0, 2.0, 2.0, 5.0],
              [2, 2],
              DType.float64,
            );
            final lower = gpu_linalg.cholesky(
              spd,
              uplo: gpu_linalg.MatrixTriangle.lower,
            );
            _expectCloseList(
              lower.matmul(lower.transpose()).toList(),
              spd.toList(),
              tolerance: 1e-4,
            );

            final upper = gpu_linalg.cholesky(
              spd,
              uplo: gpu_linalg.MatrixTriangle.upper,
            );
            _expectCloseList(
              upper.transpose().matmul(upper).toList(),
              spd.toList(),
              tolerance: 1e-4,
            );
          });
        },
      );

      test('F15.4: eigh, eigvalsh, eig, and eigvals spectral identities', () {
        ResourceScope.scope(() {
          final sym = GpuArray.fromList(
            <double>[2.0, 1.0, 1.0, 2.0],
            [2, 2],
            DType.float64,
          );
          final eighResult = gpu_linalg.eigh(sym);
          final eigvalshResult = gpu_linalg.eigvalsh(sym);
          _expectCloseList(
            eighResult.eigenvalues.toList(),
            eigvalshResult.toList(),
            tolerance: 1e-4,
          );
          _expectCloseList(eigvalshResult.toList(), <double>[
            1.0,
            3.0,
          ], tolerance: 1e-4);

          final recon = eighResult.eigenvectors
              .matmul(diag(eighResult.eigenvalues))
              .matmul(eighResult.eigenvectors.transpose());
          _expectCloseList(recon.toList(), sym.toList(), tolerance: 1e-4);
          eighResult.dispose();

          final genVals = gpu_linalg.eigvals(sym).toList().cast<Complex>();
          final realParts = genVals.map((c) => c.real).toList()..sort();
          _expectCloseList(realParts, <double>[1.0, 3.0], tolerance: 1e-4);
        });
      });

      test(
        'F15.5: lu, luFactor, and luSolve satisfy P * L * U = A and A * x = b',
        () {
          ResourceScope.scope(() {
            final matrix = GpuArray.fromList(
              <double>[2.0, 1.0, 4.0, 3.0],
              [2, 2],
              DType.float64,
            );
            final luDecomp = gpu_linalg.lu(matrix);
            final plu = luDecomp.p.matmul(luDecomp.l).matmul(luDecomp.u);
            _expectCloseList(plu.toList(), matrix.toList(), tolerance: 1e-4);
            luDecomp.dispose();

            final rhs = GpuArray.fromList(
              <double>[5.0, 11.0],
              [2],
              DType.float64,
            );
            final factored = gpu_linalg.luFactor(matrix);
            final solution = gpu_linalg.luSolve(
              factored.lu,
              factored.pivots,
              rhs,
            );
            _expectCloseList(solution.toList(), <double>[
              2.0,
              1.0,
            ], tolerance: 1e-4);
            factored.dispose();
          });
        },
      );
    });

    group('F16: linalg Solvers, Inverses & Norms', () {
      test('F16.1: solve and inv satisfy A * x = b and A * A^-1 = I', () {
        ResourceScope.scope(() {
          final matrix = GpuArray.fromList(
            <double>[4.0, 7.0, 2.0, 6.0],
            [2, 2],
            DType.float64,
          );
          final rhs = GpuArray.fromList(
            <double>[18.0, 14.0],
            [2],
            DType.float64,
          );
          final xVec = gpu_linalg.solve(matrix, rhs);
          _expectCloseList(
            gpu_linalg.matmul(matrix, xVec).toList(),
            rhs.toList(),
            tolerance: 1e-4,
          );

          final inverse = gpu_linalg.inv(matrix);
          _expectCloseList(matrix.matmul(inverse).toList(), <double>[
            1.0,
            0.0,
            0.0,
            1.0,
          ], tolerance: 1e-4);
        });
      });

      test(
        'F16.2: pinv Moore-Penrose pseudoinverse satisfies A * A^+ * A = A',
        () {
          ResourceScope.scope(() {
            final rect = GpuArray.fromList(
              <double>[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
              [3, 2],
              DType.float64,
            );
            final pseudo = gpu_linalg.pinv(rect);
            expect(pseudo.shape, equals([2, 3]));
            final reconstructed = rect.matmul(pseudo).matmul(rect);
            _expectCloseList(
              reconstructed.toList(),
              rect.toList(),
              tolerance: 1e-4,
            );
          });
        },
      );

      test('F16.3: det and slogdet match analytical determinant', () {
        ResourceScope.scope(() {
          final matrix = GpuArray.fromList(
            <double>[3.0, 8.0, 4.0, 6.0],
            [2, 2],
            DType.float64,
          );
          final detVal = (gpu_linalg.det(matrix).scalar as num).toDouble();
          expect(detVal, closeTo(-14.0, 1e-4));

          final slog = gpu_linalg.slogdet(matrix);
          expect((slog.sign.scalar as num).toDouble(), closeTo(-1.0, 1e-4));
          expect(
            (slog.logabsdet.scalar as num).toDouble(),
            closeTo(2.6390573, 1e-3),
          );
          slog.dispose();
        });
      });

      test('F16.4: matrixPower, matrixRank, and multiDot', () {
        ResourceScope.scope(() {
          final matrix = GpuArray.fromList(
            <double>[1.0, 1.0, 0.0, 1.0],
            [2, 2],
            DType.float64,
          );
          final cubed = gpu_linalg.matrixPower(matrix, 3);
          _expectCloseList(cubed.toList(), <double>[
            1.0,
            3.0,
            0.0,
            1.0,
          ], tolerance: 1e-4);

          final rank = (gpu_linalg.matrixRank(matrix).scalar as num).toInt();
          expect(rank, equals(2));

          final chain = gpu_linalg.multiDot([matrix, matrix, matrix]);
          _expectCloseList(chain.toList(), <double>[
            1.0,
            3.0,
            0.0,
            1.0,
          ], tolerance: 1e-4);
        });
      });

      test('F16.5: norm (Frobenius, L1, L2, Infinity, Nuclear) and cond', () {
        ResourceScope.scope(() {
          final vec = GpuArray.fromList(
            <double>[3.0, -4.0],
            [2],
            DType.float64,
          );
          expect(
            (gpu_linalg.norm(vec, ord: gpu_linalg.NormOrd.l2).scalar as num)
                .toDouble(),
            closeTo(5.0, 1e-4),
          );
          expect(
            (gpu_linalg.norm(vec, ord: gpu_linalg.NormOrd.l1).scalar as num)
                .toDouble(),
            closeTo(7.0, 1e-4),
          );
          expect(
            (gpu_linalg.norm(vec, ord: gpu_linalg.NormOrd.infinity).scalar
                    as num)
                .toDouble(),
            closeTo(4.0, 1e-4),
          );

          final diagMat = GpuArray.fromList(
            <double>[4.0, 0.0, 0.0, 2.0],
            [2, 2],
            DType.float64,
          );
          expect(
            (gpu_linalg.cond(diagMat).scalar as num).toDouble(),
            closeTo(2.0, 1e-4),
          );
        });
      });
    });

    group('F17: linalg Tensor Contractions', () {
      test(
        'F17.1: einsum trace, diagonal, transpose, matmul, batched bilinear',
        () {
          ResourceScope.scope(() {
            final matA = GpuArray.fromList(
              <double>[1, 2, 3, 4],
              [2, 2],
              DType.float32,
            );
            final matB = GpuArray.fromList(
              <double>[5, 6, 7, 8],
              [2, 2],
              DType.float32,
            );

            final traceVal = gpu_linalg.einsum('ii->', [matA]);
            expect((traceVal.scalar as num).toDouble(), closeTo(5.0, 1e-4));

            final diagVec = gpu_linalg.einsum('ii->i', [matA]);
            _expectCloseList(diagVec.toList(), <double>[1.0, 4.0]);

            final product = gpu_linalg.einsum('ij,jk->ik', [matA, matB]);
            _expectCloseList(product.toList(), <double>[19, 22, 43, 50]);
          });
        },
      );

      test('F17.2: tensordot with integer count and explicit axis pairs', () {
        nd.NDArray.scope(() {
          ResourceScope.scope(() {
            final valsA = List<double>.generate(12, (i) => (i + 1).toDouble());
            final valsB = List<double>.generate(12, (i) => i * 0.5);
            final gpuA = GpuArray.fromList(valsA, [2, 3, 2], DType.float32);
            final gpuB = GpuArray.fromList(valsB, [3, 2, 2], DType.float32);
            final hostA = nd.NDArray.fromList(valsA, [
              2,
              3,
              2,
            ], nd.DType.float32);
            final hostB = nd.NDArray.fromList(valsB, [
              3,
              2,
              2,
            ], nd.DType.float32);

            final gpuContracted = gpu_linalg.tensordot(gpuA, gpuB, axes: 2);
            final hostContracted = nd.tensordot(hostA, hostB, axes: 2);
            expect(gpuContracted.shape, equals(hostContracted.shape));
            _expectCloseList(gpuContracted.toList(), hostContracted.toList());
          });
        });
      });

      test('F17.3: kron Kronecker product matches NDArray.kron', () {
        nd.NDArray.scope(() {
          ResourceScope.scope(() {
            final left = GpuArray.fromList(
              <double>[1, 2, 3, 4],
              [2, 2],
              DType.float32,
            );
            final right = GpuArray.fromList(
              <double>[0, 5, 6, 7],
              [2, 2],
              DType.float32,
            );
            final hostLeft = nd.NDArray.fromList(
              <double>[1, 2, 3, 4],
              [2, 2],
              nd.DType.float32,
            );
            final hostRight = nd.NDArray.fromList(
              <double>[0, 5, 6, 7],
              [2, 2],
              nd.DType.float32,
            );

            final gpuKron = gpu_linalg.kron(left, right);
            final hostKron = nd.kron(hostLeft, hostRight);
            expect(gpuKron.shape, equals([4, 4]));
            _expectCloseList(gpuKron.toList(), hostKron.toList());
          });
        });
      });

      test('F17.4: inner and outer products match NDArray', () {
        ResourceScope.scope(() {
          final uVec = GpuArray.fromList(
            <double>[1.0, 2.0, 3.0],
            [3],
            DType.float32,
          );
          final vVec = GpuArray.fromList(
            <double>[4.0, 5.0],
            [2],
            DType.float32,
          );
          final outerMat = gpu_linalg.outer(uVec, vVec);
          expect(outerMat.shape, equals([3, 2]));
          _expectCloseList(outerMat.toList(), <double>[
            4.0,
            5.0,
            8.0,
            10.0,
            12.0,
            15.0,
          ]);

          final innerScalar = gpu_linalg.inner(
            uVec,
            GpuArray.fromList(<double>[2.0, 3.0, 4.0], [3], DType.float32),
          );
          expect((innerScalar.scalar as num).toDouble(), closeTo(20.0, 1e-4));
        });
      });

      test('F17.5: cross 3D vector product satisfies orthogonality', () {
        ResourceScope.scope(() {
          final xAxis = GpuArray.fromList(
            <double>[1.0, 0.0, 0.0],
            [3],
            DType.float32,
          );
          final yAxis = GpuArray.fromList(
            <double>[0.0, 1.0, 0.0],
            [3],
            DType.float32,
          );
          final zAxis = gpu_linalg.cross(xAxis, yAxis);
          _expectCloseList(zAxis.toList(), <double>[0.0, 0.0, 1.0]);
        });
      });
    });

    group('F18: fft Fast Fourier Transforms', () {
      test(
        'F18.1: 1D fft and ifft round-trip and Parseval energy identity',
        () {
          ResourceScope.scope(() {
            final signal = GpuArray.fromList(
              <double>[1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, -1.0],
              [8],
              DType.float64,
            );
            final spectrum = gpu_fft.fft(signal);
            expect(spectrum.shape, equals([8]));
            expect(spectrum.dtype, equals(DType.complex128));

            final recovered = gpu_fft.ifft(spectrum);
            final recoveredReals = recovered
                .toList()
                .cast<Complex>()
                .map((c) => c.real)
                .toList();
            _expectCloseList(recoveredReals, signal.toList(), tolerance: 1e-4);
          });
        },
      );

      test('F18.2: rfft and irfft real-signal round-trip', () {
        ResourceScope.scope(() {
          final realSignal = GpuArray.fromList(
            <double>[1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0],
            [8],
            DType.float64,
          );
          final halfSpectrum = gpu_fft.rfft(realSignal);
          expect(halfSpectrum.shape, equals([5]));

          final reconstructed = gpu_fft.irfft(halfSpectrum, n: 8);
          expect(reconstructed.shape, equals([8]));
          _expectCloseList(
            reconstructed.toList(),
            realSignal.toList(),
            tolerance: 1e-4,
          );
        });
      });

      test('F18.3: 2D fft2 and ifft2 spatial-frequency round-trip', () {
        ResourceScope.scope(() {
          final image = GpuArray.fromList(
            List<double>.generate(16, (i) => (i % 5).toDouble()),
            [4, 4],
            DType.float64,
          );
          final freq2d = gpu_fft.fft2(image);
          expect(freq2d.shape, equals([4, 4]));

          final spatial2d = gpu_fft.ifft2(freq2d);
          final realPixels = spatial2d
              .toList()
              .cast<Complex>()
              .map((c) => c.real)
              .toList();
          _expectCloseList(realPixels, image.toList(), tolerance: 1e-4);
        });
      });

      test(
        'F18.4: FftNorm modes (backward, ortho, forward) scale consistently',
        () {
          ResourceScope.scope(() {
            final signal = GpuArray.fromList(
              <double>[1.0, 2.0, 3.0, 4.0],
              [4],
              DType.float64,
            );
            for (final normMode in gpu_fft.FftNorm.values) {
              final forwardSpec = gpu_fft.fft(signal, norm: normMode);
              final inverseSig = gpu_fft.ifft(forwardSpec, norm: normMode);
              final reals = inverseSig
                  .toList()
                  .cast<Complex>()
                  .map((c) => c.real)
                  .toList();
              _expectCloseList(reals, signal.toList(), tolerance: 1e-4);
            }
          });
        },
      );

      test('F18.5: fftfreq, rfftfreq, fftshift, and ifftshift', () {
        ResourceScope.scope(() {
          final freqs = gpu_fft.fftfreq(4, d: 0.5);
          _expectCloseList(freqs.toList(), <double>[0.0, 0.5, -1.0, -0.5]);

          final rfreqs = gpu_fft.rfftfreq(4, d: 0.5);
          _expectCloseList(rfreqs.toList(), <double>[0.0, 0.5, 1.0]);

          final shifted = gpu_fft.fftshift(freqs);
          _expectCloseList(shifted.toList(), <double>[-1.0, -0.5, 0.0, 0.5]);

          final unshifted = gpu_fft.ifftshift(shifted);
          _expectCloseList(unshifted.toList(), freqs.toList());
        });
      });
    });

    group('F19: random Counter-Based Philox4x32 RNG', () {
      test(
        'F19.1: Philox4x32Engine and RandomState deterministic reproducibility',
        () {
          ResourceScope.scope(() {
            final firstRng = gpu_random.RandomState(12345);
            final secondRng = gpu_random.RandomState(12345);
            final firstSample = firstRng.rand([4, 4]);
            final secondSample = secondRng.rand([4, 4]);
            _expectCloseList(
              firstSample.toList(),
              secondSample.toList(),
              tolerance: 1e-12,
            );
          });
        },
      );

      test('F19.2: uniform and randint respect [low, high) bounds', () {
        ResourceScope.scope(() {
          final rng = gpu_random.RandomState(77);
          final unif = rng.uniform(low: -2.0, high: 5.0, shape: [64]);
          for (final sample in unif.toList().cast<num>()) {
            expect(sample.toDouble(), greaterThanOrEqualTo(-2.0));
            expect(sample.toDouble(), lessThan(5.0));
          }

          final ints = rng.randint(10, 20, [64]);
          for (final sample in ints.toList().cast<int>()) {
            expect(sample, greaterThanOrEqualTo(10));
            expect(sample, lessThan(20));
          }
        });
      });

      test('F19.3: randn, normal, and standardNormal empirical moments', () {
        ResourceScope.scope(() {
          final rng = gpu_random.RandomState(99);
          final gaussian = rng.normal(loc: 3.0, scale: 2.0, shape: [1024]);
          final sampleMean = (gaussian.mean().scalar as num).toDouble();
          expect(sampleMean, closeTo(3.0, 0.25));

          final stdNorm = rng.standardNormal(shape: [256]);
          expect(stdNorm.shape, equals([256]));
        });
      });

      test(
        'F19.4: exponential distribution produces positive samples with expected mean',
        () {
          ResourceScope.scope(() {
            final rng = gpu_random.RandomState(2026);
            final expSamples = rng.exponential(scale: 4.0, shape: [1024]);
            final minSample = (expSamples.min().scalar as num).toDouble();
            final meanSample = (expSamples.mean().scalar as num).toDouble();
            expect(minSample, greaterThanOrEqualTo(0.0));
            expect(meanSample, closeTo(4.0, 0.5));
          });
        },
      );

      test(
        'F19.5: choice, permutation, and shuffle preserve multiset elements',
        () {
          ResourceScope.scope(() {
            final rng = gpu_random.RandomState(555);
            final perm = rng.permutation(8);
            final sortedPerm = List<int>.from(perm.toList().cast<int>())
              ..sort();
            expect(sortedPerm, equals(<int>[0, 1, 2, 3, 4, 5, 6, 7]));

            final candidates = GpuArray.fromList(
              <int>[0, 1, 2, 3, 4, 5],
              [6],
              DType.int64,
            );
            final withoutReplace = rng.choice(
              candidates,
              shape: [6],
              replace: false,
            );
            final sortedChoice = List<int>.from(
              withoutReplace.toList().cast<int>(),
            )..sort();
            expect(sortedChoice, equals(<int>[0, 1, 2, 3, 4, 5]));

            final deck = GpuArray.fromList(
              <int>[10, 20, 30, 40, 50],
              [5],
              DType.int64,
            );
            rng.shuffle(deck);
            final sortedDeck = List<int>.from(deck.toList().cast<int>())
              ..sort();
            expect(sortedDeck, equals(<int>[10, 20, 30, 40, 50]));
          });
        },
      );
    });

    group('F20: autograd Reverse-Mode Automatic Differentiation', () {
      test(
        'F20.1: polynomial and transcendental chain rule matches analytical derivative',
        () {
          ResourceScope.scope(() {
            final xTensor = GpuArray.fromList(
              <double>[1.0, 2.0, 3.0],
              [3],
              DType.float64,
              requiresGrad: true,
            );
            // y = sum(x^2 + 3*x) => dy/dx = 2*x + 3 = [5.0, 7.0, 9.0]
            final yScalar = ((xTensor * xTensor) + (xTensor * 3.0)).sum();
            yScalar.backward();
            expect(xTensor.grad, isNotNull);
            _expectCloseList(xTensor.grad!.toList(), <double>[
              5.0,
              7.0,
              9.0,
            ], tolerance: 1e-4);
          });
        },
      );

      test(
        'F20.2: diamond computation graph accumulates gradients from multiple paths',
        () {
          ResourceScope.scope(() {
            final xLeaf = GpuArray.fromList(
              <double>[2.0, 4.0],
              [2],
              DType.float64,
              requiresGrad: true,
            );
            final branchA = xLeaf * 2.0;
            final branchB = xLeaf * 3.0;
            final combined = (branchA + branchB).sum();
            combined.backward();
            _expectCloseList(xLeaf.grad!.toList(), <double>[5.0, 5.0]);
          });
        },
      );

      test('F20.3: broadcasting backward reduces gradients to leaf shapes', () {
        ResourceScope.scope(() {
          final matrix = GpuArray.fromList(
            <double>[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
            DType.float64,
            requiresGrad: true,
          );
          final bias = GpuArray.fromList(
            <double>[0.5, 1.0, 1.5],
            [1, 3],
            DType.float64,
            requiresGrad: true,
          );
          final loss = (matrix + bias).sum();
          loss.backward();
          expect(bias.grad!.shape, equals([1, 3]));
          _expectCloseList(bias.grad!.toList(), <double>[2.0, 2.0, 2.0]);
        });
      });

      test('F20.4: matmul, transpose, and slice backward passes', () {
        ResourceScope.scope(() {
          final left = GpuArray.fromList(
            <double>[1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
            requiresGrad: true,
          );
          final right = GpuArray.fromList(
            <double>[5.0, 6.0, 7.0, 8.0],
            [2, 2],
            DType.float64,
            requiresGrad: true,
          );
          final prod = left.matmul(right);
          final sliced = prod.slice([Slice(0, 1), Slice.all()]);
          sliced.sum().backward();

          // d(sum(prod[0,:]))/d(left) has row 0 = [5+6, 7+8] = [11, 15], row 1 = [0, 0]
          _expectCloseList(left.grad!.toList(), <double>[
            11.0,
            15.0,
            0.0,
            0.0,
          ], tolerance: 1e-4);
        });
      });

      test(
        'F20.5: noGrad, detach, zeroGrad, and retainGraph control autograd state',
        () {
          ResourceScope.scope(() {
            final xLeaf = GpuArray.fromList(
              <double>[3.0],
              [1],
              DType.float64,
              requiresGrad: true,
            );
            final loss = (xLeaf * xLeaf).sum();
            loss.backward(retainGraph: true);
            _expectCloseList(xLeaf.grad!.toList(), <double>[6.0]);

            xLeaf.zeroGrad();
            expect(xLeaf.grad, isNull);
            loss.backward();
            _expectCloseList(xLeaf.grad!.toList(), <double>[6.0]);

            final detached = xLeaf.detach();
            expect(detached.requiresGrad, isFalse);

            noGrad(() {
              expect(isGradEnabled, isFalse);
              final untracked = xLeaf * 5.0;
              expect(untracked.requiresGrad, isFalse);
            });
            expect(isGradEnabled, isTrue);
          });
        },
      );
    });

    group('F21: nn Modules, Activations, Losses, Optimizers & ResourceScope', () {
      test(
        'F21.1: Linear and Sequential forward and backward parameter gradients',
        () {
          ResourceScope.scope(() {
            gpu_random.seed(42);
            final model = gpu_nn.Sequential([
              gpu_nn.Linear(3, 4),
              gpu_nn.ReLU(),
              gpu_nn.Linear(4, 2),
            ]);
            expect(model.parameters.length, equals(4));

            final batchInput = GpuArray.fromList(
              <double>[1.0, 0.5, -1.0, 2.0, -0.5, 0.25],
              [2, 3],
              DType.float64,
            );
            final out = model.forward(batchInput);
            expect(out.shape, equals([2, 2]));
            out.sum().backward();
            for (final param in model.parameters) {
              expect(param.grad, isNotNull);
            }
            model.zeroGrad();
            for (final param in model.parameters) {
              expect(param.grad, isNull);
            }
          });
        },
      );

      test(
        'F21.2: Conv2d forward and backward propagate gradients to weight, bias, and input',
        () {
          ResourceScope.scope(() {
            gpu_random.seed(101);
            final conv = gpu_nn.Conv2d(1, 2, 2, stride: 1, padding: 0);
            final image = GpuArray.fromList(
              List<double>.generate(9, (i) => (i + 1) * 0.1),
              [1, 1, 3, 3],
              DType.float64,
              requiresGrad: true,
            );
            final featureMap = conv.forward(image);
            expect(featureMap.shape, equals([1, 2, 2, 2]));
            featureMap.sum().backward();
            expect(image.grad, isNotNull);
            expect(conv.weight.grad, isNotNull);
            expect(conv.bias!.grad, isNotNull);
          });
        },
      );

      test(
        'F21.3: LayerNorm and RMSNorm normalize activations along feature dimension',
        () {
          ResourceScope.scope(() {
            final input = GpuArray.fromList(
              <double>[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
              [2, 4],
              DType.float64,
            );
            final layerNorm = gpu_nn.LayerNorm([4]);
            final lnOut = layerNorm.forward(input);
            expect(lnOut.shape, equals([2, 4]));
            final rowMeans = lnOut.mean(axis: 1).toList();
            _expectCloseList(rowMeans, <double>[0.0, 0.0], tolerance: 1e-4);

            final rmsNorm = gpu_nn.RMSNorm([4]);
            final rmsOut = rmsNorm.forward(input);
            expect(rmsOut.shape, equals([2, 4]));
          });
        },
      );

      test('F21.4: Embedding lookup and backward row gradient accumulation', () {
        ResourceScope.scope(() {
          gpu_random.seed(7);
          final embedding = gpu_nn.Embedding(6, 3);
          final tokenIds = GpuArray.fromList(
            <int>[1, 3, 1, 5],
            [2, 2],
            DType.int32,
          );
          final embedded = embedding.forward(tokenIds);
          expect(embedded.shape, equals([2, 2, 3]));
          embedded.sum().backward();
          expect(embedding.weight.grad, isNotNull);
          final weightGrad = embedding.weight.grad!.toNestedList();
          // Row 1 was looked up twice so its gradient is [2, 2, 2]; row 0 was unused [0, 0, 0]
          _expectCloseList((weightGrad[0] as List).cast<Object?>(), <double>[
            0.0,
            0.0,
            0.0,
          ]);
          _expectCloseList((weightGrad[1] as List).cast<Object?>(), <double>[
            2.0,
            2.0,
            2.0,
          ]);
        });
      });

      test('F21.5: Dropout train vs eval mode behavior', () {
        ResourceScope.scope(() {
          gpu_random.seed(11);
          final dropout = gpu_nn.Dropout(p: 0.5);
          final ones = GpuArray.ones([16, 16], DType.float64);
          dropout.eval();
          final evalOut = dropout.forward(ones);
          _expectCloseList(evalOut.toList(), ones.toList());

          dropout.train();
          final trainOut = dropout.forward(ones);
          final values = trainOut.toList().cast<num>();
          expect(values.any((v) => v == 0.0), isTrue);
          expect(values.any((v) => (v - 2.0).abs() < 1e-5), isTrue);
        });
      });

      test(
        'F21.6: functional activations (relu, sigmoid, tanh, gelu, silu, softmax, logSoftmax)',
        () {
          ResourceScope.scope(() {
            final logits = GpuArray.fromList(
              <double>[-1.0, 0.0, 1.0, 2.0],
              [1, 4],
              DType.float64,
            );
            _expectCloseList(gpu_nn.relu(logits).toList(), <double>[
              0.0,
              0.0,
              1.0,
              2.0,
            ]);
            expect(gpu_nn.sigmoid(logits).shape, equals([1, 4]));
            expect(gpu_nn.tanh(logits).shape, equals([1, 4]));
            expect(gpu_nn.gelu(logits).shape, equals([1, 4]));
            expect(gpu_nn.silu(logits).shape, equals([1, 4]));

            final probs = gpu_nn.softmax(logits, axis: -1);
            expect((probs.sum().scalar as num).toDouble(), closeTo(1.0, 1e-5));
            final logProbs = gpu_nn.logSoftmax(logits, axis: -1);
            _expectCloseList(logProbs.exp().toList(), probs.toList());
          });
        },
      );

      test(
        'F21.7: scaledDotProductAttention, MultiheadAttention, RotaryEmbedding, SwiGLU, GeGLU',
        () {
          ResourceScope.scope(() {
            gpu_random.seed(19);
            final mha = gpu_nn.MultiheadAttention(4, 2);
            final tokens = GpuArray.ones([2, 3, 4], DType.float64);
            final attnOut = mha.forward(tokens);
            expect(attnOut.shape, equals([2, 3, 4]));

            final rope = gpu_nn.RotaryEmbedding(4, maxSequenceLength: 8);
            expect(rope.forward(tokens).shape, equals([2, 3, 4]));

            final swiglu = gpu_nn.SwiGLU(4, 8, outFeatures: 4);
            expect(swiglu.forward(tokens).shape, equals([2, 3, 4]));

            final geglu = gpu_nn.GeGLU(4, 8, outFeatures: 4);
            expect(geglu.forward(tokens).shape, equals([2, 3, 4]));
          });
        },
      );

      test(
        'F21.8: TransformerEncoderLayer and TransformerDecoderLayer forward',
        () {
          ResourceScope.scope(() {
            gpu_random.seed(23);
            final encoder = gpu_nn.TransformerEncoderLayer(
              4,
              2,
              dimFeedforward: 8,
              normFirst: true,
            );
            final src = GpuArray.ones([1, 3, 4], DType.float64);
            final memory = encoder.forward(src);
            expect(memory.shape, equals([1, 3, 4]));

            final decoder = gpu_nn.TransformerDecoderLayer(
              4,
              2,
              dimFeedforward: 8,
              normFirst: true,
            );
            final tgt = GpuArray.ones([1, 2, 4], DType.float64);
            final decoded = decoder.forward(tgt, memory: memory);
            expect(decoded.shape, equals([1, 2, 4]));
          });
        },
      );

      test(
        'F21.9: mseLoss and crossEntropy losses with SGD, Adam, and AdamW optimizers',
        () {
          ResourceScope.scope(() {
            final paramSgd = GpuArray.fromList(
              <double>[2.0, -2.0],
              [1, 2],
              DType.float64,
              requiresGrad: true,
            );
            final target = GpuArray.zeros([1, 2], DType.float64);
            final sgd = gpu_nn.SGD([paramSgd], lr: 0.1, momentum: 0.9);
            final lossBefore = (gpu_nn.mseLoss(paramSgd, target).scalar as num)
                .toDouble();
            gpu_nn.mseLoss(paramSgd, target).backward();
            sgd.step();
            sgd.zeroGrad();
            final lossAfter = (gpu_nn.mseLoss(paramSgd, target).scalar as num)
                .toDouble();
            expect(lossAfter, lessThan(lossBefore));

            final logits = GpuArray.fromList(
              <double>[2.0, 0.5, -1.0, -1.0, 3.0, 0.0],
              [2, 3],
              DType.float64,
              requiresGrad: true,
            );
            final labels = GpuArray.fromList(<int>[0, 1], [2], DType.int32);
            final ceLoss = gpu_nn.crossEntropy(logits, labels);
            expect((ceLoss.scalar as num).toDouble(), greaterThan(0.0));
            ceLoss.backward();
            final adamW = gpu_nn.AdamW([logits], lr: 0.05);
            adamW.step();
          });
        },
      );

      test(
        'F21.10: ResourceScope.scope and ResourceScope.returning deterministic cleanup',
        () async {
          final device = await createWebGpuDevice(
            name: 'Scope-Lifecycle-Device',
            enableMemoryPool: false,
          );
          try {
            expect(device.activeBufferCount, equals(0));
            final surviving = ResourceScope.returning(() {
              final tempA = GpuArray.ones(
                [4, 4],
                DType.float32,
                device: device,
              );
              final tempB = GpuArray.ones(
                [4, 4],
                DType.float32,
                device: device,
              );
              final sumAB = tempA + tempB;
              expect(device.activeBufferCount, equals(3));
              return sumAB;
            });
            expect(surviving.isDisposed, isFalse);
            expect(device.activeBufferCount, equals(1));
            surviving.dispose();
            expect(device.activeBufferCount, equals(0));
          } finally {
            device.dispose();
          }
        },
      );

      test(
        'F21.11: R1–R4 sorting/searching/scans, Float32 linalg/fft/nn, CompiledWgslKernel, fused optim & Module lifecycle',
        () async {
          final secondDevice = await createWebGpuDevice(
            name: 'Tier1-Migration-Device',
          );
          try {
            ResourceScope.scope(() {
              // 1. Int64 sorting, searching, uniqueAll, bincount, cumsum, cumprod, diff
              final arr = GpuArray<Int32>.fromList(
                [3, 1, 2, 1],
                [4],
                DType.int32,
              );
              expect(sort(arr).toList(), equals([1, 1, 2, 3]));
              expect(argsort(arr).dtype, equals(DType.int64));
              final tk = topk(arr, 2);
              expect(tk.values.toList(), equals([3, 2]));
              expect(tk.indices.dtype, equals(DType.int64));
              final uAll = uniqueAll(arr);
              expect(uAll.values.toList(), equals([1, 2, 3]));
              expect(uAll.counts.toList(), equals([2, 1, 1]));
              expect(bincount(arr).toList(), equals([0, 2, 1, 1]));
              expect(cumsum(arr).toList(), equals([3, 4, 6, 7]));
              expect(cumprod(arr).toList(), equals([3, 3, 6, 6]));
              expect(diff(arr).toList(), equals([-2, 1, -1]));

              // 2. Float32 linalg & fft
              final spd32 = GpuArray<Float32>.fromList(
                [4.0, 1.0, 1.0, 3.0],
                [2, 2],
                DType.float32,
              );
              final rhs32 = GpuArray<Float32>.fromList(
                [1.0, 2.0],
                [2],
                DType.float32,
              );
              final GpuArray<Float32> sol32 = gpu_linalg.solve(spd32, rhs32);
              final GpuArray<Float32> chol32 = gpu_linalg.cholesky(spd32);
              final GpuArray<Complex64> rspec32 = gpu_fft.rfft(spd32);
              final GpuArray<Float32> irrec32 = gpu_fft.irfft(rspec32, n: 2);
              expect(sol32.dtype, equals(DType.float32));
              expect(chol32.dtype, equals(DType.float32));
              expect(rspec32.dtype, equals(DType.complex64));
              expect(irrec32.dtype, equals(DType.float32));

              // 3. CompiledWgslKernel Map<String, GpuArray> execution
              final xExpr = Expr.variable('x', bindingIndex: 0);
              final bExpr = Expr.variable('b', bindingIndex: 1);
              final alphaExpr = Expr.scalar('alpha');
              final kernel = GpuDevice.defaultDevice.jitCompiler.compileKernel(
                (xExpr * alphaExpr) + bExpr,
              );
              final jitOut = kernel.execute<Float32>(
                {'x': rhs32, 'b': rhs32},
                scalars: const {'alpha': 2.0},
              );
              _expectCloseList(jitOut.toList(), <double>[3.0, 6.0]);

              // 4. Float32 Module, fused activations, clipGradNorm/Value, SafeTensors & Module.to/dispose
              final mlp = gpu_nn.Sequential([
                gpu_nn.Linear(2, 4, dtype: DType.float32),
                gpu_nn.SiLU(),
                gpu_nn.Linear(4, 2, dtype: DType.float32),
              ]);
              final xIn = GpuArray<Float32>.fromList(
                [1.0, 2.0],
                [1, 2],
                DType.float32,
              );
              final target = GpuArray<Float32>.zeros([1, 2], DType.float32);
              final GpuArray<Float32> pred = mlp.forward(xIn);
              expect(pred.dtype, equals(DType.float32));
              final loss = gpu_nn.mseLoss(pred, target);
              loss.backward();
              final totalNorm = gpu_nn.clipGradNorm(mlp.parameters, 1.0);
              expect(totalNorm, greaterThan(0.0));
              gpu_nn.clipGradValue(mlp.parameters, 0.5);

              final opt = gpu_nn.AdamW(mlp.parameters, lr: 0.01);
              final beforeCount = GpuDevice.defaultDevice.activeBufferCount;
              opt.step();
              // Optimizer lazily allocates 2 state buffers per parameter (4 params * 2 = 8) and 0 leaked intermediates
              expect(
                GpuDevice.defaultDevice.activeBufferCount,
                lessThanOrEqualTo(beforeCount + 8),
              );
              final secondStepCount = GpuDevice.defaultDevice.activeBufferCount;
              opt.step();
              expect(
                GpuDevice.defaultDevice.activeBufferCount,
                equals(secondStepCount),
              );

              final ckptBytes = mlp.saveToSafetensors();
              final mlp2 = gpu_nn.Sequential([
                gpu_nn.Linear(2, 4, dtype: DType.float32),
                gpu_nn.SiLU(),
                gpu_nn.Linear(4, 2, dtype: DType.float32),
              ]);
              mlp2.loadFromSafetensors(ckptBytes);
              _expectCloseList(
                mlp2.forward(xIn).toList(),
                mlp.forward(xIn).toList(),
              );

              mlp2.to(secondDevice);
              expect(mlp2.parameters.first.device, same(secondDevice));
              mlp2.dispose();
              expect(mlp2.isDisposed, isTrue);
            });
          } finally {
            secondDevice.dispose();
          }
        },
      );
    });
  });
}
