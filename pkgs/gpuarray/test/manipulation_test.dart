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

import 'package:gpuarray/gpuarray.dart';
import 'package:resource_scope/resource_scope.dart';
import 'package:test/test.dart';

void main() {
  group('GpuArray Tensor Manipulation & Assembly', () {
    test(
      'Concatenation & Stacking (concatenate, stack, vstack, hstack, dstack, columnStack)',
      () {
        ResourceScope.scope(() {
          final a = GpuArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final b = GpuArray.fromList(
            [5.0, 6.0, 7.0, 8.0],
            [2, 2],
            DType.float64,
          );

          final cat0 = concatenate<Float64>([a, b], axis: 0);
          expect(cat0.shape, equals([4, 2]));
          expect(
            cat0.toList(),
            equals([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
          );

          final outCat = GpuArray.empty([2, 4], DType.float64);
          final cat1 = concatenate<Float64>([a, b], axis: 1, out: outCat);
          expect(identical(cat1, outCat), isTrue);
          expect(cat1.shape, equals([2, 4]));
          expect(
            cat1.toList(),
            equals([1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]),
          );

          final st = stack<Float64>([a, b], axis: 0);
          expect(st.shape, equals([2, 2, 2]));

          final v = vstack<Float64>([a, b]);
          expect(v.shape, equals([4, 2]));

          final h = hstack<Float64>([a, b]);
          expect(h.shape, equals([2, 4]));

          final d = dstack<Float64>([a, b]);
          expect(d.shape, equals([2, 2, 2]));

          final col = columnStack<Float64>([a, b]);
          expect(col.shape, equals([2, 4]));
        });
      },
    );

    test('Splitting (split, arraySplit, hsplit, vsplit, dsplit)', () {
      ResourceScope.scope(() {
        final a = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
          [2, 4],
          DType.float64,
        );

        final parts = split(a, 2, axis: 1);
        expect(parts.length, equals(2));
        expect(parts[0].shape, equals([2, 2]));
        expect(parts[0].toList(), equals([1.0, 2.0, 5.0, 6.0]));
        expect(parts[1].shape, equals([2, 2]));
        expect(parts[1].toList(), equals([3.0, 4.0, 7.0, 8.0]));

        final arrParts = arraySplit(a, 3, axis: 1);
        expect(arrParts.length, equals(3));
        expect(arrParts[0].shape, equals([2, 2]));
        expect(arrParts[1].shape, equals([2, 1]));
        expect(arrParts[2].shape, equals([2, 1]));

        final vparts = vsplit(a, 2);
        expect(vparts.length, equals(2));
        expect(vparts[0].shape, equals([1, 4]));

        final hparts = hsplit(a, 2);
        expect(hparts.length, equals(2));
        expect(hparts[0].shape, equals([2, 2]));
      });
    });

    test('Tiling & Repetition (tile, repeat)', () {
      ResourceScope.scope(() {
        final a = GpuArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);

        final tiled = tile(a, [2]);
        expect(tiled.shape, equals([6]));
        expect(tiled.toList(), equals([1.0, 2.0, 3.0, 1.0, 2.0, 3.0]));

        final rep = repeat(a, 2);
        expect(rep.shape, equals([6]));
        expect(rep.toList(), equals([1.0, 1.0, 2.0, 2.0, 3.0, 3.0]));

        final a2 = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        final repAx0 = a2.repeat(2, axis: 0);
        expect(repAx0.shape, equals([4, 2]));
        expect(
          repAx0.toList(),
          equals([1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]),
        );
      });
    });

    test(
      'Padding (constant, edge, reflect, symmetric, wrap), Rolling & Flipping',
      () {
        ResourceScope.scope(() {
          final a = GpuArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );

          final padded = pad(
            a,
            [
              [1, 1],
              [1, 1],
            ],
            mode: PadMode.constant,
            constantValues: 0.0,
          );
          expect(padded.shape, equals([4, 4]));
          expect(
            padded.toList(),
            equals([
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              1.0,
              2.0,
              0.0,
              0.0,
              3.0,
              4.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
            ]),
          );

          final v1d = GpuArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
          expect(
            pad(v1d, [
              [2, 2],
            ], mode: PadMode.edge).toList(),
            equals([1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0]),
          );
          expect(
            pad(v1d, [
              [2, 2],
            ], mode: PadMode.reflect).toList(),
            equals([3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0]),
          );
          expect(
            pad(v1d, [
              [2, 2],
            ], mode: PadMode.symmetric).toList(),
            equals([2.0, 1.0, 1.0, 2.0, 3.0, 3.0, 2.0]),
          );
          expect(
            pad(v1d, [
              [2, 2],
            ], mode: PadMode.wrap).toList(),
            equals([2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0]),
          );

          final rolled = roll(a, 1, axis: 0);
          expect(rolled.toList(), equals([3.0, 4.0, 1.0, 2.0]));

          final flipped = flip(a, axis: 0);
          expect(flipped.toList(), equals([3.0, 4.0, 1.0, 2.0]));
          expect(flipud(a).toList(), equals([3.0, 4.0, 1.0, 2.0]));
          expect(fliplr(a).toList(), equals([2.0, 1.0, 4.0, 3.0]));
          expect(ravel(a).toList(), equals([1.0, 2.0, 3.0, 4.0]));

          final rotated = rot90(a);
          expect(rotated.toList(), equals([2.0, 4.0, 1.0, 3.0]));
        });
      },
    );

    test(
      'Diagonals & Triangular (diag, diagonal, trace, triu, tril) including >2D',
      () {
        ResourceScope.scope(() {
          final v = GpuArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
          final diagMat = diag(v);
          expect(diagMat.shape, equals([3, 3]));
          expect(
            diagMat.toList(),
            equals([1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]),
          );

          final a = GpuArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            [3, 3],
            DType.float64,
          );

          final d = diagonal(a);
          expect(d.toList(), equals([1.0, 5.0, 9.0]));

          final tr = trace(a);
          expect(tr.shape, equals(const <int>[]));
          expect(tr.scalar, equals(15.0));

          // 3D tensor diagonal & trace
          final a3d = GpuArray.fromList(
            List<double>.generate(18, (i) => (i + 1).toDouble()),
            [2, 3, 3],
            DType.float64,
          );
          final d3d = diagonal(a3d, axis1: 1, axis2: 2);
          expect(d3d.shape, equals([2, 3]));
          expect(d3d.toList(), equals([1.0, 5.0, 9.0, 10.0, 14.0, 18.0]));

          final tr3d = trace(a3d, axis1: 1, axis2: 2);
          expect(tr3d.shape, equals([2]));
          expect(tr3d.toList(), equals([15.0, 42.0]));

          final u = triu(a);
          expect(
            u.toList(),
            equals([1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0]),
          );

          final l = tril(a);
          expect(
            l.toList(),
            equals([1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0]),
          );
        });
      },
    );

    test(
      'Axis Permutations & Shape (moveaxis, swapaxes, expandDims, broadcastTo, broadcastArrays)',
      () {
        ResourceScope.scope(() {
          final a = GpuArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
            DType.float64,
          );

          final swapped = swapaxes(a, 0, 1);
          expect(swapped.shape, equals([3, 2]));
          expect(swapped.toList(), equals([1.0, 4.0, 2.0, 5.0, 3.0, 6.0]));

          final moved = moveaxis(a, 0, 1);
          expect(moved.shape, equals([3, 2]));

          final exp = expandDims(a, 1);
          expect(exp.shape, equals([2, 1, 3]));

          final bcast = broadcastTo(a, [2, 2, 3]);
          expect(bcast.shape, equals([2, 2, 3]));

          final b = GpuArray.fromList([10.0, 20.0, 30.0], [3], DType.float64);
          final bcastPair = broadcastArrays([a, b]);
          expect(bcastPair.length, equals(2));
          expect(bcastPair[0].shape, equals([2, 3]));
          expect(bcastPair[1].shape, equals([2, 3]));
        });
      },
    );

    test('Rejects broadcasted view as out destination across operations', () {
      ResourceScope.scope(() {
        final base1D = GpuArray<Float32>.fromList([1.0], [1], DType.float32);
        final broadcasted1D = base1D.broadcastTo([4]);

        final base2D = GpuArray<Float32>.fromList([1.0], [1, 1], DType.float32);
        final broadcasted2D = base2D.broadcastTo([2, 2]);

        final in1D = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [4],
          DType.float32,
        );
        final in2D = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float32,
        );

        expect(
          () => roll(in1D, 1, out: broadcasted1D),
          throwsA(
            isA<ArgumentError>().having(
              (e) => e.message,
              'message',
              contains('Must be writeable and not a broadcasted view.'),
            ),
          ),
        );

        expect(
          () => triu(in2D, out: broadcasted2D),
          throwsA(
            isA<ArgumentError>().having(
              (e) => e.message,
              'message',
              contains('Must be writeable and not a broadcasted view.'),
            ),
          ),
        );

        expect(
          () => tril(in2D, out: broadcasted2D),
          throwsA(
            isA<ArgumentError>().having(
              (e) => e.message,
              'message',
              contains('Must be writeable and not a broadcasted view.'),
            ),
          ),
        );

        expect(
          () => pad(in1D, [
            [0, 0],
          ], out: broadcasted1D),
          throwsA(
            isA<ArgumentError>().having(
              (e) => e.message,
              'message',
              contains('Must be writeable and not a broadcasted view.'),
            ),
          ),
        );

        final tileIn = GpuArray.fromList([1.0, 2.0], [2], DType.float32);
        expect(
          () => tile(tileIn, [2], out: broadcasted1D),
          throwsA(
            isA<ArgumentError>().having(
              (e) => e.message,
              'message',
              contains('Must be writeable and not a broadcasted view.'),
            ),
          ),
        );

        final catA = GpuArray.fromList([1.0, 2.0], [2], DType.float32);
        final catB = GpuArray.fromList([3.0, 4.0], [2], DType.float32);
        expect(
          () => concatenate([catA, catB], out: broadcasted1D),
          throwsA(
            isA<ArgumentError>().having(
              (e) => e.message,
              'message',
              contains('Must be writeable and not a broadcasted view.'),
            ),
          ),
        );
      });
    });
  });
}
