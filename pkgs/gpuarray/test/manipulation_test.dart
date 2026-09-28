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

import 'package:test/test.dart';
import 'package:gpuarray/gpuarray.dart';
import 'package:resource_scope/resource_scope.dart';

void main() {
  group('GpuArray Tensor Manipulation & Assembly', () {
    test(
      'Concatenation & Stacking (concatenate, stack, vstack, hstack, dstack, column_stack)',
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

          // Concatenate along axis 0
          final cat0 = concatenate([a, b], axis: 0);
          expect(cat0.shape, equals([4, 2]));
          expect(
            cat0.toList(),
            equals([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
          );

          // Concatenate along axis 1
          final cat1 = concatenate([a, b], axis: 1);
          expect(cat1.shape, equals([2, 4]));
          expect(
            cat1.toList(),
            equals([1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]),
          );

          // Stack along new axis
          final st = stack([a, b], axis: 0);
          expect(st.shape, equals([2, 2, 2]));

          // vstack, hstack, dstack, column_stack
          final v = vstack([a, b]);
          expect(v.shape, equals([4, 2]));

          final h = hstack([a, b]);
          expect(h.shape, equals([2, 4]));

          final d = dstack([a, b]);
          expect(d.shape, equals([2, 2, 2]));

          final col = column_stack([a, b]);
          expect(col.shape, equals([2, 4]));
        });
      },
    );

    test('Splitting (split, array_split, hsplit, vsplit, dsplit)', () {
      ResourceScope.scope(() {
        final a = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
          [2, 4],
          DType.float64,
        );

        // Equal split
        final parts = split(a, 2, axis: 1);
        expect(parts.length, equals(2));
        expect(parts[0].shape, equals([2, 2]));
        expect(parts[0].toList(), equals([1.0, 2.0, 5.0, 6.0]));
        expect(parts[1].shape, equals([2, 2]));
        expect(parts[1].toList(), equals([3.0, 4.0, 7.0, 8.0]));

        // Unequal array_split
        final arrParts = array_split(a, 3, axis: 1);
        expect(arrParts.length, equals(3));
        expect(arrParts[0].shape, equals([2, 2]));
        expect(arrParts[1].shape, equals([2, 1]));
        expect(arrParts[2].shape, equals([2, 1]));

        // vsplit and hsplit
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

        // Tile 1D
        final tiled = tile(a, [2]);
        expect(tiled.shape, equals([6]));
        expect(tiled.toList(), equals([1.0, 2.0, 3.0, 1.0, 2.0, 3.0]));

        // Repeat flat
        final rep = repeat(a, 2);
        expect(rep.shape, equals([6]));
        expect(rep.toList(), equals([1.0, 1.0, 2.0, 2.0, 3.0, 3.0]));

        // Repeat along axis on 2D
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

    test('Padding, Rolling & Flipping (pad, roll, flip, rot90)', () {
      ResourceScope.scope(() {
        final a = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );

        // Pad constant
        final padded = pad(a, [
          [1, 1],
          [1, 1],
        ], constantValues: 0.0);
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

        // Roll
        final rolled = roll(a, 1, axis: 0);
        expect(rolled.toList(), equals([3.0, 4.0, 1.0, 2.0]));

        // Flip
        final flipped = flip(a, axis: 0);
        expect(flipped.toList(), equals([3.0, 4.0, 1.0, 2.0]));

        // Rot90
        final rotated = rot90(a);
        expect(rotated.toList(), equals([2.0, 4.0, 1.0, 3.0]));
      });
    });

    test('Diagonals & Triangular (diag, diagonal, trace, triu, tril)', () {
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
        expect(tr, equals(15.0));

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
    });

    test(
      'Axis Permutations & Shape (moveaxis, swapaxes, expand_dims, broadcast_to)',
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

          final exp = expand_dims(a, 1);
          expect(exp.shape, equals([2, 1, 3]));

          final bcast = broadcast_to(a, [2, 2, 3]);
          expect(bcast.shape, equals([2, 2, 3]));
        });
      },
    );
  });
}
