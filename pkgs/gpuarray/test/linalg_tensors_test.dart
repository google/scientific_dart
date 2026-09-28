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
import 'package:gpuarray/linalg.dart' as linalg;
import 'package:resource_scope/resource_scope.dart';

void main() {
  group('GpuArray Tensor Contractions & Einstein Summation (gpuarray.linalg)', () {
    test('Einstein summation (einsum)', () {
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

        // 1. Matrix multiplication 'ij,jk->ik'
        final c = linalg.einsum('ij,jk->ik', [a, b]);
        expect(c.shape, equals([2, 2]));
        final cList = c.toList().cast<double>();
        expect(cList[0], closeTo(19.0, 1e-4));
        expect(cList[1], closeTo(22.0, 1e-4));
        expect(cList[2], closeTo(43.0, 1e-4));
        expect(cList[3], closeTo(50.0, 1e-4));

        // 2. Matrix transpose 'ij->ji'
        final aT = linalg.einsum('ij->ji', [a]);
        expect(aT.shape, equals([2, 2]));
        expect(aT.toList(), equals([1.0, 3.0, 2.0, 4.0]));

        // 3. Matrix trace 'ii->'
        final tr = linalg.einsum('ii->', [a]);
        expect(tr.shape, equals([]));
        expect(tr.item(), closeTo(5.0, 1e-4));

        // 4. Outer product 'i,j->ij'
        final v1 = GpuArray.fromList([1.0, 2.0], [2], DType.float64);
        final v2 = GpuArray.fromList([3.0, 4.0], [2], DType.float64);
        final outer = linalg.einsum('i,j->ij', [v1, v2]);
        expect(outer.shape, equals([2, 2]));
        expect(outer.toList(), equals([3.0, 4.0, 6.0, 8.0]));

        // 5. Undefined output subscripts throw ArgumentError
        expect(() => linalg.einsum('ij,jk->il', [a, b]), throwsArgumentError);
        expect(() => linalg.einsum('ij->k', [a]), throwsArgumentError);
      });
    });

    test('einsum with non-contiguous transposed operands', () {
      ResourceScope.scope(() {
        final a = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [2, 3],
          DType.float64,
        );
        final aT = a.transpose([
          1,
          0,
        ]); // Shape [3, 2], non-contiguous strides [1, 3]
        final b = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );

        // einsum 'ij,jk->ik' with non-contiguous aT
        final c = linalg.einsum('ij,jk->ik', [aT, b]);
        expect(c.shape, equals([3, 2]));

        // Expected: aT @ b
        // aT = [[1, 4],
        //       [2, 5],
        //       [3, 6]]
        // b = [[1, 2],
        //      [3, 4]]
        // row0: 1*1 + 4*3 = 13, 1*2 + 4*4 = 18
        // row1: 2*1 + 5*3 = 17, 2*2 + 5*4 = 24
        // row2: 3*1 + 6*3 = 21, 3*2 + 6*4 = 30
        final cList = c.toList().cast<double>();
        expect(cList, equals([13.0, 18.0, 17.0, 24.0, 21.0, 30.0]));
      });
    });

    test(
      'Tensor contractions (tensordot, kron, inner, outer, cross, multi_dot)',
      () {
        ResourceScope.scope(() {
          // tensordot
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
          final td = linalg.tensordot(a, b, axes: 1);
          expect(td.shape, equals([2, 2]));
          expect(td.toList(), equals([19.0, 22.0, 43.0, 50.0]));

          // kron (Kronecker product)
          final k = linalg.kron(a, b);
          expect(k.shape, equals([4, 4]));
          expect(
            k.toList(),
            equals([
              5.0,
              6.0,
              10.0,
              12.0,
              7.0,
              8.0,
              14.0,
              16.0,
              15.0,
              18.0,
              20.0,
              24.0,
              21.0,
              24.0,
              28.0,
              32.0,
            ]),
          );

          // inner and outer
          final v1 = GpuArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
          final v2 = GpuArray.fromList([4.0, 5.0, 6.0], [3], DType.float64);

          final inn = linalg.inner(v1, v2);
          expect(inn.shape, equals([]));
          expect(inn.item(), equals(32.0));

          final out = linalg.outer(v1, v2);
          expect(out.shape, equals([3, 3]));
          expect(
            out.toList(),
            equals([4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 12.0, 15.0, 18.0]),
          );

          // cross product
          final u = GpuArray.fromList([1.0, 0.0, 0.0], [3], DType.float64);
          final v = GpuArray.fromList([0.0, 1.0, 0.0], [3], DType.float64);
          final uxv = linalg.cross(u, v);
          expect(uxv.toList(), equals([0.0, 0.0, 1.0]));

          // multi_dot
          final md = linalg.multi_dot([a, b, a]);
          expect(md.shape, equals([2, 2]));
        });
      },
    );

    test('cross product broadcasting and arbitrary vector axes', () {
      ResourceScope.scope(() {
        // 1. Broadcasting: [2, 3] with [3] -> [2, 3]
        final a = GpuArray.fromList(
          [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
          [2, 3],
          DType.float64,
        );
        final b = GpuArray.fromList([0.0, 0.0, 1.0], [3], DType.float64);

        final axb = linalg.cross(a, b);
        expect(axb.shape, equals([2, 3]));
        final axbList = axb.toList().cast<double>();
        // [1, 0, 0] x [0, 0, 1] = [0, -1, 0]
        // [0, 1, 0] x [0, 0, 1] = [1, 0, 0]
        expect(axbList, equals([0.0, -1.0, 0.0, 1.0, 0.0, 0.0]));

        // 2. Custom axes: axis 0 on [3, 2]
        final aAx0 = GpuArray.fromList(
          [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
          [3, 2],
          DType.float64,
        );
        final bAx0 = GpuArray.fromList(
          [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
          [3, 2],
          DType.float64,
        );
        final cAx0 = linalg.cross(aAx0, bAx0, axisa: 0, axisb: 0, axisc: 0);
        expect(cAx0.shape, equals([3, 2]));
        final cAx0List = cAx0.toList().cast<double>();
        // Col 0: [1, 0, 0] x [0, 0, 1] = [0, -1, 0]
        // Col 1: [0, 1, 0] x [0, 0, 1] = [1, 0, 0]
        // In row-major [3, 2]:
        // row 0: [0, 1]
        // row 1: [-1, 0]
        // row 2: [0, 0]
        expect(cAx0List, equals([0.0, 1.0, -1.0, 0.0, 0.0, 0.0]));

        // 3. Batch broadcasting: [2, 1, 3] and [1, 2, 3] -> [2, 2, 3]
        final a3d = GpuArray.fromList(
          [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
          [2, 1, 3],
          DType.float64,
        );
        final b3d = GpuArray.fromList(
          [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
          [1, 2, 3],
          DType.float64,
        );
        final c3d = linalg.cross(a3d, b3d);
        expect(c3d.shape, equals([2, 2, 3]));
      });
    });

    test('multi_dot with 1D vectors and multi-matrix chains', () {
      ResourceScope.scope(() {
        final v1 = GpuArray.fromList([1.0, 2.0], [2], DType.float64);
        final a = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [2, 3],
          DType.float64,
        );
        final b = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [3, 2],
          DType.float64,
        );
        final v2 = GpuArray.fromList([3.0, 4.0], [2], DType.float64);

        // 1. 1D at start: [v1 (2), a (2, 3), b (3, 2)] -> 1D vector (2)
        // a @ b = [[22, 28], [49, 64]]
        // v1 @ (a @ b) = [1, 2] @ [[22, 28], [49, 64]] = [1*22 + 2*49, 1*28 + 2*64] = [120, 156]
        final mdStart = linalg.multi_dot([v1, a, b]);
        expect(mdStart.shape, equals([2]));
        expect(mdStart.toList(), equals([120.0, 156.0]));

        // 2. 1D at end: [a (2, 3), b (3, 2), v2 (2)] -> 1D vector (2)
        // (a @ b) @ [3, 4] = [[22, 28], [49, 64]] @ [3, 4] = [22*3 + 28*4, 49*3 + 64*4] = [178, 403]
        final mdEnd = linalg.multi_dot([a, b, v2]);
        expect(mdEnd.shape, equals([2]));
        expect(mdEnd.toList(), equals([178.0, 403.0]));

        // 3. 1D at both ends: [v1 (2), a (2, 3), b (3, 2), v2 (2)] -> 0-D scalar
        // [120, 156] @ [3, 4] = 120*3 + 156*4 = 360 + 624 = 984
        final mdBoth = linalg.multi_dot([v1, a, b, v2]);
        expect(mdBoth.shape, equals([]));
        expect(mdBoth.item(), closeTo(984.0, 1e-4));

        // 4. Two operands with 1D vector:
        // [v1 (2), a (2, 3)] -> (3)
        // [1, 2] @ [[1, 2, 3], [4, 5, 6]] = [9, 12, 15]
        final md2_1 = linalg.multi_dot([v1, a]);
        expect(md2_1.shape, equals([3]));
        expect(md2_1.toList(), equals([9.0, 12.0, 15.0]));

        // [b (3, 2), v2 (2)] -> (3)
        // [[1, 2], [3, 4], [5, 6]] @ [3, 4] = [11, 25, 39]
        final md2_2 = linalg.multi_dot([b, v2]);
        expect(md2_2.shape, equals([3]));
        expect(md2_2.toList(), equals([11.0, 25.0, 39.0]));

        // [v1 (2), v2 (2)] -> scalar
        // [1, 2] @ [3, 4] = 11
        final md2_3 = linalg.multi_dot([v1, v2]);
        expect(md2_3.shape, equals([]));
        expect(md2_3.item(), closeTo(11.0, 1e-4));
      });
    });
  });
}
