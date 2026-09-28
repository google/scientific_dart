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

import 'package:criterion/criterion.dart';
import 'package:ndarray/ndarray.dart';

void main() async {
  await criterion(
    'Dart Scientific NDArray Operations Benchmark Suite',
    (c) {
      c.group('Einsum Operations', () {
        final aList = List.generate(10000, (i) => i.toDouble());
        final bList = List.generate(10000, (i) => i.toDouble());
        final aMat = NDArray<Float64>.fromList(aList, [
          100,
          100,
        ], DType.float64);
        final bMat = NDArray<Float64>.fromList(bList, [
          100,
          100,
        ], DType.float64);
        final subs2d = EinsumSubscripts.parse('ij,jk->ik');
        final outMat = NDArray<Float64>.create([100, 100], DType.float64);

        c.bench('einsum matrix mult (\'ij,jk->ik\') [100x100]', () {
          final res = einsum(subs2d, [aMat, bMat]);
          blackhole(res);
          res.dispose();
        });

        c.bench('einsum matrix mult with out: [100x100]', () {
          einsum(subs2d, [aMat, bMat], out: outMat);
        });

        final a3dList = List.generate(4000, (i) => i.toDouble());
        final b3dList = List.generate(4000, (i) => i.toDouble());
        final a3d = NDArray<Float64>.fromList(a3dList, [
          10,
          20,
          20,
        ], DType.float64);
        final b3d = NDArray<Float64>.fromList(b3dList, [
          10,
          20,
          20,
        ], DType.float64);
        final subs3d = EinsumSubscripts.parse('...ij,...jk->...ik');
        final out3d = NDArray<Float64>.create([10, 20, 20], DType.float64);

        c.bench('einsum batch matmul (\'...ij,...jk->...ik\') [10x20x20]', () {
          final res = einsum(subs3d, [a3d, b3d]);
          blackhole(res);
          res.dispose();
        });

        c.bench('einsum batch matmul with out: [10x20x20]', () {
          einsum(subs3d, [a3d, b3d], out: out3d);
        });

        final subs3Op = EinsumSubscripts.parse('ij,jk,kl->il');
        final cMat = NDArray<Float64>.fromList(bList, [
          100,
          100,
        ], DType.float64);
        c.bench('einsum 3-operand (\'ij,jk,kl->il\') [100x100]', () {
          final res = einsum(subs3Op, [aMat, bMat, cMat]);
          blackhole(res);
          res.dispose();
        });
      });

      c.group('Tensordot Operations', () {
        final aList = List.generate(10000, (i) => i.toDouble());
        final bList = List.generate(10000, (i) => i.toDouble());
        final aMat = NDArray<Float64>.fromList(aList, [
          100,
          100,
        ], DType.float64);
        final bMat = NDArray<Float64>.fromList(bList, [
          100,
          100,
        ], DType.float64);

        final axesCount2 = const TensordotAxes.count(2);
        c.bench('tensordot axes=2 [100x100]', () {
          final res = tensordot(aMat, bMat, axes: axesCount2);
          blackhole(res);
          res.dispose();
        });

        final axesExplicit = TensordotAxes.explicit([1], [0]);
        c.bench('tensordot axes=([1],[0]) [100x100]', () {
          final res = tensordot(aMat, bMat, axes: axesExplicit);
          blackhole(res);
          res.dispose();
        });
      });

      c.group('Convolution & Correlation', () {
        final a1dList = List.generate(10000, (i) => i.toDouble());
        final v1dList = List.generate(100, (i) => i.toDouble());
        final a1d = NDArray<Float64>.fromList(a1dList, [10000], DType.float64);
        final v1d = NDArray<Float64>.fromList(v1dList, [100], DType.float64);

        c.bench('correlate mode=\'full\' [N=10,000, K=100]', () {
          final res = correlate(a1d, v1d, mode: ConvMode.full);
          blackhole(res);
          res.dispose();
        });

        c.bench('convolve mode=\'full\' [N=10,000, K=100]', () {
          final res = convolve(a1d, v1d, mode: ConvMode.full);
          blackhole(res);
          res.dispose();
        });
      });

      c.group('Solvers & Optimization', () {
        c.bench('brentq root scalar [x^2 - 4]', () {
          final r = brentq((x) => x * x - 4.0, 0.0, 3.0);
          blackhole(r);
        });

        c.bench('newton root scalar [x^2 - 2]', () {
          final r = newton((x) => x * x - 2.0, 1.0, fprime: (x) => 2.0 * x);
          blackhole(r);
        });

        final x0Rosen = NDArray<Float64>.fromList(
          [-1.2, 1.0],
          [2],
          DType.float64,
        );
        c.bench('nelder_mead minimization [Rosenbrock 2D]', () {
          final r = nelder_mead(
            (x) {
              final px = x.getCell([0]).toDouble();
              final py = x.getCell([1]).toDouble();
              return 100.0 * math.pow(py - px * px, 2).toDouble() +
                  math.pow(1.0 - px, 2).toDouble();
            },
            x0Rosen,
            maxiter: 1000,
          );
          blackhole(r);
          r.x.dispose();
        });

        final x0Bowl = NDArray<Float64>.fromList(
          [5.0, 5.0],
          [2],
          DType.float64,
        );
        c.bench('lbfgs minimization [Quadratic 2D]', () {
          final r = lbfgs((x) {
            final px = x.getCell([0]).toDouble();
            final py = x.getCell([1]).toDouble();
            return (px - 1.0) * (px - 1.0) + (py - 2.0) * (py - 2.0);
          }, x0Bowl);
          blackhole(r);
          r.x.dispose();
        });
      });
    },
    config: CriterionConfig(
      generateHtmlReport: true,
      exportJson: true,
      reportDir: 'benchmark/report/operations',
    ),
  );
}
