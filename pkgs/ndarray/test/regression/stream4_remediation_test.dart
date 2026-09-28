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

import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
  group('Stream 4 Remediation Stats Tests', () {
    test('std and nanstd on 0D / axis null without data[0]', () {
      final a = NDArray.scalar(10.0, dtype: DType.float64);
      final s = std(a);
      expect(s.scalar, closeTo(0.0, 1e-6));
      s.dispose();

      final a2 = NDArray.fromList([2.0, 4.0], [2], DType.float64);
      final outStd = NDArray<Float64>.scalar(0.0, dtype: DType.float64);
      std(a2, out: outStd);
      expect(outStd.scalar, closeTo(1.0, 1e-6));
      outStd.dispose();
      a.dispose();
      a2.dispose();
    });

    test('cov and corrcoef with optional out parameter', () {
      final m = NDArray.fromList(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [2, 3],
        DType.float64,
      );
      final outCov = NDArray<Float64>.zeros([2, 2], DType.float64);
      final resCov = cov(m, out: outCov);
      expect(resCov, same(outCov));
      expect(outCov.getCell([0, 0]), closeTo(1.0, 1e-6));

      final outCorr = NDArray<Float64>.zeros([2, 2], DType.float64);
      final resCorr = corrcoef(m, out: outCorr);
      expect(resCorr, same(outCorr));
      expect(outCorr.getCell([0, 1]), closeTo(1.0, 1e-6));

      m.dispose();
      outCov.dispose();
      outCorr.dispose();
    });
  });

  group('Stream 4 Remediation Sorting Tests', () {
    test('partition and argpartition 0D fallbacks with out', () {
      final a = NDArray.scalar(42, dtype: DType.int32);
      final outP = NDArray.scalar(0, dtype: DType.int32);
      final p = partition(a, 0, out: outP);
      expect(p, same(outP));
      expect(outP.scalar, 42);

      final outAP = NDArray.scalar(99, dtype: DType.int64);
      final ap = argpartition(a, 0, out: outAP);
      expect(ap, same(outAP));
      expect(outAP.scalar, 0);

      a.dispose();
      outP.dispose();
      outAP.dispose();
    });

    test('argpartition uniqueK empty branch', () {
      final a = NDArray.fromList([5, 2, 8, 1], [4], DType.int32);
      final ap = argpartition(a, <int>[]);
      expect(ap.toList(), equals([0, 1, 2, 3]));
      a.dispose();
      ap.dispose();
    });

    test(
      'argpartition and searchsorted boolean branch without List<bool> cast',
      () {
        final aBool = NDArray.fromList(
          [true, false, true, false],
          [4],
          DType.boolean,
        );
        final ap = argpartition(aBool, 1);
        // false < true
        expect(ap.getCell([0]), isIn([1, 3]));
        expect(ap.getCell([1]), isIn([1, 3]));
        expect(ap.getCell([2]), isIn([0, 2]));
        expect(ap.getCell([3]), isIn([0, 2]));

        final vBool = NDArray.fromList([false, true], [2], DType.boolean);
        final ss = searchsorted(aBool, vBool);
        expect(ss.size, 2);

        aBool.dispose();
        ap.dispose();
        vBool.dispose();
        ss.dispose();
      },
    );
  });

  group('Stream 4 Remediation Random & Meshes & Splitting Tests', () {
    test('choice with optional out parameter', () {
      final a = NDArray.fromList([10, 20, 30], [3], DType.int32);
      final outChoice = NDArray.zeros([2], DType.int32);
      final res = choice(a, size: [2], out: outChoice, seed: 123);
      expect(res, same(outChoice));
      expect(outChoice.getCell([0]), isIn([10, 20, 30]));
      a.dispose();
      outChoice.dispose();
    });

    test('mgrid with optional out parameter', () {
      final outGrid = NDArray<Float64>.zeros([2, 3, 1], DType.float64);
      final res = mgrid([GridRange(0, 3), GridRange(0, 1)], out: outGrid);
      expect(res, same(outGrid));
      expect(outGrid.getCell([0, 0, 0]), 0.0);
      expect(outGrid.getCell([0, 2, 0]), 2.0);
      outGrid.dispose();
    });

    test('hsplit, vsplit, dsplit with optional out parameter', () {
      final a2 = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
      final outV0 = NDArray.zeros([1, 2], DType.int32);
      final outV1 = NDArray.zeros([1, 2], DType.int32);
      final resV = vsplit(a2, 2, out: [outV0, outV1]);
      expect(resV[0], same(outV0));
      expect(resV[1], same(outV1));
      expect(outV0.toList(), equals([1, 2]));
      expect(outV1.toList(), equals([3, 4]));

      a2.dispose();
      outV0.dispose();
      outV1.dispose();
    });
  });
}
