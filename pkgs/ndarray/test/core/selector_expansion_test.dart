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
import 'package:ndarray/ndarray.dart';

void main() {
  group('Operator [] and []= Selector Expansion Tests (Section 4.1)', () {
    test('mixed slice and integer index selection and assignment', () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [3, 2],
          DType.float64,
        );

        // a[Slice(0, 2), 1] extracts column 1 of rows 0 and 1
        final colView = a[[const Slice(start: 0, stop: 2), 1]];
        expect(colView.shape, [2]);

        // Assign scalar via coordinate list selector
        a[[0, 1]] = 99.0;
        expect(a.getCell([0, 1]), 99.0);

        // Assign scalar via integer row index selector
        a[1] = 50.0;
        expect(a.getCell([1, 0]), 50.0);
      });
    });

    test('multi-dimensional selection specification processing', () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
          [3, 4],
          DType.int32,
        );

        // Target individual cell via multi-dimensional index specification
        a[[2, 3]] = 999;
        expect(a.getCell([2, 3]), 999);
      });
    });

    test('NDArray int64 fancy index selector support', () {
      NDArray.scope(() {
        final a = NDArray.fromList([100, 200, 300, 400], [4], DType.int64);
        final idx = NDArray.fromList([0, 2], [2], DType.int64);
        a[idx] = 999;
        expect(a.toList(), [999, 200, 999, 400]);
      });
    });

    test('scalar coercion and scalar NDArray assignment', () {
      NDArray.scope(() {
        final floatArr = NDArray.zeros([2, 2], DType.float64);
        // Coerce int literal 42 to double on Float64 array
        floatArr[[0, 1]] = 42;
        expect(floatArr.getCell([0, 1]), 42.0);

        // Coerce single-element / scalar NDArray as RHS
        final scalarND = NDArray.scalar(88.0, dtype: DType.float64);
        floatArr[[1, 0]] = scalarND;
        expect(floatArr.getCell([1, 0]), 88.0);
      });
    });

    test('error handling for invalid selector dimension bounds and types', () {
      NDArray.scope(() {
        final a = NDArray.zeros([2, 2], DType.float64);

        // Too many selectors for array rank
        expect(
          () => a[[const Slice.all(), const Slice.all(), const Slice.all()]] =
              1.0,
          throwsArgumentError,
        );

        // Heterogeneous invalid list elements
        expect(
          () =>
              a[[
                [1, 'invalid'],
              ]],
          throwsArgumentError,
        );
      });
    });
  });
}
