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

import 'package:gpuarray/gpuarray.dart';
import 'package:test/test.dart';

void main() {
  group('Linear Algebra Solvers & Matrix Properties (F8)', () {
    test('solve and luSolve solve linear systems with out: parameter', () {
      final coefficients = GpuArray.fromList(
        <double>[3.0, 1.0, 1.0, 2.0],
        [2, 2],
        DType.float64,
      );
      final ordinates = GpuArray.fromList(
        <double>[9.0, 8.0],
        [2],
        DType.float64,
      );
      final outSolution = GpuArray.zeros([2], DType.float64);
      try {
        final solution = solve(coefficients, ordinates, out: outSolution);
        expect(identical(solution, outSolution), isTrue);
        final values = solution.toList().cast<double>();
        expect(values[0], closeTo(2.0, 1e-10));
        expect(values[1], closeTo(3.0, 1e-10));

        final factor = luFactor(coefficients);
        try {
          final luSolution = luSolve(factor.lu, factor.pivots, ordinates);
          try {
            final luValues = luSolution.toList().cast<double>();
            expect(luValues[0], closeTo(2.0, 1e-10));
            expect(luValues[1], closeTo(3.0, 1e-10));
          } finally {
            luSolution.dispose();
          }
        } finally {
          factor.dispose();
        }
      } finally {
        outSolution.dispose();
        ordinates.dispose();
        coefficients.dispose();
      }
    });

    test('inv and pinv invert matrices with out: parameter', () {
      final matrix = GpuArray.fromList(
        <double>[4.0, 7.0, 2.0, 6.0],
        [2, 2],
        DType.float64,
      );
      final outInverse = GpuArray.zeros([2, 2], DType.float64);
      try {
        final inverse = inv(matrix, out: outInverse);
        expect(identical(inverse, outInverse), isTrue);
        final values = inverse.toList().cast<double>();
        expect(values[0], closeTo(0.6, 1e-10));
        expect(values[1], closeTo(-0.7, 1e-10));
        expect(values[2], closeTo(-0.2, 1e-10));
        expect(values[3], closeTo(0.4, 1e-10));

        final pseudoInverse = pinv(matrix);
        try {
          final pinvValues = pseudoInverse.toList().cast<double>();
          for (var i = 0; i < 4; i++) {
            expect(pinvValues[i], closeTo(values[i], 1e-10));
          }
        } finally {
          pseudoInverse.dispose();
        }
      } finally {
        outInverse.dispose();
        matrix.dispose();
      }
    });

    test('det and slogdet compute determinant and sign/logabsdet', () {
      final matrix = GpuArray.fromList(
        <double>[1.0, 2.0, 3.0, 4.0],
        [2, 2],
        DType.float64,
      );
      final outDet = GpuArray.zeros([], DType.float64);
      try {
        final determinant = det(matrix, out: outDet);
        expect(identical(determinant, outDet), isTrue);
        expect(determinant.scalar as double, closeTo(-2.0, 1e-10));

        final slog = slogdet(matrix);
        try {
          expect(slog.sign.scalar as double, closeTo(-1.0, 1e-10));
          expect(
            slog.logabsdet.scalar as double,
            closeTo(math.log(2.0), 1e-10),
          );
        } finally {
          slog.dispose();
        }
      } finally {
        outDet.dispose();
        matrix.dispose();
      }
    });

    test('matrixPower, matrixRank, norm, and cond', () {
      final matrix = GpuArray.fromList(
        <double>[2.0, 0.0, 0.0, 4.0],
        [2, 2],
        DType.float64,
      );
      try {
        final cubed = matrixPower(matrix, 3);
        final inverted = matrixPower(matrix, -1);
        final rank = matrixRank(matrix);
        final frobeniusNorm = norm(matrix, ord: NormOrd.frobenius);
        final spectralNorm = norm(matrix, ord: NormOrd.l2);
        final conditionNumber = cond(matrix, ord: NormOrd.l2);
        try {
          expect(cubed.toList(), equals(<double>[8.0, 0.0, 0.0, 64.0]));
          expect(inverted.toList(), equals(<double>[0.5, 0.0, 0.0, 0.25]));
          expect(rank.scalar, equals(2));
          expect(
            frobeniusNorm.scalar as double,
            closeTo(math.sqrt(20.0), 1e-10),
          );
          expect(spectralNorm.scalar as double, closeTo(4.0, 1e-10));
          expect(conditionNumber.scalar as double, closeTo(2.0, 1e-10));
        } finally {
          cubed.dispose();
          inverted.dispose();
          rank.dispose();
          frobeniusNorm.dispose();
          spectralNorm.dispose();
          conditionNumber.dispose();
        }
      } finally {
        matrix.dispose();
      }
    });

    test('norm supports vector p-norms via NormOrd', () {
      final vector = GpuArray.fromList(<double>[3.0, -4.0], [2], DType.float64);
      try {
        final l2 = norm(vector, ord: NormOrd.l2);
        final l1 = norm(vector, ord: NormOrd.l1);
        final linf = norm(vector, ord: NormOrd.infinity);
        try {
          expect(l2.scalar as double, closeTo(5.0, 1e-10));
          expect(l1.scalar as double, closeTo(7.0, 1e-10));
          expect(linf.scalar as double, closeTo(4.0, 1e-10));
        } finally {
          l2.dispose();
          l1.dispose();
          linf.dispose();
        }
      } finally {
        vector.dispose();
      }
    });

    test('lstsq solves least-squares system and disposes record', () {
      final coefficients = GpuArray.fromList(
        <double>[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        [3, 2],
        DType.float64,
      );
      final ordinates = GpuArray.fromList(
        <double>[1.0, 2.0, 3.0],
        [3],
        DType.float64,
      );
      try {
        final result = lstsq(coefficients, ordinates);
        try {
          expect(result.solution.shape, equals(<int>[2]));
          expect(result.rank, equals(2));
          expect(result.singularValues.shape, equals(<int>[2]));
          final sol = result.solution.toList().cast<double>();
          expect(sol[0], closeTo(1.0, 1e-6));
          expect(sol[1], closeTo(2.0, 1e-6));
        } finally {
          result.dispose();
        }
        expect(result.solution.isDisposed, isTrue);
        expect(result.residuals.isDisposed, isTrue);
        expect(result.singularValues.isDisposed, isTrue);
      } finally {
        ordinates.dispose();
        coefficients.dispose();
      }
    });
  });
}
