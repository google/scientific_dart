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

import 'package:ndarray/ndarray.dart' hide exp, sin;
import 'package:symbolic_dart/symbolic_dart.dart';

int factorial(int n) => n <= 1 ? 1 : n * factorial(n - 1);

void main() {
  print('--- Symbolic Maclaurin Series of f(x) = exp(x)*sin(x) ---');
  final x = Symbol('x');
  final f = exp(x) * sin(x);

  print('Exact f(x) = $f');

  // Compute Maclaurin series up to order N = 5
  // P(x) = \sum_{k=0}^N [ d^k f / dx^k (0) ] / k! * x^k
  var currentDeriv = f;
  final coeffs = <({int numerator, int denominator})>[];

  for (var k = 0; k <= 5; k++) {
    final derivZero = currentDeriv.subs({x: 0}).asDouble;
    final intDeriv = derivZero.round();

    // Store rational coefficient intDeriv / k!
    final kFact = factorial(k);
    final gcd = BigInt.from(intDeriv).abs().gcd(BigInt.from(kFact)).toInt();
    final num = intDeriv ~/ gcd;
    final den = kFact ~/ gcd;
    coeffs.add((numerator: num, denominator: den));

    // Next derivative
    currentDeriv = currentDeriv.diff(x);
  }

  final taylorPoly = FlintRationalPoly.fromRationalCoefficients(coeffs);
  print('Maclaurin Polynomial P_5(x) (FLINT exact) = $taylorPoly');

  // Factor the exact Taylor polynomial over Q[x]
  final fac = taylorPoly.factor();
  print(
    'Factorized P_5(x) = (${fac.content.numerator}/${fac.content.denominator})'
    ' * ${fac.factors.map((f) => '(${f.factor})^${f.exponent}').join(' * ')}\n',
  );

  // Compare Exact f(x) vs Taylor P_5(x) over NDArray interval [-0.6, 0.6]
  final xGrid = NDArray.fromList(
    [-0.6, -0.3, 0.0, 0.3, 0.6],
    [5],
    DType.float64,
  );
  final exactLambda = f.lambdify([x]);
  final polyLambda = taylorPoly.toExpr(x).lambdify([x]);

  final exactVals = exactLambda.callArray([xGrid]);
  final polyVals = polyLambda.callArray([xGrid]);

  print('   x   |  Exact f(x)  | Taylor P_5(x) |  Error |f - P_5|');
  print('-------+--------------+---------------+-----------------');
  for (var i = 0; i < xGrid.shape[0]; i++) {
    final xi = xGrid.getCell([i]);
    final ex = exactVals.getCell([i]);
    final ap = polyVals.getCell([i]);
    final err = (ex - ap).abs();
    print(
      '${xi.toStringAsFixed(1).padLeft(6)} | '
      '${ex.toStringAsFixed(5).padLeft(12)} | '
      '${ap.toStringAsFixed(5).padLeft(13)} | '
      '${err.toStringAsExponential(2).padLeft(15)}',
    );
  }
}
