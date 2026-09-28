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

/// High-performance symbolic computer algebra system (CAS) for Dart,
/// powered by native SymEngine (C++) and FLINT (exact number theory & algebra).
///
/// Features:
/// - Expression trees ([Expr], [Symbol], [Integer], [Real], [Rational])
/// - Symbolic calculus ([diff], [expand], [subs])
/// - Vectorized numerical evaluation over `NDArray` via [lambdify]
/// - Exact univariate polynomial algebra and factorization over Q[x] via [FlintRationalPoly]
library;

export 'src/expr.dart';
export 'src/lambdify.dart';
export 'src/matrix.dart';
export 'src/ndarray_integration.dart';
export 'src/optimizer.dart';
export 'src/poly.dart';
