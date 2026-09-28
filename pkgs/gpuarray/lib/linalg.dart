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

/// Linear algebra decompositions, solvers, norms, and tensor contractions for
/// [GpuArray] tensors.
///
/// Includes matrix products ([matmul], [dot], [vdot], [multiDot]),
/// decompositions ([svd], [svdvals], [qr], [cholesky], [eigh], [eigvalsh],
/// [eig], [eigvals], [lu], [luFactor], [luSolve]), solvers and invariants
/// ([solve], [inv], [pinv], [det], [slogdet], [matrixPower], [matrixRank],
/// [norm], [cond], [trace], [diagonal]), and tensor contractions ([einsum],
/// [tensordot], [kron], [inner], [outer], [cross]).
library;

export 'src/linalg/linalg.dart';
