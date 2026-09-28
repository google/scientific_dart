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

/// Counter-based pseudo-random number generation (Philox 4x32-10) for
/// [GpuArray] tensors.
///
/// Provides [Philox4x32Engine], [RandomState], and top-level sampling functions
/// ([seed], [rand], [randn], [randint], [uniform], [normal], [standardNormal],
/// [standard_normal], [exponential], [choice], [permutation], [shuffle]).
library;

export 'src/random/random.dart'
    show
        Philox4x32Engine,
        RandomState,
        choice,
        defaultRng,
        exponential,
        normal,
        permutation,
        rand,
        randint,
        randn,
        seed,
        shuffle,
        standardNormal,
        standard_normal,
        uniform;
