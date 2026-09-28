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

/// Dynamic JIT kernel compilation and operator fusion engine for GPU computing.
///
/// Enables fusing complex sequences of elementwise operations (e.g. $y = \text{silu}(a \cdot x + b)$)
/// into a single, high-performance GPU compute shader pass without intermediate VRAM allocations.
library;

export 'src/backend/wgsl/kernel_fusion.dart';
export 'src/backend/wgsl/jit_compiler.dart';
